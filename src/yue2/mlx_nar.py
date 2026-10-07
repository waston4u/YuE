"""MLX acoustic flow matching — port of nar.CachedNAR.

Same protocol: one AR prefill per original chunk, per-layer truncated K/V,
NAR velocity over [ar_k ‖ nar_k] non-causal attention, midpoint ODE with
adaptive substepping. ``song_chunks`` stays torch-side (it owns the seeded
CPU noise draw); only the model math is MLX.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Callable, Sequence

import mlx.core as mx
import numpy as np
import torch

from .nar import Chunk, song_chunks
from .protocol import CONTEXT


def _logit(t: float) -> float:
    """torch.logit(x).clamp(-20, 20) in float64 — the first midpoint step
    hits t=1.0 exactly, where Python's log(t/(1-t)) raises."""
    if t >= 1.0:
        return 20.0
    if t <= 0.0:
        return -20.0
    return min(20.0, max(-20.0, math.log(t / (1.0 - t))))


#: Velocity evaluations per solver step — drives the progress total and
#: the cost/accuracy tradeoff the caller picks via ``method``.
_EVALS_PER_STEP = {"midpoint": 2, "heun": 2, "rk4": 4, "ab2": 1}


class MlxCachedNAR:
    """One original acoustic chunk; AR prefix KV is invariant during the ODE."""

    def __init__(self, model, chunk: Chunk, **kwargs):
        self.model, self.chunk = model, chunk
        noise = np.asarray(chunk.noise)
        if noise.ndim != 2 or noise.shape[1] != 64 or len(noise) < 1:
            raise ValueError("Expected nonempty acoustic noise [frames,64]")
        if not np.isfinite(noise).all():
            raise ValueError("Acoustic noise contains non-finite values")
        # The torch path casts the ODE state to the weight dtype (bf16).
        self.noise = mx.array(noise).astype(mx.bfloat16)
        self.ar_length, self.nar_length = len(chunk.ar_tokens), len(noise) + 2
        if self.ar_length < 1 or min(chunk.ar_tokens) < 0 \
                or max(chunk.ar_tokens) >= model.config["vocab_size"]:
            raise ValueError("AR prefix is empty or outside the model vocabulary")
        if self.ar_length + self.nar_length > model.config["max_position_embeddings"]:
            raise ValueError("Original acoustic chunk exceeds the model context")
        if chunk.nar_cond_end < 0:
            raise ValueError("nar_cond_end must be nonnegative")
        self.visible_length = (min(chunk.nar_cond_end, self.ar_length)
                               if chunk.nar_cond_end else self.ar_length)
        # NAR positions are the contiguous range [ar_length, ar_length+T) —
        # the RoPE offset covers them without materializing cos/sin.
        self.offset = self.ar_length
        local = mx.arange(self.nar_length)
        local = mx.minimum(local, model.config["max_latent_frames"] - 1)
        self.pos_emb = model.latent_pos_embed(local)[None]
        self.cache = []
        self._velocity_fn = None
        self._eval_progress = None
        self._prefill()

    def _velocity_core(self, state, shifted: float):
        """The compiled-able body: pure tensor math from state → velocity."""
        model = self.model
        x_nar = mx.pad(state, [(1, 1), (0, 0)])
        x = model.vae2llm(x_nar[None])
        t_row = mx.full((self.nar_length,), shifted)
        x = x + model.time_embedder(t_row)[None]
        x = x + self.pos_emb
        for layer, ar_kv in zip(model.model.layers, self.cache):
            x = layer.nar(x, ar_kv, self.offset)
        return model.llm2vae(model.model.norm(x))[0, 1:-1]

    def _prefill(self):
        """AR path over the token prefix; per-layer K/V truncated to the
        conditioned prefix length."""
        model = self.model
        ids = mx.array([self.chunk.ar_tokens])
        x = model.model.embed_tokens(ids)
        for layer in model.model.layers:
            q, k, v = layer.self_attn.project_qkv(
                layer.input_layernorm(x), offset=0)
            self.cache.append((k[:, :, :self.visible_length],
                               v[:, :, :self.visible_length]))
            h = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=layer.self_attn.scale, mask="causal")
            x = x + layer.self_attn.o_proj(
                h.transpose(0, 2, 1, 3).reshape(1, self.ar_length, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        mx.eval(self.cache, x)

    def velocity(self, state, raw_t: float):
        """v_theta(x_t, t): NAR positions = vae2llm(x) + time + pos, NAR
        projections, non-causal attention over [ar ‖ nar], llm2vae head.

        The 64 evals per chunk share one static shape, so the body is
        mx.compile'd once — fuses norm/rope/residual chains and removes
        per-eval Python graph construction."""
        if tuple(state.shape) != tuple(self.noise.shape):
            raise ValueError("ODE state shape changed")
        if self._velocity_fn is None:
            self._velocity_fn = mx.compile(self._velocity_core)
        out = self._velocity_fn(state, self.model.shift_t(raw_t))
        if self._eval_progress is not None:
            self._eval_progress()
        return out

    def solve(self, steps=32, cancelled: Callable[[], bool] | None = None,
              on_progress=None, method: str = "midpoint"):
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if method not in _EVALS_PER_STEP:
            raise ValueError(f"method must be one of {tuple(_EVALS_PER_STEP)}")
        state = self.noise
        dt = 1.0 / steps
        # Progress at velocity-eval granularity (substeps add evals) — the
        # bar advances every few seconds instead of per step.
        evals = {"n": 0}
        total_evals = int(steps) * _EVALS_PER_STEP[method]
        if on_progress is not None:
            def tick():
                evals["n"] += 1
                on_progress(min(evals["n"], total_evals), total_evals)
            self._eval_progress = tick
        try:
            prev_v = None  # ab2 history; unused by the other methods
            for step in range(steps):
                if cancelled is not None and cancelled():
                    raise InterruptedError(
                        "Cancelled during acoustic flow matching")
                state, prev_v = self._integrate(state, 1.0 - step * dt, dt,
                                                method, prev_v, cancelled)
        finally:
            self._eval_progress = None
        result = torch.from_numpy(np.asarray(state.astype(mx.float32)))
        if not torch.isfinite(result).all():
            raise FloatingPointError(
                "Acoustic flow matching produced non-finite latents")
        return result

    def _integrate(self, state, t, dt, method, prev_v, cancelled, depth=0):
        """One solver step with non-finite → halving substep recovery.
        Returns (state, prev_v); prev_v threads the ab2 history through
        substeps so the second half-step sees the first half's velocity."""
        new, prev_v = self._step_once(state, t, dt, method, prev_v, cancelled)
        if bool(mx.all(mx.isfinite(new)).item()):
            return new, prev_v
        if depth >= 3:
            raise FloatingPointError(
                "Acoustic flow matching produced non-finite latents")
        half, prev_v = self._integrate(state, t, dt / 2, method, prev_v,
                                       cancelled, depth + 1)
        return self._integrate(half, t - dt / 2, dt / 2, method, prev_v,
                               cancelled, depth + 1)

    def _step_once(self, state, t, dt, method, prev_v, cancelled):
        """One solver step at fixed dt — see _EVALS_PER_STEP for cost."""
        def v(x, tt):
            out = self.velocity(x, _logit(tt))
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            return out

        if method == "ab2":
            # Adams–Bashforth 2 — the 2nd-order multistep form (the
            # DPM-Solver++(2M) analogue for an ODE linear in t). First
            # step bootstraps with Euler.
            vel = v(state, t)
            if prev_v is None:
                return state - vel * dt, vel
            return state - (vel * 1.5 - prev_v * 0.5) * dt, vel
        if method == "heun":
            first = v(state, t)
            second = v(state - first * dt, t - dt)
            return state - (first + second) * (dt / 2), None
        if method == "rk4":
            k1 = v(state, t)
            k2 = v(state - k1 * (dt / 2), t - dt / 2)
            k3 = v(state - k2 * (dt / 2), t - dt / 2)
            k4 = v(state - k3 * dt, t - dt)
            return state - (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6), None
        # midpoint — the reference integrator.
        first = v(state, t)
        mid = state - first * (dt / 2)
        return state - v(mid, t - dt / 2) * dt, None

    def close(self):
        self.cache.clear()
        self.pos_emb = None


def synthesize(model, prefix: Sequence[int], codec: Sequence[int], seed: int,
               steps=32, context=CONTEXT, cancelled=None,
               on_progress: Callable[[int, int], None] | None = None,
               method: str = "midpoint", **kwargs):
    """Return CPU FP32 [frames,64] latents, solving original chunks serially.

    Same contract as nar.synthesize minus the torch-only knobs (attention
    backend, query tiling, AR offload — unified memory makes them moot).
    """
    chunks = song_chunks(prefix, codec, seed, context)
    output = []
    for chunk_index, chunk in enumerate(chunks):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before acoustic prefill")
        progress = None
        if on_progress is not None:
            def progress(completed, total):
                on_progress(chunk_index * total + completed,
                            total * len(chunks))
        engine = MlxCachedNAR(model, chunk)
        try:
            output.append(engine.solve(steps, cancelled,
                                       on_progress=progress, method=method))
        finally:
            engine.close()
    return torch.cat(output, dim=0)
