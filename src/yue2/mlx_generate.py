"""MLX autoregressive decode loop — mirrors sampling.generate_tokens.

Identical contract: same prefix/cancel/callback semantics, same CPU-side
distribution() + torch.multinomial (deterministic seeds), same timing dict.
Only the model forward is MLX.
"""
from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import torch

from .mlx_yue2 import make_caches
from .protocol import ABC_END, MUSIC_END, CONTEXT
from .sampling import distribution, allowed_mask


def _to_torch_row(logits) -> torch.Tensor:
    """[1,1,V] mlx logits → [1,V] CPU float32 torch row."""
    row = logits[0, -1].astype(mx.float32)
    return torch.from_numpy(np.asarray(row))[None]


def generate_tokens_mlx(model, prefix, sampling, seed, phase,
                        negative=None, cfg_scale=1.0, legacy_off=False,
                        cancelled=None, on_token=None):
    cfg = model.config
    if len(prefix) + sampling.max_tokens > CONTEXT:
        raise ValueError("Prefix + requested generation budget exceeds 24576; no implicit truncation")
    if cfg_scale != 1 and negative is None:
        raise ValueError("CFG requires a negative prefix")
    if negative is not None and len(negative) + sampling.max_tokens > CONTEXT:
        raise ValueError("Negative prefix + generation budget exceeds context")
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before prefill")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    layers = cfg["num_hidden_layers"]

    def prefill(ids):
        caches = make_caches(layers, len(ids) + sampling.max_tokens,
                             n_kv, head_dim,
                             initial_len=len(ids) + 2048)
        logits = model(mx.array([ids]), caches, offset=0)
        mx.eval([c.keys for c in caches], [c.values for c in caches], logits)
        return logits[:, -1:, :], caches

    start = time.perf_counter()
    conditional, positive = prefill(prefix)
    unconditional = negative_cache = None
    if cfg_scale != 1.0:
        unconditional, negative_cache = prefill(negative)
    prefill_seconds = time.perf_counter() - start

    mask_shape = torch.empty(1, cfg["vocab_size"], dtype=torch.float32)
    allowed = allowed_mask(mask_shape, phase)
    history, first, eos = [], None, False
    end = ABC_END if phase == "abc" else MUSIC_END
    pos = len(prefix)

    for step in range(sampling.max_tokens):
        if cancelled is not None and cancelled():
            raise InterruptedError(f"Cancelled during {phase}")
        merged = conditional if cfg_scale == 1.0 else \
            unconditional + cfg_scale * (conditional - unconditional)
        logits_row = _to_torch_row(merged)
        scores = distribution(logits_row, sampling, history, step, phase,
                              legacy_off, allowed=allowed)
        if sampling.temperature == 0:
            next_id = scores.argmax(-1, keepdim=True)
        else:
            probabilities = scores.softmax(-1)
            next_id = torch.multinomial(probabilities, 1, generator=generator)
        token = int(next_id.item())
        if first is None:
            first = time.perf_counter() - start
        if on_token is not None:
            on_token(phase, token)
        if token == end:
            eos = True
            break
        history.append(token)
        if step + 1 < sampling.max_tokens:
            ids = mx.array([[token]])
            conditional = model(ids, positive, offset=pos)
            if negative_cache is not None:
                unconditional = model(ids, negative_cache, offset=pos)
            pos += 1

    seconds = time.perf_counter() - start
    count = len(history) + int(eos)
    timing = {"seconds": seconds, "prefill_seconds": prefill_seconds,
              "ttft_seconds": first, "output_tokens": count,
              "content_tokens": len(history),
              "output_tps": count / seconds if seconds else 0,
              "prefix_tokens": len(prefix),
              "cfg_branches": 1 if cfg_scale == 1 else 2,
              "execution": "mlx", "attention": "sdpa"}
    return history, timing, not eos
