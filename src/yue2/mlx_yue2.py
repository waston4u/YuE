"""YuE2 Mixture-of-Transformers on MLX — the Apple Silicon native engine.

Weights load straight from the released ``model.safetensors`` (names map 1:1);
each Linear may be groupwise-quantized (int4/int8) during load so the bf16
source is dropped tensor-by-tensor — peak load memory stays ~one tensor.

Only the two inference paths exist here: the pure-AR decode path and the
manual per-layer NAR velocity path used by ``nar.CachedNAR``. The training
``ar_mask`` merge is torch-side only.
"""
from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    """GQA attention with per-head qk-norm applied before RoPE."""

    def __init__(self, hidden: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, eps: float, theta: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.theta = theta
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)

    def project_qkv(self, x, offset: int = 0):
        """Q [B,H,T,D], K/V [B,Hkv,T,D] — normed then RoPE'd (θ=1e6, NeoX).

        mx.fast.rope reads the sequence index from dim -2, so tensors are
        already in [B,H,T,D] layout when RoPE is applied."""
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.head_dim, traditional=False,
                         base=self.theta, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.head_dim, traditional=False,
                         base=self.theta, scale=1.0, offset=offset)
        return q, k, v

    def __call__(self, x, cache=None, offset: int = 0, causal: bool = False):
        B, T, _ = x.shape
        q, k, v = self.project_qkv(x, offset)
        if cache is not None:
            k, v = cache.update(k, v)
        mask = "causal" if causal else None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale,
                                                   mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """MoT layer — dual attention + dual MLP. AR and NAR paths are invoked
    separately at inference; there is no runtime token-type merge."""

    def __init__(self, hidden: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, intermediate: int, eps: float, theta: float):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = Attention(hidden, num_heads, num_kv_heads,
                                   head_dim, eps, theta)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = MLP(hidden, intermediate)
        self.nar_input_layernorm = RMSNorm(hidden, eps)
        self.nar_self_attn = Attention(hidden, num_heads, num_kv_heads,
                                       head_dim, eps, theta)
        self.nar_pre_mlp_layernorm = RMSNorm(hidden, eps)
        self.nar_mlp = MLP(hidden, intermediate)

    def ar(self, x, cache, offset: int, causal: bool):
        x = x + self.self_attn(self.input_layernorm(x), cache=cache,
                               offset=offset, causal=causal)
        return x + self.mlp(self.post_attention_layernorm(x))

    def nar(self, x, ar_kv, offset: int):
        """NAR velocity pass: NAR projections, attention over
        [ar_k ‖ nar_k] (non-causal), NAR o_proj + MLP."""
        q, k, v = self.nar_self_attn.project_qkv(
            self.nar_input_layernorm(x), offset)
        B, T = x.shape[:2]
        if ar_kv is not None:
            k = mx.concatenate([ar_kv[0], k], axis=2)
            v = mx.concatenate([ar_kv[1], v], axis=2)
        h = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.nar_self_attn.scale)
        h = h.transpose(0, 2, 1, 3).reshape(B, T, -1)
        x = x + self.nar_self_attn.o_proj(h)
        return x + self.nar_mlp(self.nar_pre_mlp_layernorm(x))


class _Empty(nn.Module):
    """Param-free slot so ``mlp.0``/``mlp.2`` indices match the checkpoint
    (a SiLU sits at index 1 in the released Sequential)."""


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = [nn.Linear(frequency_embedding_size, hidden_size),
                    _Empty(),
                    nn.Linear(hidden_size, hidden_size)]
        self.frequency_embedding_size = frequency_embedding_size

    def __call__(self, t):
        half = self.frequency_embedding_size // 2
        freqs = mx.exp(-math.log(10000.0)
                       * mx.arange(half, dtype=mx.float32) / half)
        args = t.astype(mx.float32)[:, None] * freqs[None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        h = self.mlp[0](emb.astype(self.mlp[0].weight.dtype))
        return self.mlp[2](nn.silu(h))


class AudioPositionEmbedding(nn.Module):
    """Non-learnable sinusoidal PE — the ``pe`` table ships in the weights."""

    def __init__(self, max_frames: int, hidden_size: int):
        super().__init__()
        self.pe = mx.zeros((max_frames, hidden_size))

    def __call__(self, position_ids):
        return self.pe[position_ids]


class KVCache:
    """Per-layer bounded K/V store: [1, n_kv, cap, D] buffers written in
    place — concatenate-per-token would copy the whole history every step
    (O(S²) traffic, ~900 MB/token at 8k context).

    Buffers start at ``initial_len`` and grow by doubling up to
    ``max_len`` — a full ``max_len`` reservation (~3.6 GiB for a 16k
    context × 28 layers × CFG) is committed whether the render needs 4k
    or 9k tokens. Consumers only ever use the views this call returns, so
    a grow never leaves a stale view behind."""

    def __init__(self, max_len: int, n_kv_heads: int, head_dim: int, dtype,
                 initial_len: int | None = None):
        self.max_len = max_len
        capacity = max_len if initial_len is None else min(initial_len, max_len)
        self.keys = mx.zeros((1, n_kv_heads, capacity, head_dim), dtype=dtype)
        self.values = mx.zeros((1, n_kv_heads, capacity, head_dim), dtype=dtype)
        self.seen = 0

    def update(self, k, v):
        """k/v: [1, n_kv, T, D] → views over the valid prefix."""
        end = self.seen + k.shape[2]
        if end > self.max_len:
            raise ValueError(
                f"KV cache capacity {self.max_len} exceeded by {end}")
        if end > self.keys.shape[2]:
            capacity = min(self.max_len, max(end, self.keys.shape[2] * 2))
            keys = mx.zeros((1, *self.keys.shape[1:2], capacity,
                             self.keys.shape[3]), dtype=self.keys.dtype)
            values = mx.zeros((1, *self.values.shape[1:2], capacity,
                               self.values.shape[3]), dtype=self.values.dtype)
            keys[:, :, :self.seen] = self.keys[:, :, :self.seen]
            values[:, :, :self.seen] = self.values[:, :, :self.seen]
            self.keys, self.values = keys, values
        self.keys[:, :, self.seen:end] = k
        self.values[:, :, self.seen:end] = v
        self.seen = end
        return self.keys[:, :, :end], self.values[:, :, :end]


def make_caches(num_layers: int, max_len: int, n_kv: int, head_dim: int,
                dtype=mx.bfloat16, initial_len: int | None = None):
    return [KVCache(max_len, n_kv, head_dim, dtype, initial_len)
            for _ in range(num_layers)]


class Backbone(nn.Module):
    """embed → N MoT layers → norm (checkpoint prefix ``model.``)."""

    def __init__(self, config: dict):
        super().__init__()
        hidden = config["hidden_size"]
        self.embed_tokens = nn.Embedding(config["vocab_size"], hidden)
        self.layers = [
            DecoderLayer(hidden, config["num_attention_heads"],
                         config["num_key_value_heads"], config["head_dim"],
                         config["intermediate_size"], config["rms_norm_eps"],
                         config["rope_theta"])
            for _ in range(config["num_hidden_layers"])]
        self.norm = RMSNorm(hidden, config["rms_norm_eps"])


class MlxYuE2Model(nn.Module):
    """AR causal LM + NAR flow-matching velocity field, MoT weights."""

    def __init__(self, config: dict):
        super().__init__()
        hidden = config["hidden_size"]
        self.config = config
        self.hidden_size = hidden
        self.latent_dim = config["latent_dim"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.model = Backbone(config)
        self.lm_head = nn.Linear(hidden, config["vocab_size"], bias=False)
        self.llm2vae = nn.Linear(hidden, config["latent_dim"], bias=True)
        self.vae2llm = nn.Linear(config["latent_dim"], hidden, bias=True)
        self.time_embedder = TimestepEmbedder(hidden)
        self.latent_pos_embed = AudioPositionEmbedding(
            config["max_latent_frames"], hidden)
        self._dtype = mx.bfloat16

    # ── AR path ──────────────────────────────────────────────────────
    def __call__(self, ids, caches=None, offset: int = 0,
                 logits_last_only: bool = True):
        """ids [1,T] int → logits [1,T|1,vocab]. Causal; caches optional."""
        x = self.model.embed_tokens(ids)
        causal = ids.shape[1] > 1
        for layer, cache in zip(self.model.layers,
                                caches or [None] * len(self.model.layers)):
            x = layer.ar(x, cache, offset, causal)
        x = self.model.norm(x)
        x = x[:, -1:, :] if logits_last_only else x
        return self.lm_head(x)

    # ── NAR pieces (driven by mlx_nar.CachedNAR) ─────────────────────
    def shift_t(self, raw_t: float) -> float:
        shift = self.config.get("timestep_shift", 1.0)
        t_sig = 1.0 / (1.0 + math.exp(-raw_t))
        return shift * t_sig / (1 + (shift - 1) * t_sig)


#: Linears that dominate per-token bandwidth — quantized when bits is set.
_QUANT_PREFIXES = ("self_attn.", "nar_self_attn.", "mlp.", "nar_mlp.")


def load_weights(model_dir, bits: int | None = 4, group_size: int = 64,
                 progress=None) -> MlxYuE2Model:
    """Load the released safetensors into an MLX model.

    Streams tensors one at a time via safetensors' MLX support: linear
    weights under the big prefixes (plus lm_head) are quantized to ``bits``
    and the bf16 source dropped immediately, so peak memory stays ~one
    tensor (~0.8 GB) regardless of the 7.26 GB file.
    """
    model_dir = Path(model_dir)
    config = _read_config(model_dir)
    model = MlxYuE2Model(config)
    st_path = model_dir / "model.safetensors"
    if not st_path.is_file():
        raise FileNotFoundError(f"No model.safetensors in {model_dir}")

    flat = {}
    tensors = _safetensors_entries(st_path)
    total = len(tensors)
    for i, (name, w) in enumerate(tensors):
        parent, _, leaf = name.rpartition(".")
        if bits and leaf == "weight" and _quantizable(parent):
            wq, scales, qb = mx.quantize(w, group_size=group_size,
                                         bits=bits)
            flat[parent + ".weight"] = wq
            flat[parent + ".scales"] = scales
            flat[parent + ".biases"] = qb
        else:
            flat[name] = w
        mx.eval(flat[name])
        if progress and i % 64 == 0:
            progress(i, total)

    if bits:
        # Swap target Linears for QuantizedLinear shells before update —
        # their params (weight/scales/biases) differ from Linear's.
        nn.quantize(model, group_size=group_size, bits=bits,
                    class_predicate=lambda path, m:
                    isinstance(m, nn.Linear) and _quantizable(path))
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten(flat))
    mx.eval(model.parameters())
    return model


def _quantizable(path: str) -> bool:
    """Per-token-bandwidth linears only — the tiny aux heads (llm2vae,
    vae2llm, time_embedder) keep bf16 so the NAR field stays exact."""
    leaf = path.split(".")[-1]
    return leaf.endswith("_proj") or path == "lm_head"


def _safetensors_entries(path: Path):
    """Stream bf16 tensors straight from the safetensors byte buffer.

    safetensors' framework="mlx" reader can't decode bf16, and mx.load
    would materialize the full 7.26 GB at once — so we read the JSON
    header ourselves and reinterpret each raw extent as uint16 → view
    bfloat16, one tensor at a time."""
    import json
    import struct
    import numpy as np
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        out = []
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if meta["dtype"] != "BF16":
                raise ValueError(
                    f"{name}: expected BF16 weights, found {meta['dtype']}")
            begin, end = meta["data_offsets"]
            f.seek(base + begin)
            raw = f.read(end - begin)
            w = mx.array(np.frombuffer(raw, np.uint16)) \
                .view(mx.bfloat16).reshape(meta["shape"])
            out.append((name, w))
    return out


def _read_config(model_dir: Path) -> dict:
    import json
    cfg = json.loads((model_dir / "config.json").read_text())
    needed = ["hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "intermediate_size",
              "vocab_size", "rms_norm_eps", "rope_theta", "latent_dim",
              "max_latent_frames"]
    missing = [k for k in needed if k not in cfg]
    if missing:
        raise ValueError(f"config.json missing {missing}")
    return cfg
