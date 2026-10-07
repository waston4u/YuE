"""MLX engine parity with the torch reference — same weights, same math.

A tiny random-init YuE2 built in both frameworks must produce matching
logits through prefill and KV-cached decode, and matching NAR velocity —
that is the gate that proves the port implements the same model.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

mlx = pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_unflatten  # noqa: E402

from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM  # noqa: E402
from yue2.mlx_yue2 import MlxYuE2Model, make_caches  # noqa: E402


def _tiny_config():
    return YuE2Config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128,
        vocab_size=512, rms_norm_eps=1e-6, rope_theta=1000000,
        max_position_embeddings=1024, latent_dim=64, max_latent_frames=256,
        timestep_shift=1.0)


def _pair():
    torch.manual_seed(0)
    tm = YuE2ForCausalLM(_tiny_config()).eval()
    flat = {k: mx.array(v.detach().float().numpy())
            for k, v in tm.state_dict().items()}
    mm = MlxYuE2Model(tm.config.to_dict())
    mm.update(tree_unflatten(flat))
    mx.eval(mm.parameters())
    return tm, mm


def test_forward_logits_match_torch():
    tm, mm = _pair()
    ids = torch.arange(24)[None] % 400
    with torch.inference_mode():
        ref = tm(ids).logits[0, -1].float()
    out = mm(mx.array(ids.numpy()))[0, -1]
    ref, out = ref.numpy(), np.asarray(out)
    assert np.abs(ref - out).max() < 0.02
    assert int(ref.argmax()) == int(out.argmax())


def test_kv_decode_matches_torch():
    tm, mm = _pair()
    ids = torch.arange(10)[None] % 400
    with torch.inference_mode():
        ref_logits = [tm(ids).logits[0, -1].float()]
        from yue2.modeling_yue2 import StaticKVCache
        cache = StaticKVCache(2, 1, 2, 64, 16, torch.float32, torch.device("cpu"))
        cache = tm(ids, past_key_values=cache, use_cache=True,
                   logits_to_keep=1).past_key_values
        for step in range(5):
            out = tm(torch.tensor([[50 + step]]), past_key_values=cache,
                     use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            ref_logits.append(out.logits[0, -1].float())
    caches = make_caches(2, 64, 2, 16)
    got = [mm(mx.array(ids.numpy()), caches, offset=0)[0, -1]]
    for step in range(5):
        got.append(mm(mx.array([[50 + step]]), caches, offset=10 + step)[0, -1])
    for i, (r, g) in enumerate(zip(ref_logits, got)):
        d = np.abs(r.numpy() - np.asarray(g)).max()
        assert d < 0.02, f"step {i} diverged: {d}"


def test_nar_velocity_matches_torch():
    tm, mm = _pair()
    from yue2.nar import CachedNAR, Chunk
    from yue2.mlx_nar import MlxCachedNAR
    torch.manual_seed(1)
    noise = torch.randn(6, 64)
    chunk = Chunk(ar_tokens=[3, 7, 42, 100], noise=noise)
    ref = CachedNAR(tm, chunk, "math")
    got = MlxCachedNAR(mm, chunk)
    state = noise.to(next(tm.parameters()).dtype)
    with torch.inference_mode():
        v_ref = ref.velocity(state, 1.234)
    v_got = got.velocity(got.noise, 1.234)
    d = np.abs(v_ref.float().numpy() - np.asarray(v_got)).max()
    assert d < 0.05, f"velocity diverged: {d}"


def test_weight_map_covers_every_tensor():
    """The real checkpoint must map 1:1 — every consumed name has a module
    parameter and vice versa (no silently dropped MoT half)."""
    import json, struct
    from pathlib import Path as P
    cache = P.home() / ".cache/huggingface/hub"
    st = next(cache.glob("models--m-a-p--YuE2-3B/snapshots/*/model.safetensors"),
              None)
    if st is None:
        pytest.skip("real weights not downloaded")
    with open(st, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    ckpt = {k for k in header if k != "__metadata__"}
    from yue2.mlx_yue2 import _quantizable
    expected = set()
    for k in ckpt:
        parent, _, leaf = k.rpartition(".")
        if leaf == "weight" and _quantizable(parent):
            expected.update((parent + ".weight", parent + ".scales",
                             parent + ".biases"))
        else:
            expected.add(k)
    cfg = json.loads((st.parent / "config.json").read_text())
    model = MlxYuE2Model(cfg)
    import mlx.nn as nn
    nn.quantize(model, group_size=64, bits=4,
                class_predicate=lambda p, m:
                isinstance(m, nn.Linear) and _quantizable(p))
    from mlx.utils import tree_flatten
    have = {k for k, _ in tree_flatten(model.parameters())}
    missing = expected - have
    extra = have - expected
    assert not missing, f"unconsumed tensors: {sorted(missing)[:10]}"
    assert not extra, f"params without weights: {sorted(extra)[:10]}"


def test_solvers_produce_finite_latents():
    """Every ODE method must integrate the same chunk to finite latents."""
    tm, mm = _pair()
    from yue2.nar import Chunk
    from yue2.mlx_nar import MlxCachedNAR, _EVALS_PER_STEP
    torch.manual_seed(2)
    noise = torch.randn(6, 64)
    chunk = Chunk(ar_tokens=[3, 7, 42, 100], noise=noise)
    for method in _EVALS_PER_STEP:
        engine = MlxCachedNAR(mm, chunk)
        latents = engine.solve(steps=3, method=method)
        engine.close()
        assert torch.isfinite(latents).all(), method
        assert latents.shape == noise.shape
    engine = MlxCachedNAR(mm, chunk)
    with pytest.raises(ValueError):
        engine.solve(steps=2, method="bogus")
    engine.close()


def test_kv_cache_grows_on_demand():
    """Growing past initial_len must produce identical logits to a full
    preallocation — growth is invisible to the decode result."""
    tm, mm = _pair()
    ids = np.arange(10)[None] % 400
    grown = make_caches(2, 40, 2, 16, initial_len=12)
    full = make_caches(2, 40, 2, 16)
    a = [mm(mx.array(ids), grown, offset=0)[0, -1]]
    b = [mm(mx.array(ids), full, offset=0)[0, -1]]
    for step in range(8):  # decode past the 12-slot initial buffer
        tok = mx.array([[50 + step]])
        a.append(mm(tok, grown, offset=10 + step)[0, -1])
        b.append(mm(tok, full, offset=10 + step)[0, -1])
    assert grown[0].keys.shape[2] > 12  # it actually grew
    for i, (g, f) in enumerate(zip(a, b)):
        d = np.abs(np.asarray(g) - np.asarray(f)).max()
        assert d < 1e-4, f"step {i} diverged after grow: {d}"
    # The hard cap still holds.
    from yue2.mlx_yue2 import KVCache
    cache = KVCache(12, 2, 16, mx.bfloat16, initial_len=4)
    with pytest.raises(ValueError):
        cache.update(mx.zeros((1, 2, 20, 16), dtype=mx.bfloat16),
                     mx.zeros((1, 2, 20, 16), dtype=mx.bfloat16))
