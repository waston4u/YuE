"""Real-kernel int8 tests: packed matmul correctness, model quantization,
MPS execution, Apple-Silicon gating, and rejection of removed options."""
import platform
import subprocess
import sys

import pytest
import torch
from torch import nn

from yue2 import quantization as quant
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from yue2.protocol import EOD, MUSIC_START, VOCAB_SIZE, Sampling
from yue2.sampling import generate_tokens


def tiny_config(**overrides):
    fields = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=1,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                  vocab_size=VOCAB_SIZE, max_position_embeddings=256,
                  max_latent_frames=64)
    fields.update(overrides)
    return YuE2Config(**fields)


def tiny_model(**overrides):
    torch.manual_seed(0)
    return YuE2ForCausalLM(tiny_config(**overrides)).eval()


def test_optional_imports_do_not_load_cuda_extensions():
    code = """
import sys, importlib.abc
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'vllm', 'triton'}:
            raise RuntimeError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, Deny())
import yue2, yue2.pipeline, yue2.quantization, yue2.sampling, yue2.nar
assert not any(m.split('.')[0] in {'vllm', 'triton'} for m in sys.modules)
assert not hasattr(__import__('yue2'), 'fast')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_int8linear_matches_reference_matmul_cpu():
    """Real packed matmul on CPU vs the bf16 Linear it replaced."""
    torch.manual_seed(1)
    linear = nn.Linear(64, 96, bias=True)
    packed = quant.Int8Linear(linear)
    x = torch.randn(7, 64)
    expected = linear(x.float())
    actual = packed(x).float()
    assert torch.isfinite(actual).all()
    error = (actual - expected).abs().max() / expected.abs().max()
    assert error < 0.05  # per-channel int8: ~1% typical error


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple Metal GPU")
def test_int8linear_matches_reference_matmul_mps():
    """Same check on the real Metal kernel — catches the bf16-input bug."""
    torch.manual_seed(1)
    linear = nn.Linear(64, 96, bias=False)
    packed = quant.Int8Linear(linear).to("mps")
    x = torch.randn(7, 64, device="mps")
    expected = linear.float().to("mps")(x.float())
    actual = packed(x).float()
    assert torch.isfinite(actual).all()
    error = (actual - expected).abs().max() / expected.abs().max()
    assert error < 0.05


def test_prepare_int8_quantizes_aligned_linears_only():
    """Real model: decoder projections pack to int8, everything else stays."""
    model = tiny_model(intermediate_size=40)  # 40 % 32 != 0 → MLPs stay bf16
    status = quant.prepare_int8_ar(model)
    replaced = {name: type(module).__name__ for name, module in model.named_modules()
                if isinstance(module, (quant.Int8Linear, nn.Linear))}
    assert isinstance(model.lm_head, quant.Int8Linear)
    for name, kind in replaced.items():
        targeted = quant.INT8_LINEAR.fullmatch(name) is not None
        if ".mlp." in name or ".nar_mlp." in name:
            assert kind == "Linear", name          # unaligned → untouched
        elif targeted:
            assert kind == "Int8Linear", name      # aligned attn projections
        else:
            assert kind == "Linear", name          # e.g. llm2vae — not a decoder layer
    assert status["mode"] == "int8"
    # Idempotent — a second pass must not re-wrap or fail.
    assert quant.prepare_int8_ar(model)["mode"] == "int8"


def test_int8_model_forward_produces_finite_logits_cpu():
    model = tiny_model()
    quant.prepare_int8_ar(model)
    ids = torch.tensor([[EOD, MUSIC_START]])
    with torch.inference_mode():
        logits = model(ids).logits
    assert torch.isfinite(logits).all()
    assert logits.shape[-1] == VOCAB_SIZE


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple Metal GPU")
def test_int8_model_generates_real_tokens_on_mps():
    """End-to-end: int8-quantized model on Metal runs the real decode loop."""
    model = tiny_model()
    quant.prepare_int8_ar(model, device="mps")
    model.to("mps")
    sampling = Sampling(temperature=0, top_p=1.0, top_k=1,
                        repetition_penalty=1.0, penalty_window=1,
                        min_tokens=0, max_tokens=8)
    history, timing, truncated = generate_tokens(
        model, [EOD, MUSIC_START], sampling, 0, "semantic")
    assert 0 < len(history) <= 8
    assert timing["execution"] == "eager"


def test_pipeline_rejects_non_apple_compute(tmp_path):
    from yue2.pipeline import YuE2Pipeline
    with pytest.raises(TypeError):  # backend parameter was removed entirely
        YuE2Pipeline(tmp_path / "m", tmp_path / "v", backend="vllm")
    with pytest.raises(ValueError, match="int8"):
        YuE2Pipeline(tmp_path / "m", tmp_path / "v", quantization="fp8")
    with pytest.raises(ValueError, match="CUDA"):
        YuE2Pipeline(tmp_path / "m", tmp_path / "v", device="cuda")
    with pytest.raises(ValueError, match="CUDA"):
        YuE2Pipeline(tmp_path / "m", tmp_path / "v", device="cuda:0")


def test_platform_gate_refuses_non_apple_silicon(monkeypatch, tmp_path):
    """Every start path must refuse off arm64 macOS."""
    from yue2._platform import require_apple_silicon
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        require_apple_silicon()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")  # Intel / Rosetta
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        require_apple_silicon()
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    require_apple_silicon()  # arm64 macOS passes


def test_mlx_auto_picks_best_fidelity_that_fits(monkeypatch, tmp_path):
    """engine=mlx quantization=auto picks by FULL working-set floors —
    bf16 ≥13 GiB free, int8 ≥9, int4 ≥5; unmeasurable free memory
    defaults to the safe middle (int8)."""
    pytest.importorskip("mlx.core")
    from yue2 import pipeline as pl
    model_dir, vae_dir = tmp_path / "m", tmp_path / "v"
    for d in (model_dir, vae_dir):
        d.mkdir()
        (d / "w.safetensors").write_bytes(b"\x00")
    monkeypatch.setattr(pl, "YuE2TextTokenizer", lambda *a, **k: object())
    for free, want in [(None, "int8"), (13.0, "none"), (12.9, "int8"),
                       (9.0, "int8"), (8.9, "int4"), (5.0, "int4")]:
        monkeypatch.setattr(pl, "_free_gib", lambda f=free: f)
        pipe = pl.YuE2Pipeline(model_dir, vae_dir, engine="mlx",
                               quantization="auto", progress=False)
        assert pipe.quantization == want, f"free={free}"


def test_cli_rejects_non_apple_options():
    from yue2 import cli
    for argv in (["generate", "--backend", "vllm"],
                 ["generate", "--quantization", "fp8"],
                 ["studio", "--backend", "vllm"],
                 ["studio", "--quantization", "fp8"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
