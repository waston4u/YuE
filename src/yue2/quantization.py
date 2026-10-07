"""int8 weight-only quantization for Apple Silicon (Metal/MPS and CPU).

Packs AR+NAR decoder projections and lm_head via ``aten._weight_int8pack_mm``
— roughly half the bf16 footprint (≈6.2 → ≈3.2 GiB) so the GPU path fits on
16 GB unified-memory machines. Originals are not retained; freeing RAM is
the point. No quantized quality or speed claim is implied beyond that.
"""
from __future__ import annotations

import re
import torch
from torch import nn

#: int8 covers both AR and NAR decoder projections plus lm_head. NAR reuses
#: the shared decoder layers during the ODE solve, so quantizing only the AR
#: half would leave it running on mixed-precision inputs anyway; doing both
#: keeps the whole stack on one weight format. Originals are NOT kept —
#: freeing RAM is the entire point on unified-memory machines.
INT8_LINEAR = re.compile(
    r"(?:model\.layers\.\d+\.(?:(?:self_attn|nar_self_attn)\.(?:q|k|v|o)_proj"
    r"|(?:mlp|nar_mlp)\.(?:gate|up|down)_proj)|lm_head)$")


class Int8Linear(nn.Module):
    """Per-output-channel int8 weights via ``aten._weight_int8pack_mm``.

    The packed kernel only computes correctly with fp32 activations — bf16
    input returns garbage on the Metal implementation — so activations are
    cast in and out per call. Halves AR weight memory (≈6.2 → ≈3.2 GiB) so
    the MPS path fits on 16 GB unified-memory machines; the same op also
    runs on CPU, which keeps the MPS-OOM fallback alive.
    """
    def __init__(self, linear):
        super().__init__()
        weight = linear.weight.detach().float()
        scales = (weight.abs().amax(dim=1) / 127.0).clamp_min(1e-8)
        packed = torch.round(weight / scales[:, None]).clamp(-127, 127).to(torch.int8)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.register_buffer("weight_int8", packed.to(linear.weight.device))
        self.register_buffer("weight_scales", scales.to(linear.weight.device))
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, value):
        shape = value.shape
        rows = value.reshape(-1, self.in_features).float()
        out = torch.ops.aten._weight_int8pack_mm(rows, self.weight_int8, self.weight_scales)
        if self.bias is not None:
            out = out + self.bias.float()
        return out.reshape(*shape[:-1], self.out_features).to(value.dtype)


def int8_supported(device):
    """Probe the packed kernel — present-but-broken counts as unsupported."""
    if not hasattr(torch.ops.aten, "_weight_int8pack_mm"):
        return False
    try:
        # The Metal kernel requires N % 32 == 0 and K % 32 == 0.
        w8 = torch.zeros(32, 64, dtype=torch.int8, device=device)
        scales = torch.ones(32, device=device)
        x = torch.zeros(2, 64, device=device)
        return bool(torch.ops.aten._weight_int8pack_mm(x, w8, scales).isfinite().all())
    except Exception:
        return False


def prepare_int8_ar(model, device=None):
    """Quantize AR+NAR decoder projections and lm_head to packed int8.

    Runs on whatever device the model currently occupies; quantizing before
    ``.to(device)`` halves the transfer and the resident footprint.
    """
    if model is None:
        raise ValueError("Load the MoT model before preparing int8")
    if getattr(model, "_yue2_int8", False):
        return quantization_status(model)
    selected = [(name, module) for name, module in model.named_modules()
                if INT8_LINEAR.fullmatch(name) and isinstance(module, nn.Linear)
                and module.in_features % 32 == 0 and module.out_features % 32 == 0]
    if not selected:
        raise ValueError("Expected unquantized YuE2 Linear layers")
    # The Metal packed kernel rejects unaligned dims — those linears stay BF16.
    if not int8_supported(device or next(model.parameters()).device):
        raise RuntimeError("aten._weight_int8pack_mm is unavailable — use quantization='none'")
    for name, original in selected:
        _replace(model, name, Int8Linear(original))
    object.__setattr__(model, "_yue2_int8", True)
    return quantization_status(model)


def _replace(model, name, replacement):
    if "." not in name:
        setattr(model, name, replacement)
        return
    parent_name, child = name.rsplit(".", 1)
    setattr(model.get_submodule(parent_name), child, replacement)


def quantization_status(model):
    int8 = bool(getattr(model, "_yue2_int8", False)) if model is not None else False
    return {"mode": "int8" if int8 else "none",
            "weight_format": "int8_weight_only" if int8 else None,
            "quality_validation": "unvalidated", "performance_validation": "unvalidated"}
