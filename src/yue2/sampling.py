"""Request-local sampling, preserving mode-specific historical arithmetic."""
from __future__ import annotations
import time
import torch
from .protocol import EOD, ABC_END, MUSIC_END, CODEC_OFFSET, CODEC_SIZE, CONTEXT


def synchronize(device):
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.synchronize()


def window_penalty(logits, recent_ids, penalty):
    if penalty == 1.0 or len(recent_ids) == 0:
        return logits
    # Only the ≤penalty_window recent tokens change — penalize those
    # positions directly instead of materializing a full-vocab map.
    recent = torch.as_tensor(recent_ids, dtype=torch.long, device=logits.device).reshape(-1)
    unique, counts = recent.unique(return_counts=True)
    alpha = penalty ** counts.to(logits.dtype)
    selected = logits[..., unique]
    penalized = torch.where(selected < 0, selected * alpha, selected / alpha)
    logits = logits.clone()
    logits[..., unique] = penalized
    return logits


def allowed_mask(scores, phase):
    """-inf everywhere except the phase's legal token range (+ its end token)."""
    end = ABC_END if phase == "abc" else MUSIC_END
    mask = torch.full_like(scores, float("-inf"))
    if phase == "abc":
        mask[..., :EOD] = 0
    else:
        mask[..., CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE] = 0
    mask[..., end] = 0
    return mask


def distribution(logits, sampling, history, step, phase, legacy_off=False, allowed=None):
    # All sampling math runs on CPU — multinomial already runs host-side for
    # deterministic seeds, so the one row transfer replaces ~25 tiny MPS
    # kernels plus allocator churn (~18ms/token measured) with ~3ms of CPU
    # math, and makes the kept set identical across devices.
    scores = logits.detach().to("cpu")
    scores = scores.clone() if legacy_off else scores.float()
    if allowed is not None and allowed.device != scores.device:
        allowed = allowed.to(scores.device)
    end = ABC_END if phase == "abc" else MUSIC_END
    scores = scores + (allowed if allowed is not None else allowed_mask(scores, phase))
    if step < sampling.min_tokens:
        scores[..., end] = -torch.inf
    scores = window_penalty(scores, history[-sampling.penalty_window:], sampling.repetition_penalty)
    if sampling.temperature == 0:
        return scores
    if sampling.temperature != 1:
        scores = scores / sampling.temperature
    threshold = scores.topk(min(sampling.top_k, scores.shape[-1])).values[..., -1, None]
    scores = scores.masked_fill(scores < threshold, -torch.inf)
    if sampling.top_p < 1:
        # -inf entries contribute 0 to the softmax, so sorting only the finite
        # candidates (~top_k of 184,704) is equivalent to sorting the whole
        # row — except that equal-probability tokens may rank differently.
        # The kept multiset of scores is identical; only which token ID fills
        # a tied boundary slot can differ from the legacy full sort (an
        # arbitrary choice either way, both deterministic).
        kept = int(scores.isfinite().sum(-1).max())
        top_values, top_indices = scores.topk(kept)
        by_index = top_indices.argsort(dim=-1)
        top_values = top_values.gather(-1, by_index)
        top_indices = top_indices.gather(-1, by_index)
        by_value = top_values.argsort(dim=-1, descending=True, stable=True)
        top_values = top_values.gather(-1, by_value)
        top_indices = top_indices.gather(-1, by_value)
        probabilities = top_values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > sampling.top_p
        removed[..., :3 if legacy_off else 1] = False
        top_values = top_values.masked_fill(removed, -torch.inf)
        scores = torch.full_like(scores, -torch.inf).scatter(-1, top_indices, top_values)
    return scores


@torch.inference_mode()
def generate_tokens(model, prefix, sampling, seed, phase, negative=None, cfg_scale=1.0,
                    legacy_off=False, cancelled=None, on_token=None):
    from .modeling_yue2 import StaticKVCache
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if len(prefix) + sampling.max_tokens > CONTEXT:
        raise ValueError("Prefix + requested generation budget exceeds 24576; no implicit truncation")
    if cfg_scale != 1 and negative is None:
        raise ValueError("CFG requires a negative prefix")
    if negative is not None and len(negative) + sampling.max_tokens > CONTEXT:
        raise ValueError("Negative prefix + generation budget exceeds context")
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before prefill")
    # The two stages deliberately reset their request-local seed, matching the preset.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    config = model.config

    def prefill(ids):
        cache = StaticKVCache(num_layers=config.num_hidden_layers, batch_size=1,
                              num_kv_heads=config.num_key_value_heads,
                              max_seq_len=len(ids) + sampling.max_tokens,
                              head_dim=config.head_dim, dtype=dtype, device=device)
        output = model(torch.tensor([ids], device=device), past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
        return output.logits[:, -1, :], output.past_key_values

    positive_cache = negative_cache = None
    synchronize(device)
    start = time.perf_counter()
    try:
        conditional, positive_cache = prefill(prefix)
        unconditional = None
        if cfg_scale != 1.0:
            unconditional, negative_cache = prefill(negative)
        synchronize(device)
        prefill_seconds = time.perf_counter() - start
        history, first, eos = [], None, False
        end = ABC_END if phase == "abc" else MUSIC_END
        # The phase mask never changes across steps — build it once, on CPU
        # where the distribution math runs.
        mask_shape = torch.empty(1, config.vocab_size,
                                 dtype=dtype if legacy_off else torch.float32)
        allowed = allowed_mask(mask_shape, phase)
        for step in range(sampling.max_tokens):
            if cancelled is not None and cancelled():
                raise InterruptedError(f"Cancelled during {phase}")
            # Preserve historical BF16 CFG subtraction/multiply/add before upcast.
            logits = conditional if cfg_scale == 1.0 else unconditional + cfg_scale * (conditional - unconditional)
            scores = distribution(logits, sampling, history, step, phase, legacy_off,
                                  allowed=allowed)
            if sampling.temperature == 0:
                next_id = scores.argmax(-1, keepdim=True)
            else:
                probabilities = scores.softmax(-1)
                # Sampling runs on CPU for deterministic seeds across devices;
                # the per-step host round-trip is inherent to autoregressive
                # decode, and keeps MPS output identical to CPU output.
                next_id = torch.multinomial(probabilities, 1, generator=generator)
            token = int(next_id.item())  # CPU read — does not drain the GPU
            if first is None:
                first = time.perf_counter() - start
            if on_token is not None:
                on_token(phase, token)
            if token == end:
                eos = True
                break
            history.append(token)
            if step + 1 < sampling.max_tokens:
                conditional = model(next_id.to(device), past_key_values=positive_cache,
                                    use_cache=True, logits_to_keep=1).logits[:, -1, :]
                if negative_cache is not None:
                    unconditional = model(next_id.to(device), past_key_values=negative_cache,
                                          use_cache=True, logits_to_keep=1).logits[:, -1, :]
        synchronize(device)
        seconds = time.perf_counter() - start
        count = len(history) + int(eos)
        timing = {"seconds": seconds, "prefill_seconds": prefill_seconds,
                  "ttft_seconds": first, "output_tokens": count, "content_tokens": len(history),
                  "output_tps": count / seconds, "prefix_tokens": len(prefix),
                  "cfg_branches": 1 if cfg_scale == 1 else 2,
                  "execution": "eager", "attention": "sdpa"}
        return history, timing, not eos
    finally:
        positive_cache = negative_cache = None
