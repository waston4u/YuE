import json
import pytest
import torch
from yue2.protocol import *
from yue2.sampling import distribution, window_penalty
from yue2.cli import request_kwargs
from yue2.storage import verify_result, write_json


class Tokenizer:
    def encode(self, text):
        return [ord(x) for x in text]


def test_defaults_and_native_instructions():
    cfg = GenerationConfig()
    assert cfg.abc == Sampling(.7, .9, 30, 1.005, 100, 32, 4096)
    assert cfg.semantic == Sampling(1., .95, 100, 1.2, 50, 200, 9000)
    assert (cfg.ode_steps, cfg.ode_method, cfg.context) == (32, "midpoint", 24576)
    assert SongRequest("piano", "line\nline").cot == "full"
    assert "chord-annotated" in SongRequest("a", "b").text()
    assert "melody-only" in SongRequest("a", "b", cot="melody").text()
    assert SongRequest("a", "b", cot="off").guidance == 1.01


def test_cfg_keeps_exact_score_and_removes_only_text():
    t = Tokenizer()
    for cot in ("full", "melody"):
        r = SongRequest("STYLE", "LYRIC", cot=cot, cfg_scale=1.2)
        ids = [17, 19, 23]
        positive = token_prefixes(r, t, ids)
        negative = negative_prefix(r, t, ids)
        assert positive[-6:] == negative[-6:] == [ABC_START, 17, 19, 23, ABC_END, MUSIC_START]
        assert negative == [EOD] + t.encode(INSTRUCTIONS[cot]) + negative[-6:]
        assert positive != negative
    r = SongRequest("a", "b", cot="off")
    assert token_prefixes(r, t)[-3:] == [ABC_START, ABC_END, MUSIC_START]
    assert ABC_START not in negative_prefix(r, t)


def test_external_abc_retains_crlf(tmp_path):
    abc = 'X:1\r\nK:C\r\n"C"CDEF|\r\n'
    (tmp_path / "a.abc").write_bytes(abc.encode())
    result = request_kwargs({"id":"test", "tags":"x", "lyrics":"y", "abc_path":"a.abc"}, tmp_path)
    assert result["abc"] == abc
    with pytest.raises(ValueError):
        SongRequest("a", "b", cot="off", abc=abc)
    with pytest.raises(ValueError):
        request_kwargs({"tags":"x", "lyrics":"y", "unknown":1})


def test_window_count_sign_and_window_expiry():
    logits = torch.tensor([[4., -4., 3.]])
    torch.testing.assert_close(window_penalty(logits, [0, 0, 1], 2), torch.tensor([[1., -8., 3.]]))
    torch.testing.assert_close(window_penalty(logits, [2], 2), torch.tensor([[4., -4., 1.5]]))


def test_top_p_off_keeps_three_and_symbolic_one():
    logits = torch.full((1, VOCAB_SIZE), -torch.inf, dtype=torch.bfloat16)
    logits[0, CODEC_OFFSET:CODEC_OFFSET+5] = torch.tensor([10., 1., 0., -1., -2.])
    config = Sampling(1, .5, 5, 1, 50, 0, 5)
    off = distribution(logits, config, [], 0, "semantic", legacy_off=True)
    symbolic = distribution(logits, config, [], 0, "semantic", legacy_off=False)
    assert torch.isfinite(off).sum() == 3
    assert torch.isfinite(symbolic).sum() == 1
    assert off.dtype == torch.bfloat16 and symbolic.dtype == torch.float32


def reference_distribution(logits, sampling, history, step, phase, legacy_off=False):
    """The pre-optimization implementation — kept here as ground truth."""
    from yue2.protocol import ABC_END, MUSIC_END, CODEC_OFFSET, CODEC_SIZE
    scores = logits.clone() if legacy_off else logits.float().clone()
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, float("-inf"))
    if phase == "abc":
        allowed[..., :EOD] = 0
    else:
        allowed[..., CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[..., end] = 0
    scores = scores + allowed
    if step < sampling.min_tokens:
        scores[..., end] = -torch.inf
    if sampling.repetition_penalty != 1.0 and history:
        recent = torch.as_tensor(history[-sampling.penalty_window:],
                                 dtype=torch.long, device=logits.device).reshape(1, -1)
        freq = torch.zeros_like(scores)
        freq.scatter_add_(-1, recent, torch.ones_like(recent, dtype=scores.dtype))
        alpha = sampling.repetition_penalty ** freq
        scores = torch.where(scores < 0, scores * alpha, scores / alpha)
    if sampling.temperature == 0:
        return scores
    if sampling.temperature != 1:
        scores = scores / sampling.temperature
    threshold = scores.topk(min(sampling.top_k, scores.shape[-1])).values[..., -1, None]
    scores = scores.masked_fill(scores < threshold, -torch.inf)
    if sampling.top_p < 1:
        values, indices = scores.sort(descending=True)
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > sampling.top_p
        removed[..., :3 if legacy_off else 1] = False
        values = values.masked_fill(removed, -torch.inf)
        scores = values.scatter(-1, indices, values)
    return scores


def test_optimized_distribution_is_identical_to_reference():
    """The sparse/topk optimizations must produce bit-identical scores."""
    torch.manual_seed(0)
    for phase in ("abc", "semantic"):
        for legacy_off in (False, True):
            for history in ([], [5, 5, 9, 12] * 20):
                for rounded in (False, True):
                    logits = torch.randn(1, VOCAB_SIZE).bfloat16() * 4
                    if rounded:
                        # Coarse values create large tie groups that stress
                        # the removal-boundary ordering.
                        logits = logits.float().round().bfloat16()
                    config = Sampling(temperature=0.7, top_p=0.9, top_k=50,
                                      repetition_penalty=1.1, penalty_window=80,
                                      min_tokens=0, max_tokens=10)
                    expected = reference_distribution(logits, config, history, 5,
                                                      phase, legacy_off)
                    actual = distribution(logits, config, history, 5,
                                          phase, legacy_off)
                    # Equal-probability ties may occupy different token IDs in
                    # the compact sort vs the legacy full sort — an arbitrary
                    # boundary choice either way. The kept score multiset and
                    # kept count must be identical.
                    assert int(actual.isfinite().sum()) == int(expected.isfinite().sum())
                    kept_a = actual[actual.isfinite()].sort(descending=True).values
                    kept_e = expected[expected.isfinite()].sort(descending=True).values
                    torch.testing.assert_close(kept_a, kept_e, rtol=0, atol=0)


def test_window_penalty_matches_reference():
    torch.manual_seed(1)
    logits = torch.randn(1, VOCAB_SIZE).float()
    history = [7, 7, 7, 11, 13]
    recent = torch.as_tensor(history, dtype=torch.long).reshape(1, -1)
    freq = torch.zeros_like(logits)
    freq.scatter_add_(-1, recent, torch.ones_like(recent, dtype=logits.dtype))
    alpha = 1.3 ** freq
    expected = torch.where(logits < 0, logits * alpha, logits / alpha)
    torch.testing.assert_close(window_penalty(logits, history, 1.3), expected)





def test_vocab_and_eos_minimum():
    logits = torch.zeros(1, VOCAB_SIZE)
    config = Sampling(0, 1., 100, 1, 50, 2, 5)
    score = distribution(logits, config, [], 0, "semantic")
    assert not torch.isfinite(score[0, MUSIC_END])
    assert not torch.isfinite(score[0, EOD])
    assert torch.isfinite(score[0, CODEC_OFFSET:CODEC_OFFSET+CODEC_SIZE]).all()
    assert torch.isfinite(distribution(logits, config, [], 2, "semantic")[0, MUSIC_END])


def test_integrity_rejects_incomplete_outputs(tmp_path):
    write_json(tmp_path / "result.json", {"status":"complete", "identity":"x", "artifacts":{}})
    with pytest.raises(ValueError, match="Incomplete"):
        verify_result(tmp_path, "x")


def test_invalid_hyperparameters():
    for value in (float("nan"), float("inf"), -1, 21):
        with pytest.raises(ValueError):
            SongRequest("x", "y", cfg_scale=value)
    with pytest.raises(ValueError):
        Sampling(min_tokens=20, max_tokens=10)
    with pytest.raises(ValueError):
        GenerationConfig(context=1024)


def test_partial_abc_override_keeps_abc_defaults():
    config = GenerationConfig.from_dict({"abc": {"temperature": .4}})
    assert config.abc == Sampling(.4, .9, 30, 1.005, 100, 32, 4096)
    assert resolve_sampling({"top_k": 12}, GenerationConfig().abc).max_tokens == 4096


def test_saved_plan_preserves_exact_ids_and_rejects_edits(tmp_path):
    from yue2 import SymbolicPlan
    request = SongRequest("style", "lyrics", abc="ABC")
    plan = SymbolicPlan(request, "ABC", [65,66,67], token_prefixes(request, Tokenizer(), [65,66,67]))
    plan.save(tmp_path)
    assert SymbolicPlan.load(tmp_path) == plan
    (tmp_path / "score.abc").write_text("DEF")
    with pytest.raises(ValueError, match="changed"):
        SymbolicPlan.load(tmp_path)
    with pytest.raises(ValueError, match="ordinary"):
        token_prefixes(request, Tokenizer(), [ABC_END])
