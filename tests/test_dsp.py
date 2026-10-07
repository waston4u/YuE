"""DSP toolkit: synthetic-signal checks, no model files."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yue2 import dsp  # noqa: E402

SR = dsp.SR


def sine(freq=440, seconds=1.0, amp=0.5):
    t = np.arange(int(SR * seconds)) / SR
    return np.stack([amp * np.sin(2 * np.pi * freq * t)] * 2, axis=1).astype(np.float32)


def test_biquad_coefficients():
    b, a = dsp.biquad("peak", 1000, 1.0, 6.0)
    assert b.shape == (3,) and a.shape == (3,)
    assert a[0] == pytest.approx(1.0)


def test_eq_boost():
    x = sine(1000)
    eq = dsp.eq_chain(("peak", 1000, 0.7, 12.0))
    y = dsp.apply_eq(x, eq)
    assert np.abs(y).max() > np.abs(x).max() * 1.5


def test_eq_cut():
    x = sine(1000)
    eq = dsp.eq_chain(("peak", 1000, 0.7, -12.0))
    y = dsp.apply_eq(x, eq)
    assert np.abs(y).max() < np.abs(x).max() * 0.7


def test_compressor_reduces_peak():
    x = sine(440, 0.5, amp=0.9)
    y = dsp.compress(x, threshold_db=-20, ratio=4, attack_ms=1, release_ms=50)
    assert np.abs(y).max() < np.abs(x).max()


def test_limiter_ceiling():
    x = sine(440, 0.5, amp=1.0) * 4
    y = dsp.limiter(x, ceiling_dbfs=-1.0)
    assert np.abs(y).max() <= 10 ** (-1 / 20) * 1.02


def test_expander_gates_silence():
    t = np.arange(SR) / SR
    x = np.stack([0.9 * np.sin(2 * np.pi * 440 * t) * (t < 0.3),
                  0.9 * np.sin(2 * np.pi * 440 * t) * (t < 0.3)], 1).astype(np.float32)
    y = dsp.expander(x, threshold_db=-40, ratio=4)
    assert np.abs(y[int(0.9 * SR):]).max() < 0.01


def test_reverb_tail_energy():
    x = np.zeros((SR // 4, 2), np.float32)
    x[0] = [1.0, 1.0]
    y = dsp.reverb(x, rt60=0.5, predelay_ms=5, mix=1.0)
    assert np.abs(y[int(0.05 * SR):]).max() > 1e-4  # tail exists past the click


def test_delay_echo():
    x = np.zeros((SR, 2), np.float32)
    x[0] = [1.0, 1.0]
    y = dsp.delay(x, time_ms=100, feedback=0.5, mix=1.0, ping_pong=False)
    assert np.abs(y[int(0.1 * SR)]).max() > 0.3
    assert np.abs(y[int(0.2 * SR)]).max() > 0.1


def test_loudness_sine():
    x = sine(1000, 1.0, amp=0.5)  # -6.02 dBFS sine ≈ -9.7 LUFS
    info = dsp.loudness(x)
    assert info["integrated_lufs"] == pytest.approx(-9.7, abs=1.0)
    assert info["peak_dbfs"] == pytest.approx(-6.02, abs=0.1)


def test_normalize_lufs():
    x = sine(440, 1.0, amp=0.3)
    y = dsp.normalize_lufs(x, -14.0)
    assert dsp.loudness(y)["integrated_lufs"] == pytest.approx(-14.0, abs=0.2)


def test_align_offset_recovers_delay():
    rng = np.random.default_rng(0)
    ref = np.stack([rng.standard_normal(SR), rng.standard_normal(SR)],
                   axis=1).astype(np.float32) * 0.3
    shift = 4800
    late = np.concatenate([np.zeros((shift, 2), np.float32), ref])[:SR]
    early = np.concatenate([ref[shift:], np.zeros((shift, 2), np.float32)])
    assert dsp.align_offset(ref, late) == pytest.approx(shift, abs=64)
    assert dsp.align_offset(ref, early) == pytest.approx(-shift, abs=64)


def test_pan_multichannel_angles():
    x = sine(440, 0.1)
    center = dsp.pan_multichannel(x, 0, "5.1")
    assert center[:, 2].max() > 0.4  # C
    assert center[:, 0].max() < 0.01 and center[:, 1].max() < 0.01
    left = dsp.pan_multichannel(x, -30, "5.1")
    assert left[:, 0].max() > 0.4 and left[:, 1].max() < 0.01
    rear = dsp.pan_multichannel(x, 180, "7.1")
    assert rear[:, 4].max() > 0.3 and rear[:, 5].max() > 0.3  # Lss/Rss


def test_binaural_lateralizes():
    x = sine(440, 0.5)
    right = dsp.binaural(x, azimuth=60)
    assert np.abs(right[:, 1]).max() > np.abs(right[:, 0]).max()
    left = dsp.binaural(x, azimuth=-60)
    assert np.abs(left[:, 0]).max() > np.abs(left[:, 1]).max()


def test_resample_lengths():
    x = sine(440, 1.0)
    y = dsp.resample(x, 48000, 44100)
    assert y.shape[0] == 44100
    z = dsp.resample(x, 48000, 48000)
    assert np.array_equal(z, x)


def test_ms_widen():
    x = sine(440, 0.5)
    wide = dsp.ms_widen(x, 0.0)  # collapse to mono-ish side=0
    assert np.allclose(wide[:, 0], wide[:, 1])
