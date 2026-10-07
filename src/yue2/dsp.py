"""Numpy-only DSP toolkit for the YuE2 session mixer.

No scipy: recursive processors are implemented as FFT-domain equalization,
block-envelope dynamics, and convolution with synthesized impulse responses —
vectorized, deterministic, and honest about what they are. Audio is float32
``[samples, channels]`` at 48 kHz throughout.
"""
from __future__ import annotations

import math

import numpy as np

SR = 48000


# ══════════════════════════════════════════════════════════════════════════════
# Equalization — FFT-domain RBJ-magnitude filters (zero phase)
# ══════════════════════════════════════════════════════════════════════════════


def biquad(kind: str, freq: float, q: float = 0.707, gain_db: float = 0.0,
           sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """RBJ biquad coefficients (b, a) for lowshelf/peak/highshelf/lowpass/highpass."""
    if not 0 < freq < sr / 2 or q <= 0:
        raise ValueError("Invalid biquad frequency or Q")
    a0 = 2 * math.pi * freq / sr
    alpha = math.sin(a0) / (2 * q)
    cw = math.cos(a0)
    A = 10 ** (gain_db / 40)
    if kind == "lowshelf":
        sq = 2 * math.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sq), 2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - sq)]
        a = [(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw),
             (A + 1) + (A - 1) * cw - sq]
    elif kind == "highshelf":
        sq = 2 * math.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sq), -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - sq)]
        a = [(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw),
             (A + 1) - (A - 1) * cw - sq]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    else:
        raise ValueError(f"Unknown biquad kind {kind!r}")
    b, a = np.asarray(b, dtype=np.float64), np.asarray(a, dtype=np.float64)
    return b / a[0], a / a[0]


def _response(b: np.ndarray, a: np.ndarray, n_fft: int) -> np.ndarray:
    """Frequency response of a biquad at rfft bin centers."""
    w = np.exp(-2j * np.pi * np.arange(n_fft // 2 + 1) / n_fft)
    num = b[0] + b[1] * w + b[2] * w * w
    den = a[0] + a[1] * w + a[2] * w * w
    return np.abs(num / den)


def apply_eq(x: np.ndarray, bands: list[tuple[np.ndarray, np.ndarray]],
             sr: int = SR) -> np.ndarray:
    """Apply a chain of biquads in the frequency domain (zero phase).

    Edge effects are minimized by reflect-padding half a second on each side.
    """
    x = np.asarray(x, dtype=np.float32)
    if not bands or x.shape[0] == 0:
        return x.copy()
    pad = min(sr // 2, x.shape[0])
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
    n = 1 << (padded.shape[0] - 1).bit_length()
    gain = np.ones(n // 2 + 1)
    for b, a in bands:
        gain *= _response(b, a, n)
    spec = np.fft.rfft(padded, n=n, axis=0)
    spec *= gain[:, None]
    out = np.fft.irfft(spec, n=n, axis=0)[:padded.shape[0]]
    return out[pad:pad + x.shape[0]].astype(np.float32)


def eq_chain(*specs) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build biquad chains from (kind, freq, q, gain_db) tuples."""
    return [biquad(kind, freq, q, gain) for kind, freq, q, gain in specs]


# ══════════════════════════════════════════════════════════════════════════════
# Dynamics — block-envelope gain computation (vectorized, deterministic)
# ══════════════════════════════════════════════════════════════════════════════


def _envelope(x: np.ndarray, sr: int, window_ms: float = 5.0) -> np.ndarray:
    """Per-sample peak envelope via moving max on a mono sum."""
    mono = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    w = max(1, int(sr * window_ms / 1000))
    pad = np.pad(mono, (w // 2, w - w // 2), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, w)
    env = windows.max(axis=1)[:mono.shape[0]]
    return env


def _smooth_attack_release(env: np.ndarray, sr: int, attack_ms: float,
                           release_ms: float, hop: int = 32) -> np.ndarray:
    """Attack/release smoothing evaluated at hop resolution, then upsampled."""
    env = np.asarray(env, dtype=np.float64)
    att = math.exp(-hop / (sr * attack_ms / 1000)) if attack_ms > 0 else 0.0
    rel = math.exp(-hop / (sr * release_ms / 1000)) if release_ms > 0 else 0.0
    n = (env.shape[0] + hop - 1) // hop
    padded = np.pad(env, (0, n * hop - env.shape[0]), mode="edge")
    frames = padded.reshape(n, hop).max(axis=1)
    out = np.empty(n)
    state = 0.0
    for i, value in enumerate(frames):
        coef = att if value > state else rel
        state = coef * state + (1 - coef) * value
        out[i] = state
    return np.repeat(out, hop)[:env.shape[0]]


def _db(value: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(value, 1e-9))


def _undb(db: np.ndarray | float) -> np.ndarray:
    return 10 ** (np.asarray(db) / 20)


def compress(x: np.ndarray, threshold_db: float = -18.0, ratio: float = 3.0,
             attack_ms: float = 10.0, release_ms: float = 120.0,
             knee_db: float = 6.0, makeup_db: float = 0.0, sr: int = SR) -> np.ndarray:
    """Feed-forward stereo-linked compressor with soft knee."""
    x = np.asarray(x, dtype=np.float32)
    env = _smooth_attack_release(_envelope(x, sr), sr, attack_ms, release_ms)
    level = _db(env)
    over = level - threshold_db
    if knee_db > 0:
        gain_db = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, over * (1 / ratio - 1),
                           (1 / ratio - 1) * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    else:
        gain_db = np.minimum(0.0, over * (1 / ratio - 1))
    gain = _undb(gain_db + makeup_db)[:, None]
    return (x * gain).astype(np.float32)


def limiter(x: np.ndarray, ceiling_dbfs: float = -1.0, lookahead_ms: float = 5.0,
            release_ms: float = 60.0, sr: int = SR) -> np.ndarray:
    """Brickwall limiter: lookahead peak envelope, unity below the ceiling."""
    x = np.asarray(x, dtype=np.float32)
    ceiling = _undb(ceiling_dbfs)
    env = _envelope(x, sr, lookahead_ms)
    needed = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))
    needed = _smooth_attack_release(needed, sr, 0.01, release_ms)
    return (x * needed[:, None]).astype(np.float32)


def expander(x: np.ndarray, threshold_db: float = -50.0, ratio: float = 2.0,
             attack_ms: float = 5.0, release_ms: float = 150.0, sr: int = SR) -> np.ndarray:
    """Downward expander — light stem cleanup, not source separation."""
    x = np.asarray(x, dtype=np.float32)
    env = _smooth_attack_release(_envelope(x, sr), sr, attack_ms, release_ms)
    level = _db(env)
    under = np.clip(level - threshold_db, None, 0)
    gain = _undb(under * (ratio - 1))[:, None]
    return (x * gain).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Time-domain FX — convolution with synthesized impulse responses
# ══════════════════════════════════════════════════════════════════════════════


def _convolve(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Linear convolution x[m,ch] * ir[m,ch] via chunked overlap-add FFT."""
    n_out = x.shape[0] + ir.shape[0] - 1
    block = max(ir.shape[0] * 2, 1 << 18)
    n_fft = 1 << (block + ir.shape[0] - 1).bit_length()
    ir_spec = np.fft.rfft(ir, n=n_fft, axis=0)
    out = np.zeros((n_out, x.shape[1]), dtype=np.float64)
    for start in range(0, x.shape[0], block):
        seg = x[start:start + block]
        spec = np.fft.rfft(seg, n=n_fft, axis=0) * ir_spec
        conv = np.fft.irfft(spec, n=n_fft, axis=0)[:seg.shape[0] + ir.shape[0] - 1]
        out[start:start + conv.shape[0]] += conv
    return out[:n_out].astype(np.float32)


def reverb_ir(rt60: float = 1.8, predelay_ms: float = 20.0, sr: int = SR,
              seed: int = 0) -> np.ndarray:
    """Stereo exponentially-decaying noise IR; decorrelated channels."""
    length = max(1, int(sr * rt60 * 1.2))
    rng = np.random.default_rng(seed)
    t = np.arange(length) / sr
    decay = np.exp(-6.907755 * t / rt60)
    ir = rng.standard_normal((length, 2)) * decay[:, None]
    ir[:, 1] = np.roll(ir[:, 1], int(sr * 0.00037))  # subtle L/R decorrelation
    pre = np.zeros((int(sr * predelay_ms / 1000), 2))
    ir = np.concatenate([pre, ir], axis=0)
    ir[pre.shape[0]] = [1.0, 0.85]  # direct-ish early tap keeps definition
    return ir.astype(np.float32)


def reverb(x: np.ndarray, rt60: float = 1.8, predelay_ms: float = 20.0,
           mix: float = 1.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Wet-only or mixed algorithmic reverb via IR convolution."""
    x = np.asarray(x, dtype=np.float32)
    wet = _convolve(x, reverb_ir(rt60, predelay_ms, sr, seed))
    wet = wet[:x.shape[0]] if wet.shape[0] >= x.shape[0] else np.pad(
        wet, ((0, x.shape[0] - wet.shape[0]), (0, 0)))
    peak = np.abs(wet).max()
    if peak > 0:
        wet = wet / peak * np.abs(x).max() if np.abs(x).max() > 0 else wet
    if mix >= 1.0:
        return wet.astype(np.float32)
    return ((1 - mix) * x + mix * wet).astype(np.float32)


def delay_ir(time_ms: float = 375.0, feedback: float = 0.3, taps: int = 8,
             ping_pong: bool = False, sr: int = SR) -> np.ndarray:
    """Impulse train IR: taps at t, 2t, 3t… with feedback decay."""
    step = max(1, int(sr * time_ms / 1000))
    ir = np.zeros((taps * step, 2), dtype=np.float32)
    for i in range(taps):
        gain = feedback ** i
        if ping_pong:
            ir[i * step, i % 2] = gain
        else:
            ir[i * step] = gain
    ir[0] += 0.0  # wet-only: first tap carries the signal
    return ir


def delay(x: np.ndarray, time_ms: float = 375.0, feedback: float = 0.3,
          mix: float = 1.0, ping_pong: bool = True, sr: int = SR) -> np.ndarray:
    """Wet-only or mixed feedback delay via IR convolution."""
    x = np.asarray(x, dtype=np.float32)
    wet = _convolve(x, delay_ir(time_ms, feedback, 8, ping_pong, sr))
    if wet.shape[0] > x.shape[0]:
        wet = wet[:x.shape[0]]
    if mix >= 1.0:
        return wet.astype(np.float32)
    return ((1 - mix) * np.pad(x, ((0, max(0, wet.shape[0] - x.shape[0])), (0, 0)))[:wet.shape[0]]
            + mix * wet).astype(np.float32)


def ms_widen(x: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Mid/side stereo width; amount 1 = unchanged."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("ms_widen expects stereo [n,2] audio")
    mid = (x[:, 0] + x[:, 1]) / 2
    side = (x[:, 0] - x[:, 1]) / 2 * amount
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Loudness — BS.1770 K-weighting + integrated gating
# ══════════════════════════════════════════════════════════════════════════════


def _k_weight_response(n_fft: int, sr: int) -> np.ndarray:
    """Combined BS.1770 pre-filter + RLB weighting magnitude at rfft bins."""
    # Stage 1: high shelf (+4 dB above ~1.68 kHz); stage 2: high-pass ~38 Hz.
    b1, a1 = biquad("highshelf", 1681.974, 0.707, 3.9998439, sr)
    b2, a2 = biquad("highpass", 38.135, 0.5, 0.0, sr)
    return _response(b1, a1, n_fft) * _response(b2, a2, n_fft)


def loudness(x: np.ndarray, sr: int = SR) -> dict:
    """Integrated loudness (gated), momentary max, and true-peak-ish max."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] < 1:
        return {"integrated_lufs": float("-inf"), "momentary_max_lufs": float("-inf"),
                "peak_dbfs": float("-inf")}
    n = 1 << (x.shape[0] - 1).bit_length()
    gain = _k_weight_response(n, sr)
    weighted = np.fft.irfft(np.fft.rfft(x, n=n, axis=0) * gain[:, None], n=n,
                          axis=0)[:x.shape[0]]
    block = int(0.4 * sr)
    hop = block // 4
    energies = []
    for start in range(0, max(1, weighted.shape[0] - block + 1), hop):
        seg = weighted[start:start + block]
        if seg.shape[0] < block:
            break
        energies.append(float((seg ** 2).sum(axis=0).mean() if seg.ndim == 1
                              else (seg ** 2).mean()))
    energies = np.asarray(energies) if energies else np.zeros(1)
    block_lufs = -0.691 + 10 * np.log10(np.maximum(energies, 1e-12))
    gated = energies[block_lufs > -70]
    if gated.size:
        rel = -0.691 + 10 * np.log10(gated.mean()) - 10
        gated = gated[block_lufs[block_lufs > -70] > rel]
    integrated = -0.691 + 10 * np.log10(max(gated.mean() if gated.size else 1e-12, 1e-12))
    return {"integrated_lufs": float(integrated),
            "momentary_max_lufs": float(block_lufs.max() if block_lufs.size else integrated),
            "peak_dbfs": float(_db(np.abs(x).max()))}


def normalize_lufs(x: np.ndarray, target: float = -14.0, sr: int = SR) -> np.ndarray:
    """Apply constant gain so integrated loudness hits the target."""
    current = loudness(x, sr)["integrated_lufs"]
    if not math.isfinite(current):
        return np.asarray(x, dtype=np.float32).copy()
    return (np.asarray(x, dtype=np.float32) * _undb(target - current)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Alignment, panning, binaural, resampling
# ══════════════════════════════════════════════════════════════════════════════


def align_offset(reference: np.ndarray, candidate: np.ndarray,
                 max_lag_ms: float = 2000.0, sr: int = SR) -> int:
    """Samples by which ``candidate`` trails ``reference`` (positive = late)."""
    def onset_env(x):
        mono = np.abs(x).mean(axis=1) if x.ndim == 2 else np.abs(x)
        hop = 64
        frames = mono[:mono.shape[0] // hop * hop].reshape(-1, hop).mean(axis=1)
        env = np.diff(frames, prepend=frames[0])
        return np.clip(env, 0, None)

    a, b = onset_env(reference), onset_env(candidate)
    n = 1 << (a.shape[0] + b.shape[0]).bit_length()
    corr = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    hop = 64
    max_frames = min(int(max_lag_ms * sr / 1000 / hop), min(a.shape[0], b.shape[0]) - 1)
    cands = np.concatenate([corr[:max_frames + 1], corr[n - max_frames:]])
    best = int(np.argmax(cands))
    lag_frames = best if best <= max_frames else best - (2 * max_frames + 1)
    return int(np.clip(-lag_frames * hop,
                       -max_lag_ms * sr / 1000, max_lag_ms * sr / 1000))


CHANNEL_LAYOUTS = {
    "stereo": ["L", "R"],
    "5.1": ["L", "R", "C", "LFE", "Ls", "Rs"],
    "7.1": ["L", "R", "C", "LFE", "Lss", "Rss", "Ls", "Rs"],
}

# Speaker azimuths per layout (degrees; LFE excluded from panning).
SPEAKER_ANGLES = {
    "stereo": [("L", -30), ("R", 30)],
    "5.1": [("Ls", -110), ("L", -30), ("C", 0), ("R", 30), ("Rs", 110)],
    "7.1": [("Lss", -150), ("Ls", -90), ("L", -30), ("C", 0),
            ("R", 30), ("Rs", 90), ("Rss", 150)],
}


def pan_multichannel(x: np.ndarray, azimuth: float = 0.0, layout: str = "5.1",
                     lfe: float = 0.0) -> np.ndarray:
    """Constant-power pair panning of a stem into a surround bed.

    ``azimuth`` degrees: 0 = front center, negative = left, positive = right,
    ±180 = rear. Returns [samples, channels] in CHANNEL_LAYOUTS[layout] order.
    """
    x = np.asarray(x, dtype=np.float32)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    channels = CHANNEL_LAYOUTS[layout]
    gains = {name: 0.0 for name in channels}
    speakers = SPEAKER_ANGLES[layout]
    a = max(-180.0, min(180.0, float(azimuth)))
    ring = sorted(speakers, key=lambda item: item[1])
    ring.append((ring[0][0], ring[0][1] + 360.0))  # wraparound pair
    for i in range(len(ring) - 1):
        name_a, ang_a = ring[i]
        name_b, ang_b = ring[i + 1]
        if ang_a <= a <= ang_b or (a >= ang_a - 360 and a <= ang_b - 360):
            if a < ang_a:
                a += 360
            span = max(ang_b - ang_a, 1e-6)
            t = (a - ang_a) / span
            gains[name_a] = math.cos(t * math.pi / 2)
            gains[name_b] = math.sin(t * math.pi / 2)
            break
    if "LFE" in gains:
        gains["LFE"] = lfe
    return mono[:, None] * np.asarray([gains[c] for c in channels], dtype=np.float32)


def binaural(x: np.ndarray, azimuth: float = 0.0, elevation: float = 0.0,
             sr: int = SR) -> np.ndarray:
    """Virtual-speaker binaural approximation: ITD + ILD + head-shadow filter.

    This is a lightweight psychoacoustic render, not measured-HRTF Atmos.
    """
    x = np.asarray(x, dtype=np.float32)
    mono = x.mean(axis=1) if x.ndim == 2 else np.asarray(x, dtype=np.float32)
    a = math.radians(np.clip(azimuth, -90, 90))
    itd = 0.00062 * math.sin(a)  # seconds, max ~0.62 ms
    near = 0.5 + 0.5 * math.cos(a) / 1.0
    far = 1.0 - near + 0.15 * math.cos(a) ** 2
    left_gain, right_gain = (near, far) if a < 0 else (far, near)
    n = mono.shape[0]
    shift = int(round(itd * sr))
    left = np.roll(mono, -shift if shift < 0 else 0) * left_gain
    right = np.roll(mono, shift if shift > 0 else 0) * right_gain
    # Head-shadow: lowpass the far ear (FFT-domain, 3 kHz).
    n_fft = 1 << (n - 1).bit_length()
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    shadow = 1 / (1 + (freqs / 3000) ** 2 * abs(math.sin(a)))
    far_ch = right if a < 0 else left
    far_spec = np.fft.rfft(far_ch, n_fft) * shadow
    far_ch = np.fft.irfft(far_spec, n_fft)[:n]
    if a < 0:
        right = far_ch
    else:
        left = far_ch
    out = np.stack([left, right], axis=1)
    if elevation:
        damp = 1 - min(abs(elevation) / 90, 1) * 0.3
        out *= damp
    return out.astype(np.float32)


def resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """FFT-domain band-limited resample (whole-signal; deterministic)."""
    x = np.asarray(x, dtype=np.float32)
    if src_sr == dst_sr:
        return x.copy()
    n_src = x.shape[0]
    n_dst = int(round(n_src * dst_sr / src_sr))
    spec = np.fft.rfft(x, axis=0)
    n_spec_src = spec.shape[0]
    n_spec_dst = n_dst // 2 + 1
    out_spec = np.zeros((n_spec_dst, spec.shape[1] if spec.ndim == 2 else 1),
                        dtype=spec.dtype)
    keep = min(n_spec_src, n_spec_dst)
    out_spec[:keep] = spec[:keep]
    out = np.fft.irfft(out_spec, n=n_dst, axis=0) * (n_dst / n_src)
    return out.astype(np.float32)
