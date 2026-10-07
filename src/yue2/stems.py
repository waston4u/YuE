"""Per-stem render orchestration: N arrangement leaves → N generation passes.

Every leaf stem is a full, separate ``pipe()`` render conditioned on a
role-appropriate score variant and an instrument-focused style suffix —
never split from a stereo master. Renders are sequential (one GPU),
resumable (verified ``result.json`` artifacts are skipped) and QC-audited
for drift and bleed. Isolation is guided, not guaranteed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .arrange import expand_leaves
from .score import (drop_chords, parse, part_events, rewrite_voice,
                    silence_voice)
from .storage import verify_result, write_json


def score_variant(score_text: str, leaf: dict) -> str:
    """Line/role → per-leaf ABC: each stem is conditioned on its own part.

    The dialect only carries Vocal + Ins voices, so real isolation needs real
    content differences — not style text alone. Vocal stems keep the Vocal
    melody; melodic stems keep the Ins line; harmony leaves get a part
    synthesized from the chord grid (root-pulse basslines, voiced chord
    arpeggios, sustained counter-tones); rhythm leaves keep meter + chords.
    """
    if leaf.get("role") == "vocal":
        return silence_voice(score_text, "Ins")
    if leaf.get("part") or leaf.get("role") == "rhythm" or \
            leaf.get("line") == "rhythm":
        # Unpitched parts read the meter grid; chord symbols carry groove.
        return silence_voice(silence_voice(score_text, "Vocal"), "Ins")
    line = leaf.get("line", "melody")
    if line in {"melody", "ins"}:
        return drop_chords(silence_voice(score_text, "Vocal"))
    events = part_events(parse(score_text), line)
    base = silence_voice(score_text, "Vocal")
    if not events:
        return silence_voice(base, "Ins")
    div = int(leaf.get("div_index") or 0)
    if div:
        events = [(t, p - 12 * div, d) for t, p, d in events]
    return rewrite_voice(base, "Ins", events)


def nominal_seconds(score) -> float:
    """Score duration in seconds from meter/tempo — stem QC baseline."""
    total_quarters = max((v.time for v in score.voices.values()), default=0)
    return float(total_quarters) * 60.0 / score.bpm


def band_energy(audio: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """Fraction of spectral energy inside [lo, hi] Hz."""
    if audio.size == 0:
        return 0.0
    mono = audio.mean(axis=1)
    spec = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(mono.shape[0], 1 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    total = float((spec ** 2).sum())
    return float((spec[mask] ** 2).sum() / total) if total > 0 else 0.0


def stem_qc(audio: np.ndarray, sr: int, nominal: float, leaf: dict,
            reference_offset: int = 0) -> dict:
    """Per-stem quality audit — reports, never claims isolation."""
    seconds = len(audio) / sr
    report = {"seconds": seconds, "nominal_seconds": nominal,
              "duration_ratio": seconds / nominal if nominal else None,
              "align_offset_samples": reference_offset,
              "peak_dbfs": float(20 * np.log10(max(np.abs(audio).max(), 1e-9)))}
    warnings = []
    if nominal and abs(seconds - nominal) / nominal > 0.15:
        warnings.append("duration_drift")
    register = leaf.get("register", "mid")
    bands = {"low": (20, 250), "low-mid": (250, 1000), "mid": (250, 4000),
             "high": (4000, 20000)}
    lo, hi = bands.get(register, (250, 4000))
    in_band = band_energy(audio, sr, lo, hi)
    report["role_band_energy"] = in_band
    if register == "low" and band_energy(audio, sr, 4000, 20000) > 0.3:
        warnings.append("possible_bleed_high_band")
    if register == "high" and band_energy(audio, sr, 20, 120) > 0.3:
        warnings.append("possible_bleed_low_band")
    if leaf.get("role") != "vocal" and in_band < 0.15:
        warnings.append("low_role_band_energy_check_content")
    report["warnings"] = warnings
    report["isolation"] = "guided_render_not_guaranteed"
    return report


def _release_memory() -> None:
    """Reclaim accelerator buffers between stems — MLX's Metal cache and
    torch's MPS cache hold freed blocks, which otherwise grows ~GBs per
    stem until the machine swaps."""
    import gc
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


class StemSession:
    """Renders arrangement leaves into ``stems/<name>/`` artifact folders."""

    def __init__(self, session_dir):
        self.dir = Path(session_dir)
        self.stems_dir = self.dir / "stems"

    def leaf_done(self, leaf: dict) -> bool:
        directory = self.stems_dir / leaf["name"]
        try:
            verify_result(directory)
            return True
        except Exception:
            return False

    def render(self, pipe, request: dict, arrangement: dict, score_text: str,
               *, tracks: list[str] | None = None, cancelled=None,
               on_token=None, on_progress=None, cleanup: str = "none") -> dict:
        """Sequential per-leaf renders; returns the stems QC report."""
        leaves = expand_leaves(arrangement)
        if tracks is not None:
            selected = set(tracks)
            leaves = [l for l in leaves if l["name"] in selected or l["track"] in selected]
            missing = selected - {l["name"] for l in leaves} - {
                t["name"] for t in arrangement.get("tracks", [])}
            if missing:
                raise ValueError(f"Unknown tracks: {sorted(missing)}")
        parsed = parse(score_text)
        nominal = nominal_seconds(parsed)
        base_seed = request["seed"]
        report = {"stems": {}, "nominal_seconds": nominal,
                  "isolation": "guided_render_not_guaranteed"}
        for index, leaf in enumerate(leaves, 1):
            name = leaf["name"]
            if on_progress:
                on_progress(name, index - 1, len(leaves), "queued")
            if cancelled is not None and cancelled():
                raise InterruptedError(f"Cancelled before stem {name}")
            directory = self.stems_dir / name
            if self.leaf_done(leaf):
                report["stems"][name] = {"status": "resumed", "dir": str(directory)}
                if on_progress:
                    on_progress(name, index, len(leaves), "resumed")
                continue
            if on_progress:
                on_progress(name, index - 1, len(leaves), "rendering")
            variant_abc = score_variant(score_text, leaf)
            style = f"{request['style']}, {leaf['style_suffix']}".strip(", ")
            lyrics = request["lyrics"] if leaf.get("role") == "vocal" else ""
            seed = base_seed + int(leaf.get("seed_offset", 0))
            # Contrastive isolation: the CFG negative branch describes the
            # mix the leaf must NOT be — "full band" for a solo part,
            # "instrumental" for vocals. Without it guidance=1.0 and the
            # model renders the whole production into every stem.
            cfg_scale = float(request.get("cfg_scale", 1.5))
            negative_style = self._negative_style(leaf)
            # Codec runs at 25 semantic tokens per audio second (48 kHz /
            # 1920 downsampling) — pass the estimate so progress bars and
            # ETA are real instead of a bare token counter.
            est_tokens = int(nominal * 25)
            result = pipe(style=style, lyrics=lyrics, abc=variant_abc,
                          cot=request.get("cot", "full"), seed=seed,
                          cfg_scale=cfg_scale,
                          negative_style=negative_style,
                          ode_steps=request.get("ode_steps"),
                          ode_method=request.get("ode_method"),
                          cancelled=cancelled,
                          on_token=(lambda ph, t, n=name:
                                    on_token(ph, t, n, est_tokens))
                                   if on_token else None,
                          on_progress=(lambda c, t, n=name:
                                       on_progress(n, c, t, "synthesizing"))
                                      if on_progress else None)
            audio = np.asarray(result.audio, dtype=np.float32)
            if cleanup == "role-eq":
                from .dsp import apply_eq, expander
                audio = apply_eq(audio, self._role_eq(leaf), result.sample_rate)
                audio = expander(audio, -55.0, 2.0, sr=result.sample_rate)
                result.audio = audio  # cleaned audio enters the hashed artifacts
            info = result.save_artifacts(directory)
            qc = stem_qc(audio, result.sample_rate, nominal, leaf)
            write_json(directory / "qc.json", qc)
            report["stems"][name] = {"status": "rendered", "qc": qc,
                                     "identity": info["identity"],
                                     "seed": seed, "style": style}
            if on_progress:
                on_progress(name, index, len(leaves), "done")
            _release_memory()
        write_json(self.dir / "stems_report.json", report)
        return report

    @staticmethod
    def _negative_style(leaf: dict) -> str:
        """The mix the CFG negative branch should describe for this leaf —
        the contrast that pushes the positive decode toward isolation."""
        part = leaf.get("part")
        if part:
            # Split kit part: negative = the rest of the kit.
            return ("full drum kit performance, all drum parts playing "
                    "together")
        if leaf.get("kit"):
            return "melodic instruments, vocals, no drums, no percussion"
        if leaf.get("role") == "vocal":
            return "instrumental only, full band, no vocals, no singing"
        return ("full band production, drums, bass, keys, guitars, "
                "all instruments playing together")

    @staticmethod
    def _role_eq(leaf: dict) -> list:
        """Register-tuned cleanup chain for a leaf (post-clean, not separation)."""
        from .dsp import biquad
        register = leaf.get("register", "mid")
        if leaf.get("part") or leaf.get("role") == "rhythm":
            return [biquad("highpass", 30, 0.707)]
        if register == "low":
            return [biquad("highpass", 30, 0.707), biquad("lowpass", 3000, 0.707)]
        if register == "high":
            return [biquad("highpass", 500, 0.707)]
        if leaf.get("role") == "vocal":
            return [biquad("highpass", 80, 0.707), biquad("peak", 3000, 1.0, 1.5)]
        return [biquad("highpass", 60, 0.707), biquad("lowpass", 16000, 0.707)]
