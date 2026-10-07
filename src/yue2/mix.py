"""Session mixer: generated stems → bus DAG → deliverables.

Mixes only the separately generated stems — never any source separation.
Every deliverable records its provenance: stems are steered renders, wet FX
are send returns, immersive beds are our own pan renders (not Dolby Atmos).
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

import numpy as np

from .arrange import expand_leaves, load_arrangement
from .dsp import (SR, apply_eq, binaural, biquad, compress, limiter,
                  loudness, ms_widen, normalize_lufs, pan_multichannel,
                  reverb, delay)
from .midi import save_midi
from .score import parse
from .storage import collect_hashes, identity, write_json


def default_session(arrangement: dict) -> dict:
    """Mix state defaults for an arrangement — editable in session.json."""
    leaves = expand_leaves(arrangement)
    return {"version": 1,
            "stems": {leaf["name"]: {"gain_db": 0.0, "pan": leaf.get("pan_azimuth", 0) / 60.0,
                                     "eq": [], "sends": {}, "mute": False, "solo": False}
                      for leaf in leaves},
            "buses": [{**bus, "gain_db": 0.0, "mute": False}
                      for bus in arrangement.get("buses", [])],
            "fx_sends": arrangement.get("fx_sends", []),
            "master": {"target_lufs": -14.0, "ceiling_dbfs": -1.0,
                       "gain_db": 0.0,
                       "eq": [], "comp": {"threshold_db": -14, "ratio": 1.5,
                                          "attack_ms": 20, "release_ms": 150,
                                          "makeup_db": 0}}}


def load_session(session_dir) -> dict:
    session_dir = Path(session_dir)
    session = json.loads((session_dir / "session.json").read_text())
    return session


def save_session(session_dir, session: dict):
    write_json(Path(session_dir) / "session.json", session)


def _undb(db):
    return float(10 ** (db / 20))


def _read_stems(session_dir, leaves):
    """stems/<leaf>/audio.flac → {name: float32 [n,2]}, zero-padded to equal length."""
    import soundfile as sf
    audio = {}
    for leaf in leaves:
        path = Path(session_dir) / "stems" / leaf["name"] / "audio.flac"
        if not path.is_file():
            continue
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != SR:
            raise ValueError(f"{path}: expected {SR} Hz audio, got {sr}")
        audio[leaf["name"]] = data
    if not audio:
        raise ValueError("No rendered stems found; run the stem render first")
    longest = max(x.shape[0] for x in audio.values())
    return {name: np.pad(x, ((0, longest - x.shape[0]), (0, 0)))
            for name, x in audio.items()}, longest


def _bus_members(bus: dict, leaf_names: list[str]) -> list[str]:
    members = []
    for pattern in bus.get("members", []):
        members += [name for name in leaf_names if fnmatch.fnmatch(name, pattern)]
    return [name for name in leaf_names if name in set(members)]


def _stereo_pan(x: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power stereo pan, pan in [-1, 1]."""
    angle = (float(np.clip(pan, -1, 1)) + 1) * np.pi / 4
    return x * np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)


def _apply_stem_state(x, state, sr):
    if state.get("eq"):
        bands = [biquad(kind, f, q, g) for kind, f, q, g in state["eq"]]
        x = apply_eq(x, bands, sr)
    return _stereo_pan(x * _undb(state.get("gain_db", 0.0)), state.get("pan", 0.0))


def _fx_render(fx: dict, source: np.ndarray, sr) -> np.ndarray:
    """Wet-only FX return for a send bus."""
    kind = fx.get("type")
    if kind == "reverb":
        return reverb(source, rt60=fx.get("rt60", 1.8),
                      predelay_ms=fx.get("predelay_ms", 20), mix=1.0, sr=sr)
    if kind == "delay":
        return delay(source, time_ms=fx.get("time_ms", 375),
                     feedback=fx.get("feedback", 0.3), mix=1.0,
                     ping_pong=fx.get("ping_pong", True), sr=sr)
    raise ValueError(f"Unknown FX type {kind!r}")


def render_session(session_dir, *, midi_type: int = 1,
                   on_progress=None) -> dict:
    """Build every deliverable from the session's rendered stems."""
    import soundfile as sf
    session_dir = Path(session_dir)
    session = load_session(session_dir)
    arrangement = load_arrangement(session_dir / "arrangement.json")
    score_text = (session_dir / "score.abc").read_text()
    score = parse(score_text)
    leaves = expand_leaves(arrangement)
    stems, length = _read_stems(session_dir, leaves)
    leaf_names = [leaf["name"] for leaf in leaves if leaf["name"] in stems]
    stem_state = session["stems"]

    soloed = {n for n in leaf_names if stem_state.get(n, {}).get("solo")}
    def audible(name):
        state = stem_state.get(name, {})
        return not state.get("mute") and (not soloed or name in soloed)

    # Stage 1: process each stem (gain/pan/EQ), collect sends.
    processed = {}
    send_inputs = {fx["name"]: np.zeros((length, 2), np.float32)
                   for fx in session.get("fx_sends", [])}
    for name in leaf_names:
        state = stem_state.get(name, {})
        if not audible(name):
            continue
        processed[name] = _apply_stem_state(stems[name], state, SR)
        for send, level in state.get("sends", {}).items():
            if send in send_inputs:
                send_inputs[send] += processed[name] * _undb(level)

    # Stage 2: sum buses; bus sends feed FX too.
    buses_out = {}
    bus_of = {}
    for bus in session.get("buses", []):
        members = [n for n in _bus_members(bus, leaf_names) if n in processed]
        bus_of[bus["name"]] = members
        summed = np.zeros((length, 2), np.float32)
        for name in members:
            summed += processed[name]
        buses_out[bus["name"]] = summed * _undb(bus.get("gain_db", 0.0)) \
            if not bus.get("mute") else np.zeros((length, 2), np.float32)
        for send in bus.get("sends", []):
            if send in send_inputs:
                send_inputs[send] += buses_out[bus["name"]]

    assigned = {n for members in bus_of.values() for n in members}
    direct = np.zeros((length, 2), np.float32)
    for name in leaf_names:
        if name in processed and name not in assigned:
            direct += processed[name]

    # Stage 3: wet-only FX returns.
    fx_out = {}
    for fx in session.get("fx_sends", []):
        source = send_inputs[fx["name"]]
        if np.abs(source).max() > 1e-6:
            wet = _fx_render(fx, source, SR)
            fx_out[fx["name"]] = wet[:length] if wet.shape[0] >= length else \
                np.pad(wet, ((0, length - wet.shape[0]), (0, 0)))

    # Stage 4: master chain → deliverables.
    master = direct.copy()
    for signal in buses_out.values():
        master += signal
    for signal in fx_out.values():
        master += signal
    master_settings = session["master"]
    if master_settings.get("eq"):
        master = apply_eq(master, [biquad(k, f, q, g)
                                   for k, f, q, g in master_settings["eq"]], SR)
    comp = master_settings.get("comp", {})
    master = compress(master, sr=SR, **comp)
    master = ms_widen(master, master_settings.get("width", 1.0))
    master = normalize_lufs(master, master_settings.get("target_lufs", -14.0), SR)
    # Output trim — the master fader. Post-normalize so it shifts loudness
    # deliberately; the limiter still protects the ceiling after it.
    master = master * _undb(master_settings.get("gain_db", 0.0))
    master = limiter(master, master_settings.get("ceiling_dbfs", -1.0), sr=SR)

    # Stage 5: immersive renders from the dry processed stems (seat azimuths).
    leaf_by_name = {leaf["name"]: leaf for leaf in leaves}
    binaural_mix = np.zeros((length, 2), np.float32)
    bed51 = np.zeros((length, 6), np.float32)
    bed71 = np.zeros((length, 8), np.float32)
    for name, signal in processed.items():
        az = float(leaf_by_name[name].get("pan_azimuth") or
                   stem_state.get(name, {}).get("pan", 0) * 60)
        binaural_mix += binaural(signal, az, sr=SR)
        bed51 += pan_multichannel(signal, az, "5.1", lfe=0.3 if
                                  leaf_by_name[name].get("register") == "low" else 0.0)
        bed71 += pan_multichannel(signal, az, "7.1", lfe=0.3 if
                                  leaf_by_name[name].get("register") == "low" else 0.0)
    binaural_mix = limiter(normalize_lufs(binaural_mix, -14.0, SR), -1.0, sr=SR)

    # Stage 6: write everything + manifest.
    if on_progress:
        on_progress("writing", 0, 1, "writing")
    (session_dir / "buses").mkdir(exist_ok=True)
    (session_dir / "fx").mkdir(exist_ok=True)
    (session_dir / "immersive").mkdir(exist_ok=True)
    for name, signal in buses_out.items():
        sf.write(session_dir / "buses" / f"{name}.flac", signal, SR, subtype="PCM_24")
    for name, signal in fx_out.items():
        sf.write(session_dir / "fx" / f"{name}.flac", signal, SR, subtype="PCM_24")
    sf.write(session_dir / "master.flac", master, SR, subtype="PCM_24")
    sf.write(session_dir / "immersive" / "binaural.flac", binaural_mix, SR,
             subtype="PCM_24")
    sf.write(session_dir / "immersive" / "bed_5_1.wav", bed51, SR, subtype="FLOAT")
    sf.write(session_dir / "immersive" / "bed_7_1.wav", bed71, SR, subtype="FLOAT")
    save_midi(session_dir / "midi" / "song.mid", score, leaves,
              smf_type=midi_type,
              mix_state={n: {"gain_db": stem_state.get(n, {}).get("gain_db", 0),
                             "pan": stem_state.get(n, {}).get("pan", 0),
                             "sends": {k: int(127 * _undb(v)) for k, v in
                                       stem_state.get(n, {}).get("sends", {}).items()}}
                         for n in leaf_names},
              identity=identity({"arrangement": arrangement,
                                 "session": session})[:8])

    report = {"status": "mixed", "sample_rate": SR,
              "loudness": loudness(master, SR),
              "binaural_loudness": loudness(binaural_mix, SR),
              "buses": sorted(buses_out), "fx_returns": sorted(fx_out),
              "stems_mixed": sorted(processed), "immersive_note":
                  "binaural approximation + multichannel beds — not Dolby Atmos",
              "session_identity": identity({"arrangement": arrangement,
                                            "session": session}),
              "artifacts": collect_hashes(session_dir, exclude=("report.json",))}
    write_json(session_dir / "report.json", report)
    if on_progress:
        on_progress("done", 1, 1, "done")
    return report
