"""Selective export: named bundles or arbitrary file picks → manifest'd ZIP.

Canonical session artifacts stay FLAC/SMF-1; conversion happens at export
time. Only files inside the session directory can ship — no model weights,
no credentials, no paths outside the session root.
"""
from __future__ import annotations

import fnmatch
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


from .storage import sha256_file, write_json

BUNDLES = {
    "everything": ["**"],
    "midi": ["score.abc", "midi/**"],
    "trackout": ["stems/*/audio.flac", "stems/*/qc.json", "stems/*/score.abc",
                 "stems_report.json"],
    "stereo": ["master.flac"],
    "binaural": ["immersive/binaural.flac"],
    "immersive": ["immersive/**"],
    "buses": ["buses/**"],
    "fx": ["fx/**"],
    "session": ["arrangement.json", "session.json", "report.json",
                "stems_report.json", "request.json", "score.abc",
                "plan.json", "midi/song.mid"],
}

AUDIO_FORMATS = {
    "flac-24": ("flac", "PCM_24"), "flac-16": ("flac", "PCM_16"),
    "wav-32f": ("wav", "FLOAT"), "wav-24": ("wav", "PCM_24"),
    "wav-16": ("wav", "PCM_16"), "mp3": ("mp3", None),
    "m4a": ("m4a", None),
}
AUDIO_SUFFIXES = {".flac", ".wav", ".aif", ".aiff", ".mp3", ".m4a"}


def available_audio_formats() -> list[str]:
    """Formats this environment can actually write."""
    import soundfile as sf
    supported = {name.upper() for name in sf.available_formats()}
    formats = [name for name, (suffix, _) in AUDIO_FORMATS.items()
               if suffix.upper() in supported]
    formats.append("m4a")  # afconvert — macOS-only product
    return sorted(set(formats))


def list_files(session_dir, bundle: str | None = None,
               files: list[str] | None = None) -> list[Path]:
    """Resolve a bundle name or explicit relative paths to session files."""
    session_dir = Path(session_dir).resolve()
    if bundle is not None and files is not None:
        raise ValueError("Pass bundle or files, not both")
    if bundle is not None:
        if bundle not in BUNDLES:
            raise ValueError(f"Unknown bundle {bundle!r}; choose from {sorted(BUNDLES)}")
        patterns = BUNDLES[bundle]
        found = [p for p in sorted(session_dir.rglob("*")) if p.is_file()
                 and "exports" not in p.relative_to(session_dir).parts
                 and any(fnmatch.fnmatch(str(p.relative_to(session_dir)), pat)
                         or fnmatch.fnmatch(str(p.relative_to(session_dir)),
                                            pat.rstrip("*"))
                         for pat in patterns)]
    else:
        if not files:
            raise ValueError("Provide a bundle name or a file selection")
        found = []
        for name in files:
            p = (session_dir / name).resolve()
            if not p.is_file() or not p.is_relative_to(session_dir):
                raise ValueError(f"Not a session file: {name!r}")
            found.append(p)
    return found


def _convert_audio(source: Path, dest_dir: Path, rel: str, audio_format: str,
                   sample_rate: int) -> Path:
    import soundfile as sf
    from .dsp import resample
    data, sr = sf.read(source, dtype="float32", always_2d=True)
    if sr != sample_rate:
        data = resample(data, sr, sample_rate)
    suffix, subtype = AUDIO_FORMATS[audio_format]
    target = dest_dir / (str(Path(rel).with_suffix("")) + "." + suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    if audio_format == "m4a":
        temp_wav = target.with_suffix(".wav")
        sf.write(temp_wav, data, sample_rate, subtype="FLOAT")
        try:
            subprocess.run(["afconvert", "-f", "m4af", "-d", "aac",
                            str(temp_wav), str(target)],
                           check=True, capture_output=True)
        finally:
            temp_wav.unlink(missing_ok=True)
    else:
        sf.write(target, data, sample_rate, subtype=subtype)
    return target


def _convert_midi(session_dir: Path, dest_dir: Path, rel: str,
                  midi_type: int) -> Path:
    from .arrange import load_arrangement
    from .midi import write_midi
    from .score import parse
    arrangement = load_arrangement(session_dir / "arrangement.json")
    score = parse((session_dir / "score.abc").read_text())
    from .arrange import expand_leaves
    target = dest_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(write_midi(score, expand_leaves(arrangement),
                                  smf_type=midi_type))
    return target


def deliver(session_dir, *, bundle: str | None = None,
            files: list[str] | None = None, audio_format: str = "flac-24",
            sample_rate: int = 48000, midi_type: int = 1,
            out=None) -> Path:
    """Export a bundle or file selection as a manifest'd ZIP."""
    session_dir = Path(session_dir).resolve()
    if audio_format not in AUDIO_FORMATS:
        raise ValueError(f"audio_format must be one of {sorted(AUDIO_FORMATS)}")
    if audio_format not in available_audio_formats():
        raise ValueError(f"{audio_format} is not writable in this environment")
    if sample_rate not in (48000, 44100):
        raise ValueError("sample_rate must be 48000 or 44100")
    if midi_type not in (0, 1):
        raise ValueError("midi_type must be 0 or 1")
    selection = list_files(session_dir, bundle, files)
    if not selection:
        raise ValueError("Selection matched no files — has the session been rendered/mixed?")

    label = bundle or "custom"
    out = Path(out) if out else session_dir / f"deliver_{label}.zip"
    if out.suffix != ".zip":
        raise ValueError("Export path must end in .zip")
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest_entries = {}
    options = {"audio_format": audio_format, "sample_rate": sample_rate,
               "midi_type": midi_type, "bundle": bundle, "files": files}
    with tempfile.TemporaryDirectory() as stage:
        stage_dir = Path(stage)
        packed = []
        for source in selection:
            rel = str(source.relative_to(session_dir))
            suffix = source.suffix.lower()
            if suffix in AUDIO_SUFFIXES and (
                    audio_format != "flac-24" or sample_rate != 48000
                    or suffix != ".flac"):
                packed.append((_convert_audio(source, stage_dir, rel,
                                              audio_format, sample_rate), rel))
            elif suffix == ".mid" and midi_type == 0:
                packed.append((_convert_midi(session_dir, stage_dir, rel,
                                            midi_type), rel))
            else:
                target = stage_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                packed.append((target, rel))
        for path, rel in packed:
            arcname = str(Path(rel).with_suffix("")) + path.suffix \
                if path.suffix.lower() in AUDIO_SUFFIXES else rel
            manifest_entries[arcname] = {"sha256": sha256_file(path),
                                         "bytes": path.stat().st_size,
                                         "source": rel}
        manifest = {"session": session_dir.name, "export": options,
                    "files": manifest_entries}
        write_json(stage_dir / "MANIFEST.json", manifest)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as bundle_zip:
            for path, rel in packed:
                arcname = str(Path(rel).with_suffix("")) + path.suffix \
                    if path.suffix.lower() in AUDIO_SUFFIXES else rel
                bundle_zip.write(path, arcname)
            bundle_zip.write(stage_dir / "MANIFEST.json", "MANIFEST.json")
    return out
