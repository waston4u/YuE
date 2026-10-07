"""Export layer: bundles, custom picks, formats, manifests, path safety."""
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yue2 import deliver  # noqa: E402
from yue2.mix import render_session  # noqa: E402


@pytest.fixture()
def mixed(session_dir):
    render_session(session_dir)
    return session_dir


def _names(zip_path):
    return set(zipfile.ZipFile(zip_path).namelist())


def test_bundle_trackout(mixed, tmp_path):
    out = deliver.deliver(mixed, bundle="trackout", out=tmp_path / "t.zip")
    names = _names(out)
    assert "MANIFEST.json" in names
    assert any(n.startswith("stems/") and n.endswith("audio.flac")
               for n in names)
    assert not any("master.flac" == n for n in names)


def test_bundle_everything(mixed, tmp_path):
    out = deliver.deliver(mixed, bundle="everything", out=tmp_path / "e.zip")
    names = _names(out)
    assert {"master.flac", "midi/song.mid", "arrangement.json",
            "MANIFEST.json"} <= names


def test_bundle_stereo_and_binaural(mixed, tmp_path):
    stereo = _names(deliver.deliver(mixed, bundle="stereo",
                                    out=tmp_path / "s.zip"))
    assert stereo == {"master.flac", "MANIFEST.json"}
    binaural = _names(deliver.deliver(mixed, bundle="binaural",
                                      out=tmp_path / "b.zip"))
    assert binaural == {"immersive/binaural.flac", "MANIFEST.json"}


def test_bundle_midi(mixed, tmp_path):
    names = _names(deliver.deliver(mixed, bundle="midi",
                                   out=tmp_path / "m.zip"))
    assert "midi/song.mid" in names and "score.abc" in names


def test_custom_files(mixed, tmp_path):
    names = _names(deliver.deliver(
        mixed, files=["master.flac", "arrangement.json"],
        out=tmp_path / "c.zip"))
    assert names == {"master.flac", "arrangement.json", "MANIFEST.json"}


def test_manifest_contents(mixed, tmp_path):
    import json
    out = deliver.deliver(mixed, bundle="stereo", out=tmp_path / "s.zip")
    manifest = json.loads(zipfile.ZipFile(out).read("MANIFEST.json"))
    entry = manifest["files"]["master.flac"]
    assert len(entry["sha256"]) == 64 and entry["bytes"] > 0
    assert manifest["export"]["audio_format"] == "flac-24"


def test_format_conversion(mixed, tmp_path):
    import soundfile as sf
    out = deliver.deliver(mixed, bundle="stereo", audio_format="wav-16",
                          sample_rate=44100, out=tmp_path / "w.zip")
    names = _names(out)
    assert "master.wav" in names
    with zipfile.ZipFile(out) as zf:
        zf.extract("master.wav", tmp_path)
    data, sr = sf.read(tmp_path / "master.wav", always_2d=True)
    assert sr == 44100


def test_path_traversal_rejected(mixed, tmp_path):
    with pytest.raises(ValueError):
        deliver.deliver(mixed, files=["../outside.txt"],
                        out=tmp_path / "x.zip")


def test_unknown_bundle_rejected(mixed, tmp_path):
    with pytest.raises(ValueError):
        deliver.deliver(mixed, bundle="nope", out=tmp_path / "x.zip")


def test_zip_extension_required(mixed, tmp_path):
    with pytest.raises(ValueError):
        deliver.deliver(mixed, bundle="stereo", out=tmp_path / "out.bin")


def test_formats_reported():
    formats = deliver.available_audio_formats()
    assert "flac-24" in formats and "wav-24" in formats
