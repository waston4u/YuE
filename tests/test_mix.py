"""Session mixer: buses, FX returns, master, immersive, MIDI deliverables."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_render_session_deliverables(session_dir):
    from yue2.mix import render_session
    report = render_session(session_dir)
    for rel in ("master.flac", "immersive/binaural.flac",
                "immersive/bed_5_1.wav", "immersive/bed_7_1.wav",
                "midi/song.mid", "report.json",
                "buses/drums.flac", "buses/music.flac", "buses/low_end.flac"):
        assert (session_dir / rel).is_file(), f"missing {rel}"
    assert "reverb" in report["fx_returns"]
    assert report["loudness"]["integrated_lufs"] == pytest.approx(-14.0, abs=0.5)
    assert "not Dolby Atmos" in report["immersive_note"]
    assert report["artifacts"]


def test_bed_channel_counts(session_dir):
    import soundfile as sf
    from yue2.mix import render_session
    render_session(session_dir)
    bed51, sr = sf.read(session_dir / "immersive" / "bed_5_1.wav",
                        always_2d=True)
    bed71, _ = sf.read(session_dir / "immersive" / "bed_7_1.wav",
                       always_2d=True)
    assert bed51.shape[1] == 6 and bed71.shape[1] == 8
    assert sr == 48000


def test_mute_removes_stem(session_dir):
    import json
    import soundfile as sf
    from yue2.mix import render_session
    session = json.loads((session_dir / "session.json").read_text())
    session["stems"]["lead_vocal"]["mute"] = True
    (session_dir / "session.json").write_text(json.dumps(session))
    report = render_session(session_dir)
    assert "lead_vocal" not in report["stems_mixed"]


def test_midi_written(session_dir):
    import struct
    from yue2.mix import render_session
    render_session(session_dir)
    data = (session_dir / "midi" / "song.mid").read_bytes()
    assert data[:4] == b"MThd"
    _, fmt, ntr, _ = struct.unpack(">IHHH", data[4:14])
    assert fmt == 1 and ntr >= 5  # conductor + vocal/piano/bass/drum_loop


def test_missing_stems_error(tmp_path):
    from yue2.mix import render_session
    with pytest.raises(Exception):
        render_session(tmp_path)
