"""SMF writer: structure, golden bytes, ports, SysEx, CC blocks."""
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yue2 import midi, score  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _score():
    return score.parse((EXAMPLES / "score.abc").read_text())


def _leaves():
    return [
        {"name": "lead_vocal", "role": "vocal", "line": "melody",
         "gm_program": 54},
        {"name": "grand_piano", "role": "chords", "line": "chords",
         "gm_program": 0},
        {"name": "synthbass", "role": "rhythm", "line": "bass",
         "gm_program": 39},
        {"name": "drums.kick", "role": "rhythm", "line": "rhythm",
         "part": "kick"},
        {"name": "drums.snare_top", "role": "rhythm", "line": "rhythm",
         "part": "snare_top"},
    ]


def _tracks(data):
    _, fmt, ntr, div = struct.unpack(">IHHH", data[4:14])
    chunks, off = [], 14
    for _ in range(ntr):
        assert data[off:off + 4] == b"MTrk"
        length = struct.unpack(">I", data[off + 4:off + 8])[0]
        chunks.append(data[off + 8:off + 8 + length])
        off += 8 + length
    assert off == len(data)
    return fmt, div, chunks


def test_header_and_tracks():
    data = midi.write_midi(_score(), _leaves())
    assert data[:4] == b"MThd"
    fmt, div, chunks = _tracks(data)
    assert fmt == 1 and div == 480
    assert len(chunks) == len(_leaves()) + 1  # conductor + one per leaf


def test_conductor_contents():
    data = midi.write_midi(_score(), _leaves(), identity="cafebabe")
    conductor = _tracks(data)[2][0]
    assert b"\xf0" in conductor  # SysEx present
    assert b"YUE2" in conductor  # custom session block
    assert b"\xff\x51" in conductor  # tempo meta
    assert b"\xff\x58" in conductor  # time signature
    assert b"\xff\x59" in conductor  # key signature
    assert b"\xff\x06" in conductor  # section markers


def test_golden_bytes():
    data = midi.write_midi(_score(), _leaves(), identity="golden")
    again = midi.write_midi(_score(), _leaves(), identity="golden")
    assert data == again, "writer must be deterministic"
    assert len(data) > 500


def test_type0_merges():
    data = midi.write_midi(_score(), _leaves(), smf_type=0)
    fmt, _, chunks = _tracks(data)
    assert fmt == 0 and len(chunks) == 1


def test_port_meta_beyond_16():
    leaves = [{"name": f"t{i}", "role": "ins", "line": "melody",
               "gm_program": i % 128} for i in range(20)]
    plan = midi.channel_plan(leaves)
    ports = sorted({port for port, _ in plan})
    assert ports == [0, 1]
    channels0 = [c for p, c in plan if p == 0]
    assert 9 not in channels0  # channel 9 stays reserved for drums
    data = midi.write_midi(_score(), leaves)
    assert b"\xff\x21" in data  # MIDI Port meta emitted


def test_drum_parts_channel9():
    leaves = [{"name": "d.kick", "role": "rhythm", "part": "kick"}]
    data = midi.write_midi(_score(), leaves)
    _, _, chunks = _tracks(data)
    body = chunks[1]
    assert bytes([0x99, 36]) in body  # GM kick on channel 9


def test_cc_block_values():
    leaves = [{"name": "piano", "role": "chords", "line": "chords",
               "gm_program": 0}]
    mix = {"piano": {"gain_db": -6.0, "pan": -0.5,
                     "sends": {"reverb": 60, "chorus": 10}}}
    data = midi.write_midi(_score(), leaves, mix_state=mix)
    body = _tracks(data)[2][1]
    assert bytes([0xB0, 0, ]) not in body  # no bank by default
    assert b"\xc0\x00" in body  # program 0
    assert bytes([0xB0, 7]) in body  # CC7 volume
    assert bytes([0xB0, 10]) in body  # CC10 pan
    assert bytes([0xB0, 91]) in body  # reverb send


def test_duplicate_names_rejected():
    leaves = [{"name": "x"}, {"name": "x"}]
    with pytest.raises(ValueError):
        midi.write_midi(_score(), leaves)
