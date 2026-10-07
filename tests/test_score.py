"""Score engine: parse, surgery, events, hints — parity with the skill tools."""
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "skills" / "yue2-music" / "scripts"))

from yue2 import score  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture()
def example():
    return score.parse((EXAMPLES / "score.abc").read_text())


def test_parse_structure(example):
    assert example.bpm == 88
    assert "Vocal" in example.voices and "Ins" in example.voices
    vocal = example.voices["Vocal"]
    assert vocal.notes, "Vocal voice should carry notes"
    assert vocal.chords, "Vocal voice should carry chord symbols"


def test_exact_fraction_timing(example):
    notes = score.note_events(example, "Vocal")
    onsets = [onset for onset, _, _ in notes]
    assert onsets == sorted(onsets)
    assert all(isinstance(o, Fraction) and isinstance(d, Fraction)
               for o, _, d in notes)
    assert all(d > 0 for _, _, d in notes)


def test_meter_and_key_maps(example):
    assert score.meter_map(example)[0] == (Fraction(0), (4, 4))
    assert score.key_map(example)[0] == (Fraction(0), "C")


def test_sections(example):
    assert example.sections
    names = [name for _, name in example.sections]
    assert names


def test_silence_voice(example):
    text = score.silence_voice(example.text, "Vocal")
    silenced = score.parse(text)
    assert score.note_events(silenced, "Vocal") == []
    assert silenced.voices["Vocal"].chords  # chord symbols survive silence
    assert score.note_events(silenced, "Ins") == \
        score.note_events(example, "Ins")


def test_chord_filtering(example):
    dropped = score.parse(score.drop_chords(example.text))
    assert not dropped.voices["Vocal"].chords
    assert score.note_events(dropped, "Vocal") == \
        score.note_events(example, "Vocal")
    only = score.parse(score.keep_chords_only(example.text))
    assert only.voices["Vocal"].chords
    assert not score.note_events(only, "Vocal")
    assert not score.note_events(only, "Ins")


def test_variant_for_line(example):
    chords_only = score.parse(score.variant_for_line(example.text, "chords"))
    assert not chords_only.voices["Vocal"].notes
    melody = score.parse(score.variant_for_line(example.text, "melody"))
    assert melody.voices["Vocal"].notes == example.voices["Vocal"].notes


def test_chord_events(example):
    events = score.chord_events(example)
    assert events
    for onset, text, root_name, root_pc, quality, bass_pc in events:
        assert 0 <= root_pc < 12
        assert isinstance(onset, Fraction)


def test_chord_pitches():
    assert score.chord_pitches(0, "maj7") == [48, 52, 55, 59]
    slashed = score.chord_pitches(7, "7", bass_pc=2)
    assert slashed[0] == 38  # D bass under a G7 voicing


def test_instrument_hints():
    hints = score.instrument_hints("epic pop, grand piano, strings, 808")
    assert "grand_piano" in hints or "strings" in hints or "synthbass" in hints


def test_parity_with_skill_parser(example):
    abc_tools = pytest.importorskip("abc_tools")
    theirs = abc_tools.parse_abc((EXAMPLES / "score.abc").read_text())
    ours = [list(n) for n in score.note_events(example, "Vocal")]
    assert ours == theirs.voices["Vocal"].notes
    ours_ins = [list(n) for n in score.note_events(example, "Ins")]
    assert ours_ins == theirs.voices["Ins"].notes
    assert example.voices["Vocal"].chords == theirs.voices["Vocal"].chords
    assert example.voices["Vocal"].bars == theirs.voices["Vocal"].bars
    assert example.voices["Vocal"].keys == theirs.voices["Vocal"].keys
