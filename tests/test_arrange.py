"""Arrangement model: catalog integrity, deriving, kit/divisi expansion."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yue2 import arrange, score  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _score():
    return score.parse((EXAMPLES / "score.abc").read_text())


def test_catalog_integrity():
    required = {"family", "line", "gm", "bus", "pan"}
    for name, spec in arrange.INSTRUMENT_CATALOG.items():
        assert required <= set(spec), f"{name} missing fields"
        assert spec["line"] in arrange.LINES
        assert -180 <= spec["pan"] <= 180
        if spec["style_section"] is None:
            assert spec["style_solo"], f"{name} needs a style template"


def test_ensemble_presets_resolve():
    for preset, members in arrange.ENSEMBLE_PRESETS.items():
        for member in members:
            assert member in arrange.INSTRUMENT_CATALOG, \
                f"{preset}: unknown instrument {member}"


def test_kit_presets_have_styles():
    for kit, parts in arrange.KIT_PRESETS.items():
        for part in parts:
            assert part in arrange.KIT_PART_STYLES, f"{kit}: {part} needs a style"


def test_make_track_defaults():
    track = arrange.make_track("violins_i")
    assert track["ensemble"] == "section"
    assert "first violins" in track["style_suffix"]
    assert track["gm_program"] == 48
    assert track["pan_azimuth"] == -30


def test_expand_kit_and_divisi():
    arrangement = {"tracks": [arrange.make_kit("full"),
                              arrange.make_track("violins_i", divisi=2)]}
    leaves = arrange.expand_leaves(arrangement)
    names = [leaf["name"] for leaf in leaves]
    # Kits collapse to one stem by default — every part render shares
    # identical conditioning and produces the full kit anyway.
    assert "drums" in names and not any(n.startswith("drums.") for n in names)
    # split_kit opts back into per-part leaves.
    arrangement["tracks"][0]["split_kit"] = True
    names = [leaf["name"] for leaf in arrange.expand_leaves(arrangement)]
    assert "drums.kick_in" in names and "drums.snare_bottom" in names
    assert "violins_i.div_a" in names and "violins_i.div_b" in names
    assert len(names) == len(set(names))


def test_derive_arrangement():
    a = arrange.derive_arrangement(
        _score(), {"style": "pop, grand piano, strings, drums"})
    names = [t["name"] for t in a["tracks"]]
    assert "lead_vocal" in names
    assert "grand_piano" in names
    assert "drums" in names
    assert any(b["name"] == "strings" for b in a["buses"])
    assert a["fx_sends"]


def test_arrangement_roundtrip(tmp_path):
    a = arrange.derive_arrangement(_score(), {"style": "piano ballad"})
    path = arrange.save_arrangement(tmp_path / "arrangement.json", a)
    loaded = arrange.load_arrangement(path)
    assert loaded["tracks"] == a["tracks"]


def test_duplicate_leaf_names_rejected():
    arrangement = {"tracks": [arrange.make_track("grand_piano", name="x"),
                              arrange.make_track("rhodes", name="x")]}
    with pytest.raises(ValueError):
        arrange.expand_leaves(arrangement)
