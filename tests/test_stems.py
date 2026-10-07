"""Stem orchestration against a fake pipeline — no model, no GPU."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yue2 import arrange, score, stems  # noqa: E402
from yue2.storage import verify_result, write_json  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCORE_TEXT = (EXAMPLES / "score.abc").read_text()
REQUEST = {"style": "pop, grand piano, drums", "lyrics": "la la",
           "cot": "full", "seed": 831001}


class FakeResult:
    """SongResult-shaped artifact writer for the fake engine."""

    def __init__(self, audio, sample_rate=48000):
        self.audio = audio
        self.sample_rate = sample_rate
        self.request_identity = "fake" + "0" * 60
        self.truncated = {"abc": False, "semantic": False}

    def save_artifacts(self, directory):
        import soundfile as sf
        from yue2.storage import collect_hashes
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sf.write(directory / "audio.flac", self.audio, self.sample_rate,
                 subtype="PCM_24")
        for name in ("prefix.npy", "semantic.npy", "latent.npy",
                     "abc_tokens.npy"):
            np.save(directory / name, np.zeros(4, dtype=np.int32))
        write_json(directory / "request.json", REQUEST)
        write_json(directory / "config.json", {})
        write_json(directory / "plan_manifest.json",
                   {"plan.json": "0" * 64})
        write_json(directory / "result.json",
                   {"status": "complete", "identity": self.request_identity,
                    "truncated": self.truncated,
                    "sample_rate": self.sample_rate,
                    "audio_seconds": len(self.audio) / self.sample_rate,
                    "weights": {}, "timing": {},
                    "artifacts": collect_hashes(directory)})
        return {"identity": self.request_identity}


class FakePipe:
    """Records every render call; returns a distinct sine per call."""

    def __init__(self):
        self.calls = []
        self.count = 0

    def __call__(self, style=None, lyrics=None, **kwargs):
        self.calls.append({"style": style, "lyrics": lyrics, **kwargs})
        self.count += 1
        t = np.arange(48000) / 48000
        audio = np.stack([0.2 * np.sin(2 * np.pi * (300 + self.count * 40) * t)] * 2,
                         axis=1).astype(np.float32)
        return FakeResult(audio)


def _arrangement():
    return {"tracks": [arrange.make_track("lead_vocal"),
                       arrange.make_track("grand_piano"),
                       arrange.make_kit("loop")],
            "buses": [], "fx_sends": []}


def test_score_variant_conditioning():
    parsed = score.parse(SCORE_TEXT)
    vocal = score.parse(stems.score_variant(
        SCORE_TEXT, {"role": "vocal", "line": "melody"}))
    assert score.note_events(vocal, "Vocal") == score.note_events(parsed, "Vocal")
    assert score.note_events(vocal, "Ins") == []
    ins = score.parse(stems.score_variant(
        SCORE_TEXT, {"role": "stem", "line": "melody"}))
    assert score.note_events(ins, "Vocal") == []
    assert not ins.voices["Vocal"].chords
    chords = score.parse(stems.score_variant(
        SCORE_TEXT, {"role": "stem", "line": "chords"}))
    assert not chords.voices["Vocal"].notes
    assert chords.voices["Vocal"].chords


def test_score_variant_synthesized_parts():
    """Each non-melody line gets its own playable part — not a shared grid."""
    parsed = score.parse(SCORE_TEXT)
    variants = {}
    for line in ("bass", "chords", "counter"):
        variants[line] = score.parse(stems.score_variant(
            SCORE_TEXT, {"role": "stem", "line": line}))
    # Distinct conditioning per line.
    texts = {l: v.voices["Ins"].notes for l, v in variants.items()}
    assert len({str(n) for n in texts.values()}) == 3
    for v in variants.values():
        assert not score.note_events(v, "Vocal")          # vocal silenced
        assert v.voices["Vocal"].chords                    # harmony stays
        assert score.note_events(v, "Ins")                 # real notes
        assert v.voices["Ins"].bars == parsed.voices["Ins"].bars
    bass_pitches = {p for _, p, _ in score.note_events(variants["bass"], "Ins")}
    assert all(36 <= p <= 47 for p in bass_pitches)        # bass register
    roots = {c[3] for c in score.chord_events(parsed)}
    assert bass_pitches <= {36 + (r - 36) % 12 for r in roots}


def test_score_variant_rhythm_and_divisi():
    kit = score.parse(stems.score_variant(
        SCORE_TEXT, {"role": "stem", "line": "rhythm", "part": "kick"}))
    assert score.note_events(kit, "Ins") == []
    assert kit.voices["Vocal"].chords
    a = score.parse(stems.score_variant(
        SCORE_TEXT, {"role": "stem", "line": "counter", "div_index": 0}))
    b = score.parse(stems.score_variant(
        SCORE_TEXT, {"role": "stem", "line": "counter", "div_index": 1}))
    pa = [p for _, p, _ in score.note_events(a, "Ins")]
    pb = [p for _, p, _ in score.note_events(b, "Ins")]
    assert [p - 12 for p in pa] == pb


def test_expand_leaves_skips_bypassed():
    arrangement = {"tracks": [
        arrange.make_track("grand_piano"),
        {**arrange.make_track("bass_electric"), "enabled": False},
        {**arrange.make_kit("standard"), "enabled": False}],
        "buses": [], "fx_sends": []}
    assert [l["name"] for l in arrange.expand_leaves(arrangement)] == [
        "grand_piano"]


def test_render_one_pass_per_leaf(tmp_path):
    pipe = FakePipe()
    session = stems.StemSession(tmp_path)
    leaves = arrange.expand_leaves(_arrangement())
    report = session.render(pipe, REQUEST, _arrangement(), SCORE_TEXT)
    assert pipe.count == len(leaves)
    assert set(report["stems"]) == {l["name"] for l in leaves}
    for leaf in leaves:
        result = verify_result(tmp_path / "stems" / leaf["name"])
        assert result["status"] == "complete"


def test_styles_and_seeds(tmp_path):
    pipe = FakePipe()
    stems.StemSession(tmp_path).render(pipe, REQUEST, _arrangement(), SCORE_TEXT)
    by_leaf = {}
    for call in pipe.calls:
        by_leaf[call["seed"] - REQUEST["seed"]] = call
    styles = [c["style"] for c in pipe.calls]
    assert all(REQUEST["style"].split(",")[0] in s for s in styles)
    vocal_call = [c for c in pipe.calls if "lead vocal" in c["style"]][0]
    assert vocal_call["lyrics"] == REQUEST["lyrics"]
    non_vocal = [c for c in pipe.calls if "lead vocal" not in c["style"]]
    assert all(c["lyrics"] == "" for c in non_vocal)


def test_resume_skips_verified(tmp_path):
    pipe = FakePipe()
    session = stems.StemSession(tmp_path)
    session.render(pipe, REQUEST, _arrangement(), SCORE_TEXT)
    first_count = pipe.count
    session.render(pipe, REQUEST, _arrangement(), SCORE_TEXT)
    assert pipe.count == first_count  # nothing re-rendered


def test_tracks_subset(tmp_path):
    pipe = FakePipe()
    stems.StemSession(tmp_path).render(
        pipe, REQUEST, _arrangement(), SCORE_TEXT, tracks=["grand_piano"])
    assert pipe.count == 1


def test_cancelled_stops(tmp_path):
    pipe = FakePipe()
    with pytest.raises(InterruptedError):
        stems.StemSession(tmp_path).render(
            pipe, REQUEST, _arrangement(), SCORE_TEXT, cancelled=lambda: True)
    assert pipe.count == 0


def test_qc_report_fields(tmp_path):
    pipe = FakePipe()
    report = stems.StemSession(tmp_path).render(
        pipe, REQUEST, _arrangement(), SCORE_TEXT)
    for name, entry in report["stems"].items():
        assert entry["status"] == "rendered"
        qc = entry["qc"]
        assert "duration_ratio" in qc and "warnings" in qc
        assert qc["isolation"] == "guided_render_not_guaranteed"


def test_genre_presets_and_picks():
    """Genre styles expand to a real band; explicit picks bypass hints."""
    from yue2.score import instrument_hints
    assert "genre:house" in instrument_hints("2026 house groove")
    arrangement = arrange.arrangement_from_picks(
        ["grand_piano", "drums", "bass_electric"])
    names = [t["name"] for t in arrangement["tracks"]]
    assert names == ["grand_piano", "drums", "bass_electric"]
    leaves = arrange.expand_leaves(arrangement)
    leaf_names = [l["name"] for l in leaves]
    # Kits collapse to ONE stem by default — part leaves would all render
    # the same full kit and stack in the mix.
    assert "drums" in leaf_names
    assert not any(n.startswith("drums.") for n in leaf_names)
    # Opting in via split_kit restores per-part leaves.
    arrangement["tracks"][1]["split_kit"] = True
    assert "drums.kick" in [l["name"] for l in
                            arrange.expand_leaves(arrangement)]
    # Genre preset resolves through the same path.
    house = arrange.arrangement_from_picks(["genre:house"])
    assert len(house["tracks"]) >= 5


def test_ode_steps_overrides_config():
    from yue2.protocol import SongRequest
    assert SongRequest(style="x", lyrics="", ode_steps=16).ode_steps == 16
    assert SongRequest(style="x", lyrics="").ode_steps is None
    import pytest as _pt
    with _pt.raises(ValueError):
        SongRequest(style="x", lyrics="", ode_steps=0)
