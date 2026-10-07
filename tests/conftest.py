"""Shared fixtures: a fully-formed fake session directory."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture()
def session_dir(tmp_path):
    """Session dir with score, arrangement, session.json and fake stems."""
    import soundfile as sf
    from yue2 import arrange, mix, score
    from yue2.storage import write_json

    directory = tmp_path / "session"
    (directory / "stems").mkdir(parents=True)
    score_text = (EXAMPLES / "score.abc").read_text()
    (directory / "score.abc").write_text(score_text)
    request = {"style": "pop, grand piano, drums", "lyrics": "la",
               "cot": "full", "seed": 831001}
    write_json(directory / "request.json", request)
    parsed = score.parse(score_text)
    arrangement = {"tracks": [arrange.make_track("lead_vocal"),
                              arrange.make_track("grand_piano"),
                              arrange.make_track("synthbass"),
                              arrange.make_kit("loop")],
                   "buses": [{"name": "drums", "members": ["drums.*"],
                              "sends": []},
                             {"name": "music", "members": ["grand_piano"],
                              "sends": ["reverb"]},
                             {"name": "low_end", "members": ["synthbass"],
                              "sends": []}],
                   "fx_sends": [{"name": "reverb", "type": "reverb",
                                 "rt60": 0.8, "predelay_ms": 10}]}
    arrange.save_arrangement(directory / "arrangement.json", arrangement)
    mix.save_session(directory, mix.default_session(arrangement))
    t = np.arange(48000) / 48000
    for index, leaf in enumerate(arrange.expand_leaves(arrangement)):
        leaf_dir = directory / "stems" / leaf["name"]
        leaf_dir.mkdir(parents=True)
        freq = 160 if leaf.get("register") == "low" else 330 + index * 110
        audio = np.stack([0.2 * np.sin(2 * np.pi * freq * t)] * 2,
                         axis=1).astype(np.float32)
        sf.write(leaf_dir / "audio.flac", audio, 48000, subtype="PCM_24")
    return directory
