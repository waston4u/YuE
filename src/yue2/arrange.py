"""Session orchestration: instrument catalog, ensembles, kits, arrangements.

An arrangement is the editable instrument list for a session. Each *track*
is a catalog instrument or a kit/ensemble preset; ``expand_leaves`` resolves
kits and divisi into *leaf stems* — one leaf = one generation pass.

Score dialect reminder: the model emits Vocal/Ins voices + chord symbols, not
N independent parts. A leaf's ``line`` selects which score content conditions
its render (its voice's melody, the chord grid, chord roots, or the meter
grid); ``style_suffix`` steers the instrument. Isolation is guided, never
guaranteed — mic-perspective kit parts are separate performances, not one
kit recording split by microphone.
"""
from __future__ import annotations

from pathlib import Path

from .storage import write_json

LINES = {"melody", "counter", "chords", "bass", "rhythm"}

# ══════════════════════════════════════════════════════════════════════════════
# Instrument catalog — family → leaf instruments with render defaults
# ══════════════════════════════════════════════════════════════════════════════
# {family, line, style_solo, style_section, gm, drum_note, bus, pan, register}


def _entry(family, line, solo, section, gm, bus, pan, register="mid", drum_note=None):
    return {"family": family, "line": line, "style_solo": solo,
            "style_section": section, "gm": gm, "bus": bus, "pan": pan,
            "register": register, "drum_note": drum_note}


INSTRUMENT_CATALOG = {
    # ── Strings ─────────────────────────────────────────────────────────
    "violin_solo": _entry("strings", "melody", "solo violin only, expressive vibrato", None, 40, "strings", -20),
    "violins_i": _entry("strings", "melody", None, "first violins, ensemble strings only", 48, "strings", -30),
    "violins_ii": _entry("strings", "counter", None, "second violins, ensemble strings only", 48, "strings", -15),
    "viola_solo": _entry("strings", "counter", "solo viola only, warm tone", None, 41, "strings", -5),
    "violas": _entry("strings", "counter", None, "viola section, ensemble strings only", 48, "strings", -8),
    "cello_solo": _entry("strings", "counter", "solo cello only, lyrical", None, 42, "strings", 8, "low-mid"),
    "celli": _entry("strings", "counter", None, "cello section, ensemble strings only", 48, "strings", 12, "low-mid"),
    "contrabasses": _entry("strings", "bass", None, "double bass section only", 43, "strings", 45, "low"),
    "harp": _entry("strings", "chords", "concert harp only, arpeggios", None, 46, "strings", 10),
    # ── Woodwinds ───────────────────────────────────────────────────────
    "flute": _entry("woodwinds", "melody", "solo flute only", "flute section only", 73, "woodwinds", -15),
    "piccolo": _entry("woodwinds", "counter", "solo piccolo only", None, 72, "woodwinds", -10, "high"),
    "oboe": _entry("woodwinds", "melody", "solo oboe only", "oboe section only", 68, "woodwinds", -5),
    "clarinet": _entry("woodwinds", "counter", "solo clarinet only", "clarinet section only", 71, "woodwinds", 5),
    "bass_clarinet": _entry("woodwinds", "bass", "solo bass clarinet only", None, 71, "woodwinds", 10, "low"),
    "bassoon": _entry("woodwinds", "bass", "solo bassoon only", "bassoon section only", 70, "woodwinds", 15, "low-mid"),
    "contrabassoon": _entry("woodwinds", "bass", "contrabassoon only", None, 70, "woodwinds", 20, "low"),
    "sax_alto": _entry("woodwinds", "melody", "solo alto saxophone only", None, 65, "woodwinds", 0),
    "sax_tenor": _entry("woodwinds", "melody", "solo tenor saxophone only", None, 66, "woodwinds", 0),
    "sax_bari": _entry("woodwinds", "bass", "solo baritone saxophone only", None, 67, "woodwinds", 5, "low"),
    # ── Brass ───────────────────────────────────────────────────────────
    "trumpet": _entry("brass", "melody", "solo trumpet only", None, 56, "brass", 10),
    "trumpets": _entry("brass", "melody", None, "trumpet section only", 56, "brass", 10),
    "horn": _entry("brass", "counter", "solo french horn only", None, 60, "brass", 25),
    "horns": _entry("brass", "counter", None, "french horn section only", 60, "brass", 25),
    "trombone": _entry("brass", "counter", "solo trombone only", None, 57, "brass", 30, "low-mid"),
    "trombones": _entry("brass", "counter", None, "trombone section only", 57, "brass", 30, "low-mid"),
    "bass_trombone": _entry("brass", "bass", "bass trombone only", None, 57, "brass", 35, "low"),
    "tuba": _entry("brass", "bass", "solo tuba only", None, 58, "brass", 40, "low"),
    # ── Orchestral percussion ───────────────────────────────────────────
    "timpani": _entry("percussion", "rhythm", "timpani only", None, 47, "percussion", 5, "low", 47),
    "snare_orch": _entry("percussion", "rhythm", "orchestral snare drum only", None, 38, "percussion", 0, "mid", 38),
    "bass_drum_orch": _entry("percussion", "rhythm", "concert bass drum only", None, 36, "percussion", 5, "low", 36),
    "cymbals": _entry("percussion", "rhythm", "orchestral cymbals only", None, 49, "percussion", 0, "high", 49),
    "glockenspiel": _entry("percussion", "melody", "glockenspiel only", None, 9, "percussion", -10, "high", None),
    "xylophone": _entry("percussion", "melody", "xylophone only", None, 13, "percussion", -5, "high", None),
    "percussion_orch": _entry("percussion", "rhythm", "orchestral percussion only", None, 47, "percussion", 15, "mid", 70),
    # ── Voices ──────────────────────────────────────────────────────────
    "lead_vocal": _entry("voices", "melody", "dry solo lead vocal only", None, 54, "voices", 0),
    "adlibs": _entry("voices", "counter", "vocal ad-libs and harmonies only", None, 54, "voices", 15),
    "backing_vocals": _entry("voices", "chords", "backing vocal harmonies only", None, 54, "voices", 10),
    "choir_s": _entry("voices", "melody", None, "choir sopranos only", 52, "voices", -20, "high"),
    "choir_a": _entry("voices", "counter", None, "choir altos only", 52, "voices", -8),
    "choir_t": _entry("voices", "counter", None, "choir tenors only", 52, "voices", 8),
    "choir_b": _entry("voices", "bass", None, "choir basses only", 52, "voices", 20, "low"),
    "choir_full": _entry("voices", "chords", None, "full SATB choir only", 52, "voices", 0),
    # ── Keys / synths ───────────────────────────────────────────────────
    "grand_piano": _entry("keys", "chords", "grand piano only", None, 0, "keys", -5),
    "rhodes": _entry("keys", "chords", "rhodes electric piano only", None, 4, "keys", 5),
    "organ": _entry("keys", "chords", "organ only", None, 16, "keys", 0),
    "synth_pad": _entry("keys", "chords", "analog synth pad only", None, 89, "keys", 0),
    "synth_lead": _entry("keys", "melody", "mono synth lead only", None, 80, "keys", 10),
    "synth_pluck": _entry("keys", "chords", "plucked synth only", None, 90, "keys", -10),
    "synthbass": _entry("keys", "bass", "synth bass only", None, 39, "low_end", 0, "low"),
    # ── Guitars / basses ────────────────────────────────────────────────
    "guitar_rhythm": _entry("guitars", "chords", "clean rhythm guitar only", None, 27, "guitars", -20),
    "guitar_lead": _entry("guitars", "melody", "lead guitar only", None, 29, "guitars", 15),
    "guitar_nylon": _entry("guitars", "chords", "nylon string guitar only", None, 24, "guitars", -15),
    "bass_electric": _entry("guitars", "bass", "electric bass guitar only", None, 33, "low_end", 0, "low"),
    "bass_upright": _entry("guitars", "bass", "upright acoustic bass only", None, 32, "low_end", 10, "low"),
}

# Ensemble presets → catalog instrument leaves.
ENSEMBLE_PRESETS = {
    "string_quartet": ["violins_i", "violins_ii", "violas", "celli"],
    "strings": ["violins_i", "violins_ii", "violas", "celli", "contrabasses"],
    "woodwinds": ["flute", "oboe", "clarinet", "bassoon"],
    "brass": ["trumpets", "horns", "trombones", "tuba"],
    "orch_percussion": ["timpani", "snare_orch", "bass_drum_orch", "cymbals"],
    "choir_satb": ["choir_s", "choir_a", "choir_t", "choir_b"],
    "concert_band": ["flute", "clarinet", "sax_alto", "trumpets", "horns",
                     "trombones", "tuba", "percussion_orch"],
    "full_orchestra": ["violins_i", "violins_ii", "violas", "celli",
                       "contrabasses", "harp", "flute", "piccolo", "oboe",
                       "clarinet", "bassoon", "trumpets", "horns",
                       "trombones", "tuba", "timpani", "percussion_orch"],
}

# Drum kits → mic/parts leaves (each a separate style-steered render).
KIT_PRESETS = {
    "loop": ["drum_loop"],
    "standard": ["kick", "snare_top", "hihat", "overheads"],
    "full": ["kick_in", "kick_out", "snare_top", "snare_bottom", "hihat",
             "tom1", "tom2", "tom3", "overhead_l", "overhead_r", "room",
             "percussion"],
}

KIT_PART_STYLES = {
    "drum_loop": "drum kit loop only",
    "kick": "isolated kick drum only", "kick_in": "isolated kick drum, inside mic only",
    "kick_out": "isolated kick drum, outside mic only",
    "snare_top": "isolated snare drum, top mic only",
    "snare_bottom": "isolated snare drum, bottom mic only",
    "hihat": "isolated hi-hat only", "hihat_open": "isolated open hi-hat only",
    "toms": "isolated tom drums only", "tom1": "isolated high tom only",
    "tom2": "isolated mid tom only", "tom3": "isolated floor tom only",
    "overheads": "drum overheads, cymbals only",
    "overhead_l": "drum overhead left, cymbals only",
    "overhead_r": "drum overhead right, cymbals only",
    "room": "drum room mics only, ambient",
    "percussion": "hand percussion only",
    "timpani": "timpani only", "snare_orch": "orchestral snare only",
    "bass_drum_orch": "concert bass drum only", "cymbals": "orchestral cymbals only",
}

# Genre presets → a working band when the style prompt names a genre but no
# instruments. Members are catalog instruments, ensembles, or "drums".
GENRE_PRESETS = {
    "genre:house": ["drums", "synthbass", "synth_pad", "synth_pluck", "rhodes"],
    "genre:edm": ["drums", "synthbass", "synth_pad", "synth_pluck", "synth_lead"],
    "genre:pop": ["drums", "bass_electric", "grand_piano", "synth_pad",
                  "guitar_nylon"],
    "genre:hiphop": ["drums", "synthbass", "rhodes", "synth_pad"],
    "genre:rock": ["drums", "bass_electric", "guitar_rhythm", "guitar_lead",
                   "organ"],
    "genre:funk": ["drums", "bass_electric", "rhodes", "guitar_rhythm",
                   "horns"],
    "genre:jazz": ["drums", "bass_upright", "grand_piano", "sax_tenor",
                   "trumpet"],
    "genre:folk": ["guitar_nylon", "bass_upright", "violins_i", "organ"],
    "genre:ambient": ["synth_pad", "strings", "grand_piano", "harp"],
    "genre:reggae": ["drums", "bass_electric", "guitar_rhythm", "organ"],
    "genre:latin": ["drums", "percussion_orch", "bass_electric",
                    "guitar_nylon", "trumpets"],
    "genre:lofi": ["drums", "rhodes", "bass_upright", "synth_pad"],
    "genre:ballad": ["drums", "grand_piano", "strings", "bass_upright"],
}

# ══════════════════════════════════════════════════════════════════════════════
# Arrangement model
# ══════════════════════════════════════════════════════════════════════════════


def hint_tracks(key: str) -> list[dict]:
    """One score.instrument_hints key → catalog/preset/kit tracks."""
    if key in GENRE_PRESETS:
        tracks = []
        for member in GENRE_PRESETS[key]:
            tracks.extend(hint_tracks(member))
        return tracks
    if key in INSTRUMENT_CATALOG:
        return [make_track(key)]
    if key in ENSEMBLE_PRESETS:
        return [make_track(instrument) for instrument in ENSEMBLE_PRESETS[key]]
    if key == "drums":
        return [make_kit("standard")]
    if key == "percussion":
        return [make_track("percussion_orch")]
    return []


def arrangement_from_picks(picks: list[str], request: dict | None = None) -> dict:
    """Arrangement from explicit instrument/ensemble/kit picks — the session
    sheet's instrument selector bypasses style-derived orchestration."""
    tracks, seen = [], set()
    for key in picks:
        for track in hint_tracks(key):
            if track["name"] in seen:
                continue
            seen.add(track["name"])
            tracks.append(track)
    if not tracks:
        raise ValueError("instrument picks produced no tracks")
    leaves = expand_leaves({"tracks": tracks})
    return {"version": 1, "tracks": tracks, "buses": _default_buses(leaves),
            "fx_sends": list(DEFAULT_FX),
            "derived_from": {"picks": picks,
                             "style": (request or {}).get("style", "")}}


def make_track(instrument: str, name: str | None = None, **overrides) -> dict:
    """A leaf instrument track seeded from the catalog."""
    spec = INSTRUMENT_CATALOG[instrument]
    ensemble = overrides.pop("ensemble", "section" if spec["style_section"] else "solo")
    style = spec["style_section"] if ensemble == "section" and spec["style_section"] else spec["style_solo"]
    track = {"name": name or instrument, "instrument": instrument,
             "role": "vocal" if spec["family"] == "voices" else "stem",
             "ensemble": ensemble, "divisi": overrides.pop("divisi", 1),
             "line": overrides.pop("line", spec["line"]),
             "register": overrides.pop("register", spec["register"]),
             "style_suffix": overrides.pop("style_suffix", style),
             "gm_program": overrides.pop("gm_program", spec["gm"]),
             "drum_note": overrides.pop("drum_note", spec["drum_note"]),
             "pan_azimuth": overrides.pop("pan_azimuth", spec["pan"]),
             "seed_offset": overrides.pop("seed_offset", 0),
             "bus_family": spec["bus"]}
    track.update(overrides)
    return track


def make_kit(kit: str = "standard", name: str = "drums", **overrides) -> dict:
    """A kit track that expands into one leaf per part."""
    parts = list(overrides.pop("parts", KIT_PRESETS[kit]))
    track = {"name": name, "instrument": "drum_kit", "role": "rhythm",
             "kit": kit, "parts": parts, "ensemble": "section", "divisi": 1,
             # Per-part renders are NOT isolated: every part shares identical
             # score conditioning, so each renders the full kit — N copies in
             # the mix. Collapsed by default; the orchestration "Split kit"
             # toggle opts back into one leaf per part.
             "split_kit": False,
             "line": "rhythm", "register": "mid",
             "style_suffix": overrides.pop("style_suffix", "drum kit only"),
             "gm_program": None, "drum_note": None,
             "pan_azimuth": overrides.pop("pan_azimuth", 0),
             "seed_offset": overrides.pop("seed_offset", 0),
             "bus_family": "drums"}
    track.update(overrides)
    return track


def expand_leaves(arrangement: dict) -> list[dict]:
    """Tracks → leaf stems: kit parts and divisi become individual renders."""
    leaves = []
    for track in arrangement.get("tracks", []):
        if not track.get("enabled", True):
            continue  # bypassed track — no leaves, no renders
        if track.get("kit"):
            if not track.get("split_kit"):
                # One kit stem — the model renders the full kit regardless,
                # so N part stems only stack N copies of the same drums.
                leaves.append({**track, "part": None, "track": track["name"],
                               "div_index": 0})
                continue
            for part in track["parts"]:
                leaves.append({**track, "name": f"{track['name']}.{part}",
                               "part": part, "track": track["name"],
                               "style_suffix": KIT_PART_STYLES.get(
                                   part, f"isolated {part} only"),
                               "drum_note": track.get("drum_note")})
        else:
            divisi = max(1, int(track.get("divisi", 1)))
            for i in range(divisi):
                leaf = {**track, "part": None, "track": track["name"],
                        "div_index": i}
                if divisi > 1:
                    leaf["name"] = f"{track['name']}.div_{'abcd'[i]}"
                    leaf["style_suffix"] = f"{track['style_suffix']}, divisi {'abcd'[i]}"
                leaves.append(leaf)
    names = [leaf["name"] for leaf in leaves]
    if len(names) != len(set(names)):
        raise ValueError("Leaf stem names must be unique")
    return leaves


def _default_buses(leaves: list[dict]) -> list[dict]:
    families = {}
    for leaf in leaves:
        families.setdefault(leaf.get("bus_family", "music"), []).append(leaf["name"])
    buses = [{"name": family, "members": sorted(members), "sends": []}
             for family, members in sorted(families.items())]
    return buses


DEFAULT_FX = [{"name": "reverb", "type": "reverb", "rt60": 1.8, "predelay_ms": 20},
              {"name": "delay", "type": "delay", "time_ms": 375, "feedback": 0.3}]


def derive_arrangement(score, request: dict) -> dict:
    """Draft an arrangement from the parsed score + style prompt.

    Always editable afterwards — this is a starting orchestration, not truth.
    """
    from .score import instrument_hints, note_events
    tracks, seen = [], set()

    def add(track):
        if track["name"] not in seen:
            seen.add(track["name"])
            tracks.append(track)

    vocal_notes = note_events(score, "Vocal") if "Vocal" in score.voices else []
    if vocal_notes:
        add(make_track("lead_vocal"))

    hints = instrument_hints(request.get("style", ""))
    for key in hints:
        for track in hint_tracks(key):
            add(track)

    ins_notes = note_events(score, "Ins") if "Ins" in score.voices else []
    if ins_notes and not any(t["line"] == "melody" and t["role"] != "vocal"
                             for t in tracks):
        add(make_track("synth_lead", name="ins_lead", style_suffix="lead instrument only"))

    leaves = expand_leaves({"tracks": tracks})
    return {"version": 1, "tracks": tracks, "buses": _default_buses(leaves),
            "fx_sends": list(DEFAULT_FX),
            "derived_from": {"style": request.get("style", ""), "hints": hints}}


def save_arrangement(path, arrangement: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expand_leaves(arrangement)  # validates names before writing
    write_json(path, arrangement)
    return path


def load_arrangement(path) -> dict:
    import json
    arrangement = json.loads(Path(path).read_text())
    expand_leaves(arrangement)
    return arrangement
