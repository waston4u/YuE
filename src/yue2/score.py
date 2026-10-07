"""Native two-voice ABC score engine for the YuE2 package.

Same strict dialect as ``skills/yue2-music/scripts/abc_tools.py``: ``Vocal``
and ``Ins`` voices, ``Z`` full-measure rests, ``"chord"`` symbols inside the
Vocal voice, mid-line ``M:``/``K:`` changes, exact ``Fraction`` timing. Adds
stem surgery (silence voices, chord filtering), exact event queries for MIDI
export, and style-prompt instrument hints for orchestration drafting.

The skill keeps its own standalone copy; parity is enforced by tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

VOICES = ("Vocal", "Ins")
DURATIONS = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48}
QUALITIES = ("", "m", "dim", "aug", "7", "maj7", "m7", "dim7", "m7b5",
             "sus4", "sus2", "6", "m6", "7sus4", "m(maj7)")
PITCH_NAME = r"[A-G](?:bb|##|b|#)?"
CHORD = re.compile(PITCH_NAME + "(?:" + "|".join(re.escape(q) for q in QUALITIES)
                   + ")(?:/" + PITCH_NAME + ")?")
TOKEN = re.compile(
    r'"(?P<chord>[^"\n]*)"|\[K:(?P<key>[^\]\n]+)\]|'
    r"(?P<acc>\^\^|__|\^|_|=)?(?P<note>[A-Ga-gz])"
    r"(?P<oct>[,']*)(?P<duration>[0-9]*)(?P<tie>-?)"
)
NATURAL = dict(zip("CDEFGAB", (0, 2, 4, 5, 7, 9, 11)))
KEYS = {
    **dict(zip(("Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"), range(-7, 8))),
    **dict(zip(("Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m"), range(-7, 8))),
}
PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTAL = {"bb": -2, "b": -1, "": 0, "#": 1, "##": 2}
# Semitone sets above the chord root for every supported quality.
QUALITY_INTERVALS = {
    "": (0, 4, 7), "m": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
    "dim7": (0, 3, 6, 9), "m7b5": (0, 3, 6, 10), "sus4": (0, 5, 7),
    "sus2": (0, 2, 7), "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9),
    "7sus4": (0, 5, 7, 10), "m(maj7)": (0, 3, 7, 11),
}


class ScoreError(ValueError):
    """Unsupported notation or a failed structural invariant."""


def fail(condition: bool, message: str) -> None:
    if condition:
        raise ScoreError(message)


def key_accidentals(key: str) -> dict[str, int]:
    fail(key not in KEYS, f"Unsupported key {key!r}; use a standard major or minor K: field")
    count = KEYS[key]
    result = {letter: 0 for letter in NATURAL}
    for letter in ("FCGDAEB" if count > 0 else "BEADGCF")[:abs(count)]:
        result[letter] = 1 if count > 0 else -1
    return result


def meter_value(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", text)
    fail(match is None, f"Unsupported meter {text!r}; write an explicit fraction")
    n, d = map(int, match.groups())
    fail(d > 1024 or d & (d - 1) != 0, f"Unsupported meter denominator {d}")
    return n, d


@dataclass
class Voice:
    meter: tuple[int, int]
    key: str
    time: Fraction = Fraction(0)
    notes: list = field(default_factory=list)
    bars: list = field(default_factory=list)
    chords: list = field(default_factory=list)
    keys: list = field(default_factory=list)
    pending: tuple | None = None


@dataclass
class Score:
    text: str
    unit: Fraction
    bpm: int
    voices: dict[str, Voice]
    music_lines: dict[int, str]
    sections: list = field(default_factory=list)


def parse_bar(body: str, voice: Voice, unit: Fraction, context: str) -> None:
    n, d = voice.meter
    length = Fraction(4 * n, d)
    start = voice.time
    offset = Fraction(0)
    local = {}  # Native exporters propagate accidentals by letter, across octaves.
    if body == "Z":
        fail(voice.pending is not None, f"{context}: tie enters a full-measure rest")
        offset = length
    else:
        cursor = 0
        while cursor < len(body):
            if body[cursor].isspace():
                cursor += 1
                continue
            match = TOKEN.match(body, cursor)
            fail(match is None, f"{context}: unsupported token at {body[cursor:cursor + 24]!r}")
            cursor = match.end()
            chord, key = match.group("chord", "key")
            fail(offset >= length, f"{context}: event after the measure end")
            if chord is not None:
                fail(CHORD.fullmatch(chord) is None, f"{context}: unsupported chord {chord!r}")
                voice.chords.append((start + offset, chord))
                continue
            if key is not None:
                key_accidentals(key)
                voice.key = key
                voice.keys.append((start + offset, key))
                local = {}
                continue
            note, acc, octave, tie = match.group("note", "acc", "oct", "tie")
            units = int(match.group("duration") or "1")
            fail(units not in DURATIONS, f"{context}: unsupported duration {units}; split it into tied supported lengths")
            duration = units * unit * 4
            fail(offset + duration > length, f"{context}: note/rest exceeds meter duration")
            fail("," in octave and "'" in octave, f"{context}: mixed octave marks")
            if note == "z":
                fail(bool(acc or octave or tie), f"{context}: a rest cannot have accidentals, octave marks or ties")
                fail(voice.pending is not None, f"{context}: tie enters a rest")
            else:
                letter = note.upper()
                written = 60 + NATURAL[letter] + (12 if note.islower() else 0)
                written += 12 * (octave.count("'") - octave.count(","))
                alteration = local.get(letter, key_accidentals(voice.key)[letter])
                if acc:
                    alteration = {"=": 0, "_": -1, "__": -2, "^": 1, "^^": 2}[acc]
                    local[letter] = alteration
                pitch = written + alteration
                if voice.pending is not None:
                    old_pitch, old_written = voice.pending
                    # An unmarked continuation retains its tied accidental across
                    # a barline. It does not alter later untied notes in that bar.
                    if not acc and written == old_written:
                        pitch = old_pitch
                    fail(pitch != old_pitch, f"{context}: tie changes pitch from {old_pitch} to {pitch}")
                    voice.notes[-1][2] += duration
                else:
                    fail(not 0 <= pitch <= 127, f"{context}: pitch {pitch} is outside MIDI range")
                    voice.notes.append([start + offset, pitch, duration])
                voice.pending = (pitch, written) if tie else None
            offset += duration
    fail(offset != length, f"{context}: duration {offset} quarter notes != meter duration {length}")
    voice.bars.append((start, length, voice.meter))
    voice.time += length


def parse(text: str) -> Score:
    """Fail closed for unsupported tokens; resolve sounding notes, not token counts."""
    lines = text.splitlines()
    fail(len(lines) < 12, "Incomplete native two-voice ABC")
    fail(lines[0:2] != ["X:1", "T:"], "Expected native X:1 and blank T: header")
    fail(not lines[2].startswith("M:"), "Missing header M:")
    meter = meter_value(lines[2][2:])
    unit_match = re.fullmatch(r"L:1/([1-9][0-9]*)", lines[3])
    fail(unit_match is None, "Expected L:1/<power of two>, usually L:1/32")
    denominator = int(unit_match.group(1))
    fail(denominator > 1024 or denominator & (denominator - 1) != 0, "Unsupported L: denominator")
    unit = Fraction(1, denominator)
    tempo_match = re.fullmatch(r"Q:1/4=([1-9][0-9]*)", lines[4])
    fail(tempo_match is None, "Expected integer quarter-note tempo Q:1/4=<BPM>")
    expected_voices = ['V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
                       'V: Ins clef=treble name="Ins Melody" snm="Inst."']
    fail(lines[5:7] != expected_voices, "Preserve native Vocal and Ins voice definitions")
    fail(not lines[7].startswith("K:"), "Missing header K:")
    key = lines[7][2:]
    key_accidentals(key)
    voices = {name: Voice(meter, key, keys=[(Fraction(0), key)]) for name in VOICES}
    music_lines = {}
    sections = []
    cursor = 8
    group = 0
    while cursor < len(lines):
        names = []
        while cursor < len(lines) and lines[cursor].startswith("% "):
            names.append(lines[cursor][2:].strip())
            cursor += 1
        fail(cursor == len(lines), "Dangling section comment without music")
        group += 1
        if names:
            sections.append((voices["Vocal"].time, " ".join(names)))
        counts = []
        for name in VOICES:
            context = f"group {group}, {name}"
            fail(cursor >= len(lines) or lines[cursor] != f"V: {name}", f"{context}: expected V: {name}")
            cursor += 1
            voice = voices[name]
            fields = set()
            while cursor < len(lines) and lines[cursor].startswith(("M:", "K:")):
                field_name, value = lines[cursor].split(":", 1)
                fail(field_name in fields, f"{context}: duplicate {field_name}: field")
                fields.add(field_name)
                if field_name == "M":
                    voice.meter = meter_value(value)
                else:
                    key_accidentals(value)
                    voice.key = value
                    voice.keys.append((voice.time, value))
                cursor += 1
            fail(cursor >= len(lines), f"{context}: missing music line")
            line = lines[cursor]
            fail(not line.endswith("|"), f"{context}: music line must end with a plain barline")
            music_lines[cursor] = name
            cursor += 1
            bars = []
            for bar in line[:-1].split("|"):
                bar = bar.strip()
                fail(not bar, f"{context}: empty measure or unsupported double/repeat barline")
                rest = re.fullmatch(r"Z([2-4])?", bar)
                if rest:
                    bars.extend(["Z"] * int(rest.group(1) or "1"))
                else:
                    bars.append(bar)
            fail(not 1 <= len(bars) <= 4, f"{context}: expected 1–4 measures after expanding Z rests")
            counts.append(len(bars))
            for bar in bars:
                parse_bar(bar, voice, unit, f"{context}, bar {len(voice.bars) + 1}")
        fail(counts[0] != counts[1], f"group {group}: voices have different measure counts")
    for name, voice in voices.items():
        fail(voice.pending is not None, f"{name}: unresolved tie at end of score")
    fail(voices["Ins"].chords != [], "Native chord symbols belong in Vocal, not Ins")
    fail(voices["Vocal"].bars != voices["Ins"].bars, "Voice meter/time grids differ")
    fail(voices["Vocal"].keys != voices["Ins"].keys, "Voice key-change timelines differ")
    return Score(text, unit, int(tempo_match.group(1)), voices, music_lines, sections)


def parse_abc(text: str) -> Score:
    """Public entry point; no model load, files, or optional dependencies."""
    return parse(text)


# ══════════════════════════════════════════════════════════════════════════════
# Event queries
# ══════════════════════════════════════════════════════════════════════════════


def note_events(score: Score, voice: str = "Vocal") -> list[tuple[Fraction, int, Fraction]]:
    """(onset_quarters, midi_pitch, duration_quarters) sounding notes of a voice."""
    fail(voice not in score.voices, f"Unknown voice {voice!r}")
    return [(t, p, d) for t, p, d in score.voices[voice].notes]


def pitch_name(value: str) -> int:
    """Parse a chord pitch name like ``Bb`` or ``F#`` into a semitone class."""
    match = re.fullmatch(r"([A-G])(bb|##|b|#)?", value)
    fail(match is None, f"Unsupported pitch name {value!r}")
    return (PITCH_CLASS[match.group(1)] + ACCIDENTAL[match.group(2) or ""]) % 12


def chord_events(score: Score) -> list[tuple[Fraction, str, str, int, str, int | None]]:
    """(onset_quarters, chord_text, root_name, root_pc, quality, bass_pc) from Vocal symbols."""
    events = []
    for onset, symbol in score.voices["Vocal"].chords:
        match = CHORD.fullmatch(symbol)
        fail(match is None, f"Unsupported chord {symbol!r}")
        body, _, bass_name = symbol.partition("/")
        # Split root name (1-3 chars) from quality suffix deterministically.
        root_name = re.match(PITCH_NAME, body).group(0)
        quality = body[len(root_name):]
        fail(quality not in QUALITY_INTERVALS, f"Unsupported chord quality in {symbol!r}")
        bass_pc = pitch_name(bass_name) if bass_name else None
        events.append((onset, symbol, root_name, pitch_name(root_name), quality, bass_pc))
    return events


def chord_pitches(root_pc: int, quality: str, bass_pc: int | None = None,
                  low: int = 48, high: int = 83) -> list[int]:
    """Close-voiced chord tones inside [low, high]; slash bass placed at the bottom."""
    pcs = sorted({(root_pc + i) % 12 for i in QUALITY_INTERVALS[quality]})
    tones = sorted(next(p for p in range(pc, high + 1, 12) if p >= low) for pc in pcs)
    if bass_pc is not None and bass_pc != root_pc:
        # Slash bass one octave below the voicing window.
        bass = low - 12 + (bass_pc - (low - 12)) % 12
        tones = [bass] + [p for p in tones if p != bass]
    return tones


def meter_map(score: Score) -> list[tuple[Fraction, tuple[int, int]]]:
    """(onset_quarters, (numerator, denominator)) meter changes, first at 0."""
    return [(bar[0], bar[2]) for i, bar in enumerate(score.voices["Vocal"].bars)
            if i == 0 or bar[2] != score.voices["Vocal"].bars[i - 1][2]]


def key_map(score: Score) -> list[tuple[Fraction, str]]:
    return list(score.voices["Vocal"].keys)


# ══════════════════════════════════════════════════════════════════════════════
# Surgery — text transforms validated by re-parsing
# ══════════════════════════════════════════════════════════════════════════════


def _transform_music(text: str, transform) -> str:
    """Apply a TOKEN.sub transform to each voice music line, then re-validate."""
    score = parse(text)
    lines = text.splitlines(keepends=True)
    for index, name in score.music_lines.items():
        lines[index] = TOKEN.sub(lambda m, n=name: transform(m, n), lines[index])
    output = "".join(lines)
    parse(output)
    return output


def silence_voice(text: str, name: str) -> str:
    """Replace a voice's notes with equal rests; chord symbols stay in place."""
    fail(name not in VOICES, f"Unknown voice {name!r}")

    def transform(match, voice):
        if voice == name and match.group("note") is not None:
            return "z" + match.group("duration")
        return match.group(0)

    output = _transform_music(text, transform)
    result = parse(output)
    fail(any(result.voices[name].notes), f"silence_voice left notes in {name}")
    return output


def drop_chords(text: str) -> str:
    """Remove every chord symbol; melodies and meter unchanged."""
    def transform(match, voice):
        return "" if match.group("chord") is not None else match.group(0)

    output = _transform_music(text, transform)
    fail(any(v.chords for v in parse(output).voices.values()), "drop_chords left a chord symbol")
    return output


def keep_chords_only(text: str) -> str:
    """Silence both melodies; the chord-symbol grid remains for harmony/rhythm stems."""
    output = silence_voice(silence_voice(text, "Vocal"), "Ins")
    result = parse(output)
    fail(any(v.notes for v in result.voices.values()), "keep_chords_only left melody notes")
    return output


def variant_for_line(text: str, line: str) -> str:
    """Score variant for an orchestration ``line``: melody|counter|chords|bass|rhythm."""
    if line in {"melody", "ins"}:
        return text
    if line in {"counter", "chords", "bass", "rhythm"}:
        return keep_chords_only(text)
    raise ValueError(f"Unknown line {line!r}")


# ══════════════════════════════════════════════════════════════════════════════
# Part synthesis — a distinct Ins voice per orchestration line
# ══════════════════════════════════════════════════════════════════════════════
# The dialect carries one Ins melody plus chord symbols, so "which instrument
# plays what" needs real content differences, not style text alone: bass gets
# root pulses, chords get voiced arpeggios, counter gets sustained inner tones.


def _abc_pitch(pitch: int, key: str) -> str:
    """MIDI pitch → ABC note token spelled in ``key`` (accidentals as needed)."""
    accs = key_accidentals(key)
    best = None
    for letter, natural in NATURAL.items():
        for octave in range(-3, 4):
            diff = pitch - (60 + natural + 12 * octave)
            if not -2 <= diff <= 2:
                continue
            rank = (diff != accs[letter], abs(diff), abs(octave))
            if best is None or rank < best[0]:
                best = (rank, letter, octave, diff)
    fail(best is None, f"Cannot spell MIDI pitch {pitch} in key {key!r}")
    _, letter, octave, diff = best
    token = (letter.lower() + "'" * (octave - 1)) if octave > 0 \
        else letter + "," * -octave
    if diff != accs[letter]:
        token = {-2: "__", -1: "_", 0: "=", 1: "^", 2: "^^"}[diff] + token
    return token


def _dur_units(duration: Fraction, unit: Fraction) -> list[int]:
    """Split a duration into supported unit counts; exact or fail."""
    q = duration / (unit * 4)
    fail(q.denominator != 1, f"Duration {duration} is off the L: unit grid")
    remaining = int(q)
    fail(remaining <= 0, "Cannot emit a zero-length event")
    chunks = []
    for size in sorted(DURATIONS, reverse=True):
        while remaining >= size:
            chunks.append(size)
            remaining -= size
    return chunks


def _bar_body(start: Fraction, length: Fraction, events: list,
              unit: Fraction, key: str) -> str:
    """ABC for one bar: events clipped to the bar, rests fill the gaps."""
    end = start + length
    tokens, cursor = [], start
    for onset, pitch, duration in sorted(events):
        onset, stop = max(onset, start), min(onset + duration, end)
        if onset >= stop or onset < cursor:
            continue
        if onset > cursor:
            tokens += [f"z{n}" for n in _dur_units(onset - cursor, unit)]
        spelled = _abc_pitch(pitch, key)
        tokens.append("-".join(f"{spelled}{n}"
                               for n in _dur_units(stop - onset, unit)))
        cursor = stop
    if cursor < end:
        tokens += [f"z{n}" for n in _dur_units(end - cursor, unit)]
    return " ".join(tokens) if tokens else "Z"


def render_part(score: Score, voice: str, events: list) -> list[str]:
    """Bar bodies for ``events`` on a voice's meter grid, in parse order."""
    keys = score.voices[voice].keys
    bodies = []
    for start, length, _meter in score.voices[voice].bars:
        key = keys[0][1]
        for at, candidate in keys:
            if at <= start:
                key = candidate
        bodies.append(_bar_body(start, length, events, score.unit, key))
    return bodies


def rewrite_voice(text: str, voice: str, events: list) -> str:
    """Replace a voice's music with ``events`` on the same meter/key grid."""
    fail(voice not in VOICES, f"Unknown voice {voice!r}")
    score = parse(text)
    bars = render_part(score, voice, events)
    lines = text.splitlines(keepends=True)
    cursor = 0
    for index, name in sorted(score.music_lines.items()):
        if name != voice:
            continue
        raw = lines[index].rstrip("\n")
        count = 0
        for chunk in raw[:-1].split("|"):
            rest = re.fullmatch(r"Z([2-4])?", chunk.strip())
            count += int(rest.group(1) or "1") if rest else 1
        lines[index] = "|".join(bars[cursor:cursor + count]) + "|\n"
        cursor += count
    fail(cursor != len(bars), "rewrite_voice lost sync with the bar grid")
    output = "".join(lines)
    parse(output)  # fail closed: synthesized bars must satisfy every invariant
    return output


def _pc_in_range(pc: int, low: int, high: int) -> int:
    """Pitch class folded into [low, high] (window must be ≥ an octave)."""
    pitch = low + (pc - low) % 12
    return pitch if pitch <= high else pitch - 12


def part_events(score: Score, line: str) -> list:
    """Distinct Ins part per orchestration line, derived from the chord grid.

    ``bass`` → quarter-note root pulses (slash bass honored) in the bass
    octave; ``chords``/``texture`` → a cycling chord-tone arpeggio in the
    mid register; ``counter`` → sustained inner-voice tones alternating
    through the chord's upper notes. Empty when the score has no chords —
    the caller decides the fallback.
    """
    chords = chord_events(score)
    if not chords:
        return []
    end = score.voices["Vocal"].time
    beat = Fraction(1)
    events = []
    for index, (onset, _sym, _name, root_pc, quality, bass_pc) in enumerate(chords):
        stop = chords[index + 1][0] if index + 1 < len(chords) else end
        if stop <= onset:
            continue
        if line == "bass":
            pitch = _pc_in_range(bass_pc if bass_pc is not None else root_pc, 36, 47)
            at = onset
            while at < stop:
                events.append((at, pitch, min(beat, stop - at)))
                at += beat
        elif line in {"chords", "texture"}:
            tones = chord_pitches(root_pc, quality, bass_pc, low=48, high=71)
            at, k = onset, 0
            while at < stop:
                events.append((at, tones[k % len(tones)], min(beat, stop - at)))
                at += beat
                k += 1
        elif line == "counter":
            tones = chord_pitches(root_pc, quality, bass_pc, low=60, high=79)
            inner = tones[1:] or tones
            events.append((onset, inner[index % len(inner)], stop - onset))
        else:
            return []
    return events


# ══════════════════════════════════════════════════════════════════════════════
# Style hints for orchestration drafting
# ══════════════════════════════════════════════════════════════════════════════

HINTS = {
    "piano": "grand_piano", "rhodes": "rhodes", "organ": "organ",
    "guitar": "guitar_rhythm", "acoustic guitar": "guitar_nylon",
    "nylon": "guitar_nylon", "bass": "bass_electric", "synth bass": "synthbass",
    "synthbass": "synthbass", "808": "synthbass", "pad": "synth_pad",
    "synth": "synth_pad", "strings": "strings", "violin": "violins_i",
    "cello": "celli", "harp": "harp", "flute": "flute", "sax": "sax_tenor",
    "trumpet": "trumpet", "horn": "horns", "trombone": "trombones",
    "brass": "brass", "choir": "choir_full", "orchestra": "full_orchestra",
    "orchestral": "full_orchestra", "drums": "drums", "percussion": "percussion",
    # Genres → full-band presets (resolved in arrange.GENRE_PRESETS).
    "deep house": "genre:house", "tech house": "genre:house",
    "house": "genre:house", "edm": "genre:edm", "dance": "genre:edm",
    "techno": "genre:edm", "trance": "genre:edm", "electro": "genre:edm",
    "synthwave": "genre:edm", "dubstep": "genre:edm",
    "drum and bass": "genre:edm", "dnb": "genre:edm", "garage": "genre:edm",
    "disco": "genre:funk", "funk": "genre:funk", "soul": "genre:funk",
    "motown": "genre:funk", "gospel": "genre:funk",
    "hip hop": "genre:hiphop", "hip-hop": "genre:hiphop",
    "trap": "genre:hiphop", "rap": "genre:hiphop", "r&b": "genre:hiphop",
    "rnb": "genre:hiphop", "reggaeton": "genre:hiphop",
    "hard rock": "genre:rock", "rock": "genre:rock", "indie": "genre:rock",
    "metal": "genre:rock", "punk": "genre:rock", "grunge": "genre:rock",
    "shoegaze": "genre:rock",
    "jazz": "genre:jazz", "blues": "genre:jazz", "swing": "genre:jazz",
    "bossa": "genre:jazz", "bossa nova": "genre:jazz",
    "folk": "genre:folk", "country": "genre:folk",
    "americana": "genre:folk", "bluegrass": "genre:folk",
    "acoustic": "genre:folk",
    "ambient": "genre:ambient", "cinematic": "genre:ambient",
    "soundtrack": "genre:ambient", "film score": "genre:ambient",
    "reggae": "genre:reggae", "ska": "genre:reggae",
    "latin": "genre:latin", "salsa": "genre:latin", "samba": "genre:latin",
    "bossanova": "genre:latin",
    "lo-fi": "genre:lofi", "lofi": "genre:lofi", "chillhop": "genre:lofi",
    "ballad": "genre:ballad", "pop": "genre:pop", "k-pop": "genre:pop",
    "synth-pop": "genre:pop", "synthpop": "genre:pop", "city pop": "genre:pop",
    "new wave": "genre:pop", "madonna": "genre:pop",
}


def instrument_hints(style: str) -> list[str]:
    """Map a style prompt to catalog keys/presets; longest keyword match wins."""
    found, order = [], sorted(HINTS, key=len, reverse=True)
    lowered = style.lower()
    covered = [False] * len(lowered)
    for hint in order:
        for match in re.finditer(re.escape(hint), lowered):
            span = range(match.start(), match.end())
            if not any(covered[i] for i in span):
                key = HINTS[hint]
                if key not in found:
                    found.append(key)
                for i in span:
                    covered[i] = True
    return found
