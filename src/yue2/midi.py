"""Standard MIDI File export for YuE2 sessions — stdlib only.

Type 1 (default): conductor track (tempo/meter/key/markers/SysEx) plus one
track per leaf stem. Type 0 merges everything into a single track.

Channel plan: melodic stems use channels 0-8 and 10-15; kit/drum parts use
channel 9. Stems beyond one 16-channel port emit MIDI Port (0x21) and MIDI
Channel Prefix (0x20) meta events so >16 instruments map deterministically.

SysEx at tick 0: GM System On, GS Reset, XG System On, plus a custom YuE2
session block (manufacturer 0x7D) carrying the request identity prefix.
"""
from __future__ import annotations

import struct
from fractions import Fraction

from .score import (Score, chord_events, chord_pitches, key_map, meter_map,
                    note_events, pitch_name)

PPQ = 480
DRUM_CHANNEL = 9
MELODIC_CHANNELS = [c for c in range(16) if c != DRUM_CHANNEL]

# GM drum-map notes per kit part.
DRUM_NOTES = {
    "kick": 36, "kick_in": 35, "kick_out": 36,
    "snare_top": 38, "snare_bottom": 38, "snare_orch": 38,
    "hihat": 42, "hihat_open": 46,
    "tom1": 48, "tom2": 45, "tom3": 41,
    "toms": 45, "percussion": 70, "drum_loop": 36,
    "overheads": 51, "overhead_l": 49, "overhead_r": 51, "room": 57,
    "timpani": 47, "cymbals": 49, "glockenspiel": 9, "xylophone": 13,
}
DEFAULT_DRUM_NOTE = 38


# ══════════════════════════════════════════════════════════════════════════════
# Byte-level SMF helpers
# ══════════════════════════════════════════════════════════════════════════════


def _varlen(value: int) -> bytes:
    if value < 0:
        raise ValueError("Negative delta time")
    out = bytearray([value & 0x7F])
    value >>= 7
    while value:
        out.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(out)


def _meta(kind: int, data: bytes) -> bytes:
    return b"\xff" + bytes([kind]) + _varlen(len(data)) + data


def _text_meta(kind: int, text: str) -> bytes:
    return _meta(kind, text.encode("ascii", errors="replace"))


def _sysex(payload: bytes) -> bytes:
    return b"\xf0" + _varlen(len(payload) + 1) + payload + b"\xf7"


def _ev(delta: int, data: bytes) -> bytes:
    return _varlen(delta) + data


def _cc(channel: int, control: int, value: int) -> bytes:
    return bytes([0xB0 | channel, control & 0x7F, max(0, min(127, int(value)))])


def _pc(channel: int, program: int) -> bytes:
    return bytes([0xC0 | channel, max(0, min(127, int(program)))])


def _on(channel: int, pitch: int, velocity: int) -> bytes:
    return bytes([0x90 | channel, pitch & 0x7F, max(1, min(127, int(velocity)))])


def _off(channel: int, pitch: int) -> bytes:
    return bytes([0x90 | channel, pitch & 0x7F, 0])


def _eot() -> bytes:
    return _meta(0x2F, b"")


def _track(events: list[tuple[int, bytes]]) -> bytes:
    """events: [(absolute_tick, bytes)] → a sorted MTrk chunk."""
    events = sorted(events, key=lambda item: item[0])
    body, last = bytearray(), 0
    for tick, data in events:
        if tick < last:
            raise ValueError("MIDI events out of order")
        body += _ev(tick - last, data)
        last = tick
    body += _ev(0, _eot())
    return b"MTrk" + struct.pack(">I", len(body)) + bytes(body)


def _ticks(quarters: Fraction) -> int:
    return int(quarters * PPQ)


# ══════════════════════════════════════════════════════════════════════════════
# SysEx
# ══════════════════════════════════════════════════════════════════════════════

GM_RESET = bytes([0x7E, 0x7F, 0x09, 0x01])
GS_RESET = bytes([0x41, 0x10, 0x42, 0x12, 0x40, 0x00, 0x7F, 0x00, 0x41])
XG_RESET = bytes([0x43, 0x10, 0x4C, 0x00, 0x00, 0x7E, 0x00])


def session_sysex(identity: str = "") -> bytes:
    """YuE2 session block: manufacturer 0x7D + 'YUE2' + 8-char identity tag."""
    tag = (identity or "yue2sess").encode("ascii", errors="replace")[:8].ljust(8, b"0")
    return bytes([0x7D]) + b"YUE2" + tag


# ══════════════════════════════════════════════════════════════════════════════
# Track builders
# ══════════════════════════════════════════════════════════════════════════════


def conductor_track(score: Score, identity: str = "") -> list[tuple[int, bytes]]:
    events: list[tuple[int, bytes]] = [
        (0, _text_meta(0x03, "conductor")),
        (0, _sysex(GM_RESET)), (0, _sysex(GS_RESET)), (0, _sysex(XG_RESET)),
        (0, _sysex(session_sysex(identity))),
        (0, _meta(0x51, struct.pack(">I", 60_000_000 // score.bpm)[1:])),
    ]
    for onset, (n, d) in meter_map(score):
        dd = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5}.get(d)
        if dd is None:
            raise ValueError(f"Meter denominator {d} is not a power of two")
        events.append((_ticks(onset), _meta(0x58, bytes([n, dd, 24, 8]))))
    for onset, key in key_map(score):
        minor = key.endswith("m")
        root = key[:-1] if minor else key
        sf = (pitch_name(root) - (3 if minor else 0)) % 12
        sf = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4,
              3: -3, 10: -2, 5: -1}.get(sf, 0)
        events.append((_ticks(onset), _meta(0x59, struct.pack("bB", sf, int(minor)))))
    for onset, name in score.sections:
        events.append((_ticks(onset), _text_meta(0x06, name)))
    return events


def _cc_block(channel: int, program: int | None, gain_db: float = 0.0,
              pan: float = 0.0, reverb: int = 40, chorus: int = 0,
              bank: tuple[int, int] | None = None) -> list[tuple[int, bytes]]:
    """Tick-0 control block: bank+program, volume, pan, expression, sends."""
    out = []
    if bank is not None:
        out.append(_cc(channel, 0, bank[0]))
        out.append(_cc(channel, 32, bank[1]))
    if program is not None:
        out.append(_pc(channel, program))
    volume = max(0.0, min(127.0, 127.0 + gain_db * 127.0 / 60.0)) if gain_db <= 0 else 127
    out.append(_cc(channel, 7, volume))
    out.append(_cc(channel, 10, 64 + pan * 63))
    out.append(_cc(channel, 11, 127))
    out.append(_cc(channel, 64, 0))
    out.append(_cc(channel, 91, reverb))
    out.append(_cc(channel, 93, chorus))
    return [(0, data) for data in out]


def _notes(events: list[tuple[int, bytes]], channel: int,
           notes: list[tuple[Fraction, int, Fraction]], velocity: int = 90) -> None:
    for onset, pitch, duration in notes:
        start, end = _ticks(onset), _ticks(onset + duration)
        if end <= start:
            continue
        events.append((start, _on(channel, pitch, velocity)))
        events.append((end, _off(channel, pitch)))


def _chord_notes(score: Score) -> list[tuple[Fraction, int, Fraction]]:
    events = chord_events(score)
    total = score.voices["Vocal"].time
    notes = []
    for index, (onset, symbol, root_name, root_pc, quality, bass_pc) in enumerate(events):
        end = events[index + 1][0] if index + 1 < len(events) else total
        for pitch in chord_pitches(root_pc, quality, bass_pc):
            notes.append((onset, pitch, end - onset))
    return notes


def _bass_notes(score: Score) -> list[tuple[Fraction, int, Fraction]]:
    events = chord_events(score)
    total = score.voices["Vocal"].time
    notes = []
    for index, (onset, symbol, root_name, root_pc, quality, bass_pc) in enumerate(events):
        end = events[index + 1][0] if index + 1 < len(events) else total
        pc = root_pc if bass_pc is None else bass_pc
        pitch = 36 + (pc - 36) % 12  # E1–B2 register window
        notes.append((onset, pitch, end - onset))
    return notes


def _drum_pattern(score: Score, part: str) -> list[tuple[Fraction, int, Fraction]]:
    """Meter-aware grid pattern per kit part — a documented GM interpretation."""
    note = DRUM_NOTES.get(part, DEFAULT_DRUM_NOTE)
    unit = Fraction(1, 4)  # quarter-note grid
    notes = []
    for start, length, (n, d) in score.voices["Vocal"].bars:
        bar_quarters = Fraction(4 * n, d)
        beats = n if d == 4 else max(1, int(bar_quarters))
        strong = [start + Fraction(4 * i, d) for i in range(n)]
        if part in {"kick", "kick_in", "kick_out", "bass_drum_orch", "drum_loop"}:
            hits = [strong[0], strong[len(strong) // 2]] if len(strong) > 1 else strong
        elif part in {"snare_top", "snare_bottom", "snare_orch"}:
            hits = strong[1::2]
        elif part in {"hihat", "hihat_open"}:
            hits = [start + Fraction(2 * i, d) for i in range(2 * n)]
        elif part in {"toms", "tom1", "tom2", "tom3"}:
            hits = [strong[-1]]
        elif part in {"overheads", "overhead_l", "overhead_r", "cymbals"}:
            hits = strong
        else:
            hits = strong
        for hit in hits:
            notes.append((hit, note, unit))
    return notes


def stem_notes(score: Score, leaf: dict) -> list[tuple[Fraction, int, Fraction]]:
    """Score content for one leaf stem, chosen by its ``line``/``part``."""
    line = leaf.get("line", "melody")
    part = leaf.get("part")
    if part or leaf.get("role") == "rhythm":
        return _drum_pattern(score, part or "drum_loop")
    if line in {"melody", "ins"}:
        voice = "Vocal" if leaf.get("role") == "vocal" else "Ins"
        return note_events(score, voice)
    if line == "bass":
        return _bass_notes(score)
    if line in {"chords", "counter", "texture"}:
        return _chord_notes(score)
    return note_events(score, "Ins")


# ══════════════════════════════════════════════════════════════════════════════
# Writer
# ══════════════════════════════════════════════════════════════════════════════


def channel_plan(leaves: list[dict]) -> list[tuple[int, int]]:
    """(port, channel) per leaf; melodic channels skip 9; drums take ch9 on port 0."""
    plan, melodic_index = [], 0
    for leaf in leaves:
        if leaf.get("part") or leaf.get("role") == "rhythm":
            plan.append((0, DRUM_CHANNEL))
        else:
            port = melodic_index // len(MELODIC_CHANNELS)
            channel = MELODIC_CHANNELS[melodic_index % len(MELODIC_CHANNELS)]
            plan.append((port, channel))
            melodic_index += 1
    return plan


def write_midi(score: Score, leaves: list[dict], *, smf_type: int = 1,
               mix_state: dict | None = None, identity: str = "",
               raw_sysex: bytes | None = None) -> bytes:
    """Serialize the score + arrangement into SMF bytes.

    ``leaves``: leaf stem dicts (name, role, part, line, gm_program, drum_note).
    ``mix_state``: optional {name: {gain_db, pan, sends{reverb,chorus}}} → CC values.
    ``raw_sysex``: optional caller payload appended after the session block.
    """
    if smf_type not in (0, 1):
        raise ValueError("smf_type must be 0 or 1")
    names = [leaf.get("name", f"stem{i}") for i, leaf in enumerate(leaves)]
    if len(names) != len(set(names)):
        raise ValueError("Leaf stem names must be unique for MIDI export")
    mix_state = mix_state or {}
    conductor = conductor_track(score, identity)
    if raw_sysex:
        conductor.append((0, _sysex(raw_sysex)))

    tracks: list[list[tuple[int, bytes]]] = []
    for leaf, (port, channel) in zip(leaves, channel_plan(leaves)):
        name = leaf.get("name", "stem")
        state = mix_state.get(name, {})
        sends = state.get("sends", {})
        events = [_text_meta(0x03, name)]
        if port:
            events.append(_meta(0x21, bytes([port])))
        events.append(_meta(0x20, bytes([channel])))
        track_events = [(0, data) for data in events]
        drum = leaf.get("part") or leaf.get("role") == "rhythm"
        track_events += _cc_block(
            channel, None if drum else leaf.get("gm_program", 0),
            gain_db=state.get("gain_db", 0.0), pan=state.get("pan", 0.0),
            reverb=sends.get("reverb", 40), chorus=sends.get("chorus", 0),
            bank=leaf.get("bank"))
        _notes(track_events, channel, stem_notes(score, leaf),
               velocity=100 if drum else 90)
        tracks.append(track_events)

    if smf_type == 0:
        merged = conductor + [event for track in tracks for event in track]
        tracks = [merged]
    else:
        tracks.insert(0, conductor)
    header = b"MThd" + struct.pack(">IHHH", 6, smf_type, len(tracks), PPQ)
    return header + b"".join(_track(events) for events in tracks)


def save_midi(path, score: Score, leaves: list[dict], **kwargs):
    """Write song.mid and return the path."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(write_midi(score, leaves, **kwargs))
    return path
