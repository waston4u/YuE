# MIDI export

`yue2 midi --session <dir>` writes `midi/song.mid` — a Standard MIDI File
(PPQ 480) generated from the session's score + arrangement + mix state.
`yue2 mix` writes it automatically too.

## Layout (Type 1, default)

- **Track 0 — conductor:** tempo map (`Q:`), time signatures (`M:`), key
  signatures (`K:`), section markers from `% comments`, and SysEx.
- **Tracks 1..N — one per leaf stem** (kit parts included). Content depends
  on the leaf's `line`:

| Line | Notes emitted |
|---|---|
| `melody` | its voice's score notes (Vocal or Ins) |
| `chords`, `counter`, `texture` | close-voiced chord tones (C4–B4), slash basses honored |
| `bass` | chord roots in E1–B2 at each chord onset |
| `rhythm` / kit parts | GM drum-map hits on channel 9, meter-aware grid |

Drum-map: kick 36 (kick_in 35), snare 38 (top/bottom share the note but
keep separate tracks), hihat 42, toms 41–50, overheads → ride/crash,
timpani 47, cymbals 49, orch snare 38, percussion 70.

## Channels and ports

Melodic stems take channels 0–8 and 10–15; channel 9 is drums. Beyond one
16-channel port, tracks emit **MIDI Port** (`0x21`) and **MIDI Channel
Prefix** (`0x20`) meta events so >16 instruments map deterministically.

## Controls (CC) per track

Bank select CC0/32 + program change (arrangement `gm_program`), CC7 volume,
CC10 pan, CC11 expression, CC64 sustain, CC91 reverb send, CC93 chorus
send — values sourced from `session.json` mix state at export time.

## SysEx

At tick 0: GM System On, GS Reset, XG System On, then a custom YuE2 session
block: `F0 7D "YUE2" <8-char identity> F7`. The identity is the first 8 hex
chars of the session hash. `write_midi(..., raw_sysex=...)` appends a
caller payload.

## Interpretation honesty

Chord voicings and drum patterns are **documented interpretations** of the
symbolic grid — the score carries chord symbols and meter, not voicings or
drum parts. Regenerate with a different arrangement/session state to
revoice.

## Type 0

`yue2 midi --midi-type 0` (or `deliver --midi-type 0`) merges conductor +
all instrument events into a single track for simple players.
