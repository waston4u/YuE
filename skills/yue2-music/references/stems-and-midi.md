# Stems, mixing, MIDI, and the Studio app

The `yue2` package includes a production suite on top of single-song
generation: per-instrument stems, a bus-DAG mixer, MIDI export, immersive
renders, selective ZIP export, and a native macOS app. Package modules:

| Module | Provides |
|---|---|
| `yue2.score` | Native ABC engine: parse, surgery, chord/note events, hints |
| `yue2.arrange` | Instrument catalog, ensemble presets, kits, `arrangement.json` |
| `yue2.stems` | `StemSession` — one `pipe()` render per leaf stem, resume + QC |
| `yue2.midi` | SMF Type-1/0 writer: per-track CCs, ports >16 ch, SysEx |
| `yue2.dsp` | numpy DSP: EQ, compressor, limiter, reverb/delay, LUFS, panning |
| `yue2.mix` | `render_session` → buses, FX returns, master, immersive, report |
| `yue2.deliver` | Named bundles or custom file picks → manifest'd ZIP |
| `yue2.studio.server` | Local HTTP/SSE API for the SwiftUI app |

## CLI quick reference

```bash
yue2 arrange --request req.json --score score.abc --output session/  # draft arrangement
yue2 stems   --request req.json --output session/ [--tracks a,b] [--cleanup role-eq]
yue2 mix     --session session/     # buses + fx + master + immersive + midi
yue2 midi    --session session/ [--midi-type 0|1]
yue2 deliver --session session/ --bundle trackout [--audio-format wav-24] [--sample-rate 44100]
yue2 deliver --session session/ --files stems/x/audio.flac master.flac --out pick.zip
yue2 studio  --serve [--port 8787]  # local API for the app
tools/build_studio_app.sh           # build YuE2 Studio.app (arm64, macOS 14+)
```

## The honesty contract (repeat it when reporting)

- Stems are **separate generation passes** conditioned on score surgery +
  instrument style suffixes — **guided, not guaranteed isolation**. The QC
  report (`stems_report.json`, per-stem `qc.json`) flags duration drift and
  band-energy bleed; do not describe stems as "separated" or "isolated."
- Kit mic-perspective parts (`snare_top`/`snare_bottom`, overheads/room)
  are **separate renders**, not one kit recording mic'd differently.
- Orchestral section stems are **style-steered renders**, not sampled
  sections; the checkpoint's realism is not controllable.
- The score dialect carries Vocal/Ins + chords only — it does not emit N
  score parts. The arrangement supplies the instrument list; each leaf is
  conditioned on its line's score content (melody voice, chord grid,
  chord roots, or meter grid).
- MIDI voicings and drum patterns are **documented interpretations** of
  the symbolic grid, editable by re-arrangement.
- `immersive/binaural.flac` is an approximation; `bed_5_1`/`bed_7_1` are
  multichannel beds — **not Dolby Atmos encodes**.

## When to use what

- User wants stems/trackout → `yue2 stems` then `yue2 mix`, export
  `--bundle trackout`.
- User wants MIDI for a DAW → `yue2 mix` (writes `midi/song.mid`) or
  `yue2 deliver --bundle midi`.
- User wants a final master → `yue2 mix` → `master.flac` (−14 LUFS,
  −1 dBFS ceiling); binaural for headphones; 5.1/7.1 beds for surround.
- Editing the band → PUT `/api/arrangement` or edit `arrangement.json`
  directly before `yue2 stems`; re-render only changed `--tracks`.

See `docs/stems.md`, `docs/midi.md`, `docs/studio.md` for full details.
