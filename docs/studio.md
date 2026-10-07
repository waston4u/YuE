# YuE2 Studio (macOS, Apple Silicon)

A native SwiftUI desktop app — **arm64, macOS 14+ only** — that drives the
Python engine over a local HTTP/SSE API. Same simple request in (style +
lyrics + cot + seed), full production suite out.

## Build

```bash
tools/build_studio_app.sh
# → studio/YuE2Studio/build/Build/Products/Release/YuE2 Studio.app
```

Requires Xcode on Apple Silicon. The app is ad-hoc signed for personal use
(no App Store/notarization scope). The engine is your normal `yue2`
install — validate with `yue2 doctor` first.

## Run

```bash
yue2 studio --serve --port 8787      # local API (127.0.0.1 only)
yue2 studio --serve --no-engine      # arrange/mix/export only, no model
open "studio/YuE2Studio/build/Build/Products/Release/YuE2 Studio.app"
```

The app can also launch the engine itself (`EngineService` spawns
`yue2 studio --serve`); point it at your `yue2` binary in settings.

## API surface

`yue2.studio.server` — stdlib `ThreadingHTTPServer`, one job at a time,
sessions resumable via artifact verification.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | liveness |
| `POST /api/session` | create session `{style, lyrics, cot, seed, abc?}` → plan + draft arrangement |
| `GET /api/sessions` / `POST /api/open` | session registry |
| `GET /api/status` | JSON snapshot, or SSE with `Accept: text/event-stream` |
| `GET/PUT /api/arrangement` | orchestration read/write (re-derives mix state on PUT) |
| `GET/PUT /api/mix-state` | `session.json` read/write (faders, pans, sends, master) |
| `POST /api/render` | render stems `{tracks?, cleanup?}` |
| `POST /api/mix` | render deliverables (buses, FX, master, immersive, MIDI) |
| `GET /api/files/<rel>` | download any session artifact |
| `POST /api/export` | `{bundle}` or `{files}` + `{audio_format, sample_rate, midi_type}` → ZIP download |
| `GET /api/formats` | writable audio formats on this machine |
| `POST /api/cancel` | cancel the running job |

## App views

- **Session** — per-stem progress list, render-all / mix / cancel,
  native AVFoundation preview per stem, Reveal in Finder.
- **Orchestration** — the editable track table (instrument, ensemble,
  divisi, line, style suffix); leaf-stem count.
- **Mixer** — channel strips (fader/pan/mute/solo) per leaf stem, backed
  by `session.json`.
- **Deliverables** — bundle picker (Everything / MIDI only / Trackout /
  Stereo / Binaural / Immersive / Buses / FX / Session), per-file checklist
  for custom exports, format pickers (FLAC 24/16, WAV 32f/24/16, MP3 when
  libsndfile supports it, AAC/m4a via `afconvert`, 48k/44.1k, MIDI type
  1/0) → `NSSavePanel` → ZIP with `MANIFEST.json`.

## Immersive honesty

`immersive/binaural.flac` is a lightweight virtual-speaker approximation
(ITD + ILD + head-shadow), **not** a Dolby Atmos render. `bed_5_1.wav` /
`bed_7_1.wav` are multichannel beds for import into a DAW or an external
Atmos renderer — placement comes from each instrument's orchestral seating
azimuth, overridable per track.
