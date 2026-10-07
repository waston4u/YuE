# Stems and orchestration

YuE2 Studio renders **one separate generation pass per arrangement leaf** —
never splits a stereo master. The session's `arrangement.json` is the
editable instrument list; N leaf stems = N renders.

## How it works

```text
request (style + lyrics + cot + seed)
  → score.abc (Vocal / Ins / chord symbols)
  → arrangement.json (editable tracks → leaf stems)
  → stems/<leaf>/ — a full pipe() render per leaf
```

Each leaf render is conditioned on a surgically edited score variant chosen
by its `line`:

| `line` | Score conditioning |
|---|---|
| `melody` | its voice's notes (Vocal → lyrics + chords, Ins → melody only) |
| `counter` | chord grid only (style steers a counterline) |
| `chords` | chord grid only |
| `bass` | chord roots + low-register hint |
| `rhythm` | chord grid + meter (drum parts) |

The `style_suffix` steers the instrument ("first violins, ensemble strings
only", "isolated snare drum, bottom mic only"). Seeds are
`base_seed + seed_offset` per leaf.

## Honest constraints

- **Isolation is guided, not guaranteed.** The released checkpoint cannot
  emit isolated multitrack audio. Each stem is a style-steered render —
  bleed is possible and reported in `qc.json`/`stems_report.json`, never
  claimed away.
- **Kit/mic parts are separate performances.** `snare_top` vs
  `snare_bottom` are distinct renders of the same score context, not one
  kit recording captured by different microphones.
- **The score dialect carries two voices + chords.** It does not produce
  30 independent parts; the arrangement supplies the instrument list.
- **Orchestral sections are steered renders**, not sampled sections.
  Realism depends on the checkpoint.

## Arrangement files

```json
{"tracks": [{"name": "violins_i", "instrument": "violins_i",
             "ensemble": "section", "divisi": 1, "line": "melody",
             "style_suffix": "first violins, ensemble strings only",
             "gm_program": 48, "pan_azimuth": -30, "seed_offset": 0},
            {"name": "drums", "instrument": "drum_kit", "kit": "full",
             "parts": ["kick_in", "snare_top", "snare_bottom", "hihat",
                       "tom1", "tom2", "tom3", "overhead_l", "overhead_r",
                       "room", "percussion"], "line": "rhythm"}],
 "buses": [{"name": "strings", "members": ["violins_i"], "sends": []},
           {"name": "drums", "members": ["drums.*"], "sends": []}],
 "fx_sends": [{"name": "reverb", "type": "reverb", "rt60": 1.8}]}
```

- **Kits** (`loop` / `standard` / `full`) expand to one leaf per part —
  `drums.kick_in`, `drums.snare_top`, …
- **Ensemble presets** (`strings`, `woodwinds`, `brass`, `orch_percussion`,
  `choir_satb`, `concert_band`, `full_orchestra`) expand to section leaves.
- **`divisi: N`** splits a section into `name.div_a` / `div_b` renders.
- Every catalog instrument carries GM program, default bus family,
  orchestral seating azimuth, and register hints used by the mixer and
  immersive renders.

## CLI

```bash
yue2 arrange --request examples/song.json --score score.abc --output session/
yue2 stems   --request examples/song.json --output session/ [--tracks a,b] [--cleanup role-eq]
yue2 mix     --session session/
```

Renders are sequential and resumable — verified `stems/<leaf>/result.json`
artifacts are skipped on re-run.
