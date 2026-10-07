"""Small CLI sharing the Python pipeline's defaults and artifact protocol."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from .storage import write_json, verify_result, identity


def kit_root():
    return Path(os.environ.get("YUE2_KIT", Path(__file__).resolve().parents[2]))


def model_paths(args):
    root = kit_root()
    local_model = root / "models/YuE2-3B"
    local_vae = root / "models" / ("YuE2-Vae-legacy" if args.vae == "legacy" else "YuE2-Vae")
    model = args.model or (str(local_model) if local_model.exists() else "m-a-p/YuE2-3B")
    vae = args.vae if args.vae not in {"standard", "legacy"} else (
        str(local_vae) if local_vae.exists() else "m-a-p/" + local_vae.name)
    return model, vae


def get_pipe(args):
    from .pipeline import YuE2Pipeline
    from .protocol import GenerationConfig
    model, vae = model_paths(args)
    config = GenerationConfig.from_dict(json.loads(Path(args.config).read_text())) if args.config else None
    return YuE2Pipeline.from_pretrained(model, vae=vae, revision=args.revision,
             vae_revision=args.vae_revision, device=args.device, memory_budget_gib=args.budget,
             quantization=args.quantization, offload_ar=args.offload_ar,
             local_files_only=args.offline, generation_config=config,
             engine=getattr(args, "engine", "auto"),
             vae_core_frames=512 if args.budget <= 12 else 1024,
             progress=not getattr(args, "quiet", False))


def request_kwargs(data, base=Path.cwd()):
    allowed = {"style", "tags", "lyrics", "cot", "seed", "abc", "cfg_scale", "id", "abc_sampling", "semantic_sampling"}
    metadata = {"lang", "eval_index", "clip_id", "prompt"}
    unknown = set(data) - allowed - metadata - {"abc_path"}
    if unknown:
        raise ValueError(f"Unknown request fields: {sorted(unknown)}")
    result = {k: v for k, v in data.items() if k in allowed}
    if "abc_path" in data:
        if data.get("abc") is not None:
            raise ValueError("Pass abc or abc_path, not both")
        result["abc"] = (base / data["abc_path"]).read_bytes().decode("utf-8")
    if data.get("prompt") is not None:
        from .protocol import SongRequest
        request = SongRequest(style=result.get("style", result.get("tags")), lyrics=result["lyrics"],
                              cot=result.get("cot", "full"))
        if data["prompt"] != request.text():
            raise ValueError("Historical literal prompt does not match native instruction/style/lyrics")
    return result


def doctor(args):
    import torch
    from .storage import model_identity
    model, vae = model_paths(args)
    versions = {}
    for package in ("torch", "transformers", "huggingface-hub", "safetensors", "tiktoken", "soundfile"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    from .quantization import int8_supported
    mps = torch.backends.mps.is_available()
    report = {"dependencies_ready": all(versions.values()), "versions": versions,
              "platform": "apple-silicon", "mps_available": mps,
              "int8_mps": int8_supported("mps") if mps else False,
              "model": model, "vae": vae, "default_cot": "full", "default_cfg": {"full": 1., "melody": 1., "off": 1.01},
              "validated": False, "note": "Environment readiness is not quality or real-24GB acceptance."}
    if args.verify_hashes:
        from .storage import resolve_model
        report["weights"] = {"model": model_identity(resolve_model(model, local_files_only=args.offline)),
                             "vae": model_identity(resolve_model(vae, local_files_only=args.offline))}
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["dependencies_ready"] else 1


def generate(args):
    if args.request:
        path = Path(args.request)
        data = json.loads(path.read_text())
        base = path.parent
    else:
        data, base = {}, Path.cwd()
    for key in ("id", "style", "cot", "seed", "cfg_scale"):
        value = getattr(args, key)
        if value is not None:
            data[key] = value
    if args.lyrics_file:
        data["lyrics"] = Path(args.lyrics_file).read_bytes().decode("utf-8")
    elif args.lyrics is not None:
        data["lyrics"] = args.lyrics
    if args.abc_file:
        data["abc"] = Path(args.abc_file).read_bytes().decode("utf-8")
        data.pop("abc_path", None)
    if not data:
        data = {"id": "first_song", "style": "Mandarin, warm piano, acoustic pop, female vocal",
                "lyrics": "[Verse]\n晚风轻轻吹过窗前\n你留下的笑还在昨天\n[Chorus]\n让这首歌陪你走远\n把所有想念唱成明天"}
    kwargs = request_kwargs(data, base)
    pipe = get_pipe(args)
    directory = Path(args.output or "runs/default") / kwargs.get("id", "song")
    if args.resume and (directory / "result.json").exists():
        req_kwargs = {k: v for k, v in kwargs.items() if k not in {"abc_sampling", "semantic_sampling"}}
        request = pipe._request(**req_kwargs)
        config = pipe.effective_config(request, kwargs.get("abc_sampling"), kwargs.get("semantic_sampling"))
        expected = identity({"request": request.to_dict(), "config": config, "weights": pipe.weights})
        result = verify_result(directory, expected)
        print(json.dumps({"resumed": True, "result": str(directory / "result.json"), "truncated": result["truncated"]}))
        return 0
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Nonempty output {directory}; use --resume or a new output directory")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == "plan":
            kwargs.pop("semantic_sampling", None)
            plan = pipe.plan(**kwargs)
            plan.save(directory)
            print(json.dumps({"stage": "plan", "output": str(directory), "truncated": plan.truncated}))
        else:
            result = pipe(**kwargs)
            result.save_artifacts(directory)
            print(json.dumps({"status": "complete", "output": str(directory), "truncated": result.truncated,
                              "seconds": result.timing["e2e_seconds"]}))
        return 0
    except BaseException as exc:
        write_json(directory / "failure.json", {"status": "failed", "type": type(exc).__name__, "reason": str(exc),
                                                "request": data})
        raise
    finally:
        pipe.close()


def batch(args):
    if args.concurrency != 1:
        raise ValueError("The minimal torch pipeline currently supports concurrency=1; do not share a pipeline concurrently")
    path = Path(args.input)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r.get("id") for r in rows]
    if any(x is None for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("Every batch request must have a unique id")
    pipe = get_pipe(args)
    output = Path(args.output or "runs/batch")
    receipts, failures = [], 0
    try:
        for index, row in enumerate(rows, 1):
            if not args.quiet:
                print(f"Song {index}/{len(rows)}: {row['id']}", file=sys.stderr, flush=True)
            kwargs = request_kwargs(row, path.parent)
            if args.cot is not None:
                kwargs["cot"] = args.cot
            directory = output / row["id"]
            try:
                request = pipe._request(**{k:v for k,v in kwargs.items() if k not in {"abc_sampling", "semantic_sampling"}})
                cfg = pipe.effective_config(request, kwargs.get("abc_sampling"), kwargs.get("semantic_sampling"))
                expected = identity({"request": request.to_dict(), "config": cfg, "weights": pipe.weights})
                if args.resume and (directory / "result.json").exists():
                    receipt = verify_result(directory, expected)
                else:
                    if directory.exists() and any(directory.iterdir()):
                        raise FileExistsError("Nonempty request output; use --resume or a new run")
                    receipt = pipe(**kwargs).save_artifacts(directory)
                receipts.append({"id": row["id"], "status": "complete", "identity": receipt["identity"]})
            except Exception as exc:
                failures += 1
                failure = {"id": row["id"], "status": "failed", "reason": str(exc), "type": type(exc).__name__}
                write_json(directory / "failure.json", failure)
                receipts.append(failure)
            write_json(output / "batch.json", {"complete": len(receipts) == len(rows) and not failures,
                       "expected": len(rows), "failed": failures, "results": receipts})
    finally:
        pipe.close()
    return int(failures > 0)


def _session_request(args, session_dir):
    """Request dict for a session: --request file, saved request.json, or inline args."""
    if getattr(args, "request", None):
        data = json.loads(Path(args.request).read_text())
        write_json(session_dir / "request.json", data)
        return data
    saved = session_dir / "request.json"
    if saved.is_file():
        return json.loads(saved.read_text())
    raise ValueError("Pass --request or run inside a session directory")


def _session_score(args, session_dir, pipe=None, request=None):
    """Score ABC: --score file > saved score.abc > engine plan."""
    if getattr(args, "score", None):
        text = Path(args.score).read_bytes().decode("utf-8")
    elif (session_dir / "score.abc").is_file():
        text = (session_dir / "score.abc").read_text()
    elif pipe is not None and request is not None:
        from .protocol import SongRequest
        plan = pipe.plan(request=SongRequest(**request))
        text = plan.abc or ""
    else:
        raise ValueError("No score yet — pass --score or render the plan first")
    (session_dir / "score.abc").write_text(text)
    return text


def arrange_cmd(args):
    from .arrange import derive_arrangement, load_arrangement, save_arrangement
    from .score import parse
    session_dir = Path(args.output)
    session_dir.mkdir(parents=True, exist_ok=True)
    if args.arrangement:
        arrangement = load_arrangement(args.arrangement)
    else:
        request = _session_request(args, session_dir)
        pipe = None if args.score or (session_dir / "score.abc").exists() else get_pipe(args)
        score_text = _session_score(args, session_dir, pipe, request)
        arrangement = derive_arrangement(parse(score_text), request)
    save_arrangement(session_dir / "arrangement.json", arrangement)
    from .arrange import expand_leaves
    print(json.dumps({"tracks": len(arrangement["tracks"]),
                      "leaf_stems": len(expand_leaves(arrangement)),
                      "output": str(session_dir / "arrangement.json")}))
    return 0


def stems_cmd(args):
    from .arrange import derive_arrangement, load_arrangement, save_arrangement
    from .score import parse
    from .stems import StemSession
    session_dir = Path(args.output)
    session_dir.mkdir(parents=True, exist_ok=True)
    request = _session_request(args, session_dir)
    pipe = get_pipe(args)
    try:
        score_text = _session_score(args, session_dir, pipe, request)
        if args.arrangement:
            arrangement = load_arrangement(args.arrangement)
        elif (session_dir / "arrangement.json").is_file():
            arrangement = load_arrangement(session_dir / "arrangement.json")
        else:
            arrangement = derive_arrangement(parse(score_text), request)
        save_arrangement(session_dir / "arrangement.json", arrangement)
        tracks = args.tracks.split(",") if args.tracks else None
        report = StemSession(session_dir).render(
            pipe, request, arrangement, score_text, tracks=tracks,
            cleanup=args.cleanup)
        print(json.dumps({"stems": len(report["stems"]),
                          "output": str(session_dir / "stems_report.json")}))
        return 0
    finally:
        pipe.close()


def midi_cmd(args):
    from .arrange import expand_leaves, load_arrangement
    from .midi import save_midi
    from .score import parse
    session_dir = Path(args.session)
    score_text = (session_dir / "score.abc").read_text()
    arrangement = load_arrangement(session_dir / "arrangement.json")
    path = save_midi(session_dir / "midi" / "song.mid", parse(score_text),
                     expand_leaves(arrangement), smf_type=args.midi_type)
    print(json.dumps({"midi": str(path), "type": args.midi_type}))
    return 0


def mix_cmd(args):
    from .mix import render_session
    report = render_session(Path(args.session))
    print(json.dumps({"status": report["status"],
                      "loudness": report["loudness"]["integrated_lufs"],
                      "output": str(Path(args.session) / "report.json")}))
    return 0


def deliver_cmd(args):
    from .deliver import deliver
    path = deliver(Path(args.session), bundle=args.bundle,
                   files=args.files, audio_format=args.audio_format,
                   sample_rate=args.sample_rate, midi_type=args.midi_type,
                   out=args.out)
    print(json.dumps({"zip": str(path)}))
    return 0


def studio_cmd(args):
    if args.open:
        app = Path(__file__).resolve().parents[2] / "studio" / "YuE2Studio"
        import subprocess
        subprocess.run(["open", str(app)], check=False)
    if not args.serve:
        print("Pass --serve to run the local API, or --open to open the app.")
        return 0
    from .studio.server import serve
    engine_factory = None if args.no_engine else lambda: get_pipe(args)
    serve(port=args.port, sessions_root=args.sessions_dir,
          engine_factory=engine_factory, min_free_gib=args.min_free)
    return 0


def parser():
    p = argparse.ArgumentParser(description="YuE2: style + lyrics → symbolic plan → song")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("doctor", "generate", "batch"):
        q = sub.add_parser(name)
        q.add_argument("--model")
        q.add_argument("--vae", default="standard", help="standard (listening), legacy (paper evaluation), local path or HF repo")
        q.add_argument("--revision")
        q.add_argument("--vae-revision")
        q.add_argument("--device", default="auto")
        q.add_argument("--engine", choices=("auto", "mlx", "torch"), default="auto",
                       help="auto: MLX on Apple Silicon when available")
        q.add_argument("--budget", type=float, default=24)
        q.add_argument("--quantization", choices=("none", "int4", "int8", "auto"),
                       default="auto",
                       help="auto: best fidelity that fits — MLX picks "
                            "bf16/int8/int4 by free memory; torch picks "
                            "int8 when GPU memory is tight")
        q.add_argument("--offload-ar", action="store_true")
        q.add_argument("--offline", action="store_true")
        q.add_argument("--models-dir",
                       help="Model-weight cache root, e.g. an external SSD "
                            "(sets HF_HOME for all hub downloads)")
        q.add_argument("--config")
        q.add_argument("--output")
        if name == "doctor":
            q.add_argument("--verify-hashes", action="store_true")
        else:
            q.add_argument("--cot", choices=("full", "melody", "off"))
            q.add_argument("--resume", action="store_true")
            q.add_argument("--quiet", "--no-progress", action="store_true",
                           help="Hide YuE2 progress on stderr; keep result output on stdout")
            if name == "batch":
                q.add_argument("--input", required=True)
                q.add_argument("--concurrency", type=int, default=1)
            else:
                q.add_argument("--request")
                q.add_argument("--id")
                q.add_argument("--style")
                q.add_argument("--lyrics")
                q.add_argument("--lyrics-file")
                q.add_argument("--abc-file")
                q.add_argument("--seed", type=int)
                q.add_argument("--cfg-scale", type=float)
                q.add_argument("--stage", choices=("plan", "audio"), default="audio")
    # ── Studio suite ────────────────────────────────────────────────────
    def engine_args(q):
        q.add_argument("--model")
        q.add_argument("--vae", default="standard")
        q.add_argument("--revision")
        q.add_argument("--vae-revision")
        q.add_argument("--device", default="auto")
        q.add_argument("--engine", choices=("auto", "mlx", "torch"), default="auto",
                       help="auto: MLX on Apple Silicon when available")
        q.add_argument("--budget", type=float, default=24)
        q.add_argument("--quantization", choices=("none", "int4", "int8", "auto"),
                       default="auto",
                       help="auto: best fidelity that fits — MLX picks "
                            "bf16/int8/int4 by free memory; torch picks "
                            "int8 when GPU memory is tight")
        q.add_argument("--offload-ar", action="store_true")
        q.add_argument("--offline", action="store_true")
        q.add_argument("--models-dir",
                       help="Model-weight cache root, e.g. an external SSD "
                            "(sets HF_HOME for all hub downloads)")
        q.add_argument("--config")

    q = sub.add_parser("arrange", help="Draft an arrangement for a session")
    engine_args(q)
    q.add_argument("--request")
    q.add_argument("--score", help="Existing ABC score (skips the model)")
    q.add_argument("--arrangement", help="Use this arrangement.json instead of deriving")
    q.add_argument("--output", required=True, help="Session directory")

    q = sub.add_parser("stems", help="Render arrangement stems (one pass per leaf)")
    engine_args(q)
    q.add_argument("--request")
    q.add_argument("--score")
    q.add_argument("--arrangement")
    q.add_argument("--tracks", help="Comma-separated leaf/track names to render")
    q.add_argument("--cleanup", choices=("none", "role-eq"), default="none")
    q.add_argument("--output", required=True)
    q.add_argument("--quiet", "--no-progress", action="store_true")

    q = sub.add_parser("midi", help="Write the session MIDI file")
    q.add_argument("--session", required=True)
    q.add_argument("--midi-type", type=int, choices=(0, 1), default=1)

    q = sub.add_parser("mix", help="Mix stems → buses, FX, master, immersive, MIDI")
    q.add_argument("--session", required=True)

    q = sub.add_parser("deliver", help="Export a bundle or file selection as ZIP")
    q.add_argument("--session", required=True)
    q.add_argument("--bundle", choices=("everything", "midi", "trackout", "stereo",
                                       "binaural", "immersive", "buses", "fx", "session"))
    q.add_argument("--files", nargs="*", help="Session-relative paths for a custom export")
    q.add_argument("--audio-format", default="flac-24",
                   choices=("flac-24", "flac-16", "wav-32f", "wav-24", "wav-16",
                            "mp3", "m4a"))
    q.add_argument("--sample-rate", type=int, choices=(48000, 44100), default=48000)
    q.add_argument("--midi-type", type=int, choices=(0, 1), default=1)
    q.add_argument("--out", help="Output ZIP path")

    q = sub.add_parser("studio", help="YuE2 Studio local API + macOS app")
    engine_args(q)
    q.add_argument("--serve", action="store_true")
    q.add_argument("--open", action="store_true")
    q.add_argument("--port", type=int, default=8787)
    q.add_argument("--sessions-dir")
    q.add_argument("--no-engine", action="store_true",
                   help="Serve without a model (arrange/mix/export only)")
    q.add_argument("--min-free", type=float, default=4.0,
                   help="GiB of free RAM required before rendering "
                        "(lower at your own risk — MPS OOMs cleanly but "
                        "heavy swap makes the Mac unusable)")
    return p


def main(argv=None):
    from ._platform import require_apple_silicon
    require_apple_silicon("YuE2")
    argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(argv)
    # --models-dir relocates the HF cache (e.g. onto a fast external SSD).
    if getattr(args, "models_dir", None):
        os.environ["HF_HOME"] = str(Path(args.models_dir).expanduser())
    return {"doctor": doctor, "generate": generate, "batch": batch,
            "arrange": arrange_cmd, "stems": stems_cmd, "midi": midi_cmd,
            "mix": mix_cmd, "deliver": deliver_cmd, "studio": studio_cmd}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
