"""Benchmark NAR ODE solvers on real weights — time vs latent fidelity.

Loads the model once, solves one deterministic acoustic chunk under each
``(method, steps)`` config, and reports wall time plus distance to the
midpoint@32 reference. Picks the session-sheet "Render detail" tiers on
measurement instead of guessing.

    python scripts/bench_solver.py
    python scripts/bench_solver.py --bits 8 --configs midpoint:32,ab2:16
    python scripts/bench_solver.py --frames 1500 --wav ab2:24

Requires the YuE2-3B checkpoint in the HF cache (or --model-dir). Not a
CI test — real weights, minutes of GPU time.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from yue2.mlx_nar import MlxCachedNAR, _EVALS_PER_STEP  # noqa: E402
from yue2.mlx_yue2 import load_weights  # noqa: E402
from yue2.nar import song_chunks  # noqa: E402
from yue2.protocol import CONTEXT  # noqa: E402
from yue2.storage import resolve_model  # noqa: E402


def build_chunk(frames: int, prefix_tokens: int, seed: int):
    """One deterministic chunk — codec IDs well inside the codec vocab."""
    rng = np.random.default_rng(seed)
    codec = rng.integers(0, 1024, size=frames).tolist()
    prefix = rng.integers(1, 40000, size=prefix_tokens).tolist()
    return song_chunks(prefix, codec, seed, CONTEXT)[0]


def solve_once(model, chunk, method: str, steps: int):
    engine = MlxCachedNAR(model, chunk)
    try:
        start = time.perf_counter()
        latents = engine.solve(steps, method=method)
        return latents, time.perf_counter() - start
    finally:
        engine.close()


def metrics(latents: torch.Tensor, reference: torch.Tensor):
    diff = (latents - reference).double()
    l2 = diff.norm().item() / reference.double().norm().item()
    cos = torch.nn.functional.cosine_similarity(
        latents.flatten().double(), reference.flatten().double(), dim=0).item()
    return l2, cos, diff.abs().max().item()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", default=None)
    p.add_argument("--bits", type=int, default=4, choices=[4, 8])
    p.add_argument("--frames", type=int, default=3000,
                   help="codec frames per chunk (~25/sec of audio)")
    p.add_argument("--prefix-tokens", type=int, default=2000)
    p.add_argument("--seed", type=int, default=831001)
    p.add_argument("--configs", default="midpoint:32,midpoint:16,heun:12,"
                                        "rk4:8,rk4:12,ab2:12,ab2:16,ab2:24")
    p.add_argument("--reference", default="midpoint:32")
    p.add_argument("--wav", default=None, metavar="METHOD:STEPS",
                   help="also VAE-decode this config to bench_METHOD_STEPS.wav")
    p.add_argument("--out", default=".")
    args = p.parse_args()

    model_dir = args.model_dir or resolve_model(
        "m-a-p/YuE2-3B", local_files_only=True)
    print(f"weights: {model_dir} (int{args.bits})", flush=True)
    mx.metal.set_cache_limit(4 * 1024 ** 3)
    model = load_weights(model_dir, bits=args.bits)
    mx.eval(list(model.parameters()))
    print("weights loaded", flush=True)

    chunk = build_chunk(args.frames, args.prefix_tokens, args.seed)
    print(f"chunk: ar={len(chunk.ar_tokens)} nar={len(chunk.noise) + 2}",
          flush=True)

    configs = [tuple(c.split(":")) for c in args.configs.split(",")]
    ref_key = tuple(args.reference.split(":"))
    results, reference = {}, None

    for method, steps_s in configs:
        steps = int(steps_s)
        latents, seconds = solve_once(model, chunk, method, steps)
        results[(method, steps)] = (latents, seconds)
        if (method, steps_s) == ref_key or (method, steps) == tuple(
                [ref_key[0], int(ref_key[1])]):
            reference = latents
        mx.metal.clear_cache()
        print(f"  {method:8s}@{steps:<3d} {seconds:7.1f}s "
              f"({seconds / (steps * _EVALS_PER_STEP[method]):.2f}s/eval)",
              flush=True)

    if reference is None:
        print("warning: reference config not run — skipping distances")
    print(f"\n{'method':<9}{'steps':>6}{'evals':>6}{'time':>8}"
          f"{'rel L2':>10}{'cosine':>10}{'max|d|':>10}")
    for method, steps_s in configs:
        steps = int(steps_s)
        latents, seconds = results[(method, steps)]
        row = f"{method:<9}{steps:>6}{steps * _EVALS_PER_STEP[method]:>6}" \
              f"{seconds:>7.1f}s"
        if reference is not None:
            l2, cos, mx_d = metrics(latents, reference)
            row += f"{l2:>10.4f}{cos:>10.5f}{mx_d:>10.3f}"
        print(row)

    if args.wav:
        method, steps_s = args.wav.split(":")
        latents = results.get((method, int(steps_s)), (None, None))[0]
        if latents is None:
            sys.exit(f"--wav config {args.wav} was not in --configs")
        from yue2.pipeline import YuE2Pipeline
        pipe = YuE2Pipeline.from_pretrained(local_files_only=True,
                                            engine="mlx")
        audio = pipe.decode(latents)
        path = Path(args.out) / f"bench_{method}_{steps_s}.wav"
        import soundfile as sf
        sf.write(path, audio, 48000)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
