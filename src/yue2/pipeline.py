"""One request protocol for local and Hugging Face song generation."""
from __future__ import annotations
from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
from pathlib import Path
import dataclasses
import json
import subprocess
import sys
import threading
import time
import numpy as np
import torch

from ._platform import require_apple_silicon
from .protocol import SongRequest, GenerationConfig, token_prefixes, negative_prefix, CODEC_OFFSET, resolve_sampling
from .storage import resolve_model, model_identity, identity, write_json, collect_hashes, sha256_file, copy_model_files
from .tokenization_yue2 import YuE2TextTokenizer
from .sampling import generate_tokens
from .progress import Progress


#: Free RAM below which "auto" picks CPU over MPS. GPU inference needs
#: weights (~6.2 GiB bf16) plus KV cache and VAE headroom; under this,
#: CPU with swap-friendly malloc is slower but actually completes.
MPS_AUTO_MIN_FREE_GIB = 7.0

#: Free RAM needed for the int8-packed MPS path. Measured: ~2.4 GiB peak
#: during load (safetensors mmap, ~1 GiB packed weights) plus KV + VAE
#: headroom while rendering. Between this and MPS_AUTO_MIN_FREE_GIB,
#: "auto" prefers quantized GPU inference over unquantized CPU inference.
MPS_INT8_MIN_FREE_GIB = 3.0

#: Free RAM the MLX engine needs for each weight format — the FULL working
#: set, not just weights: KV caches (grow to ~3 GiB), torch runtime + VAE
#: (~2 GiB), the Metal allocator pool (≤4 GiB) and working headroom.
#: Weights alone are 7.3/4.7/2.4 GiB for bf16/int8/int4; budgeting for
#: weights only let a bf16 load start with <2 GiB left for KV — instant
#: swap. Imported by the studio server's readiness gate — keep in sync.
MLX_BF16_MIN_FREE_GIB = 13.0
MLX_INT8_MIN_FREE_GIB = 9.0
MLX_INT4_MIN_FREE_GIB = 5.0
MLX_MIN_FREE_GIB = {"none": MLX_BF16_MIN_FREE_GIB,
                    "int8": MLX_INT8_MIN_FREE_GIB,
                    "int4": MLX_INT4_MIN_FREE_GIB}


def _free_gib():
    """Free + inactive + speculative RAM in GiB via vm_stat (macOS)."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True,
                             text=True, timeout=5).stdout
        page_size, pages = 16384, 0
        for line in out.splitlines():
            if "page size of" in line:
                page_size = int(line.split("of")[1].split()[0])
            elif line.startswith(("Pages free:", "Pages inactive:",
                                  "Pages speculative:")):
                pages += int(line.split(":")[1].strip().rstrip("."))
        return pages * page_size / 2**30
    except Exception:
        return None


def _looks_like_oom(error):
    text = f"{type(error).__name__} {error}".lower()
    return ("out of memory" in text or "defaultcpuallocator" in text
            or ("mps" in text and "alloc" in text))


def _on_mps(model):
    try:
        return next(model.parameters()).device.type == "mps"
    except Exception:
        return False


def _int8_mps_supported():
    from .quantization import int8_supported
    return torch.backends.mps.is_available() and int8_supported("mps")


def _int8_cpu_supported():
    from .quantization import int8_supported
    return int8_supported("cpu")


@dataclass
class SymbolicPlan:
    request: SongRequest
    abc: str | None
    abc_ids: list[int]
    prefix: list[int]
    timing: dict = field(default_factory=dict)
    truncated: bool = False

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.abc is not None:
            (directory / "score.abc").write_bytes(self.abc.encode("utf-8"))
        np.save(directory / "abc_tokens.npy", np.asarray(self.abc_ids, dtype=np.int32))
        np.save(directory / "prefix.npy", np.asarray(self.prefix, dtype=np.int32))
        write_json(directory / "plan.json", {"request": self.request.to_dict(), "timing": self.timing,
                                              "truncated": self.truncated, "prefix": self.prefix,
                                              "abc_ids": self.abc_ids, "abc": self.abc})


        names = ["plan.json", "abc_tokens.npy", "prefix.npy"] + (["score.abc"] if self.abc is not None else [])
        write_json(directory / "plan_manifest.json", {name: sha256_file(directory / name) for name in names})

    @classmethod
    def load(cls, directory):
        """Restore exact planner output without decoding and retokenizing its ABC."""
        directory = Path(directory)
        hashes = json.loads((directory / "plan_manifest.json").read_text())
        if not {"plan.json", "abc_tokens.npy", "prefix.npy"} <= hashes.keys():
            raise ValueError("Incomplete saved plan")
        for name, digest in hashes.items():
            if name not in {"plan.json", "abc_tokens.npy", "prefix.npy", "score.abc"} or (directory / name).is_symlink():
                raise ValueError("Invalid plan artifact")
            if sha256_file(directory / name) != digest:
                raise ValueError("Saved plan changed; supply modified ABC as an external planner input")
        data = json.loads((directory / "plan.json").read_text())
        for field, filename in (("abc_ids", "abc_tokens.npy"), ("prefix", "prefix.npy")):
            array = np.load(directory / filename, allow_pickle=False)
            if array.ndim != 1 or array.dtype.kind not in "iu" or array.tolist() != data[field]:
                raise ValueError("Saved plan token array mismatch")
        if data["abc"] is not None and ("score.abc" not in hashes or (directory / "score.abc").read_bytes() != data["abc"].encode("utf-8")):
            raise ValueError("Saved ABC text mismatch")
        return cls(SongRequest(**data["request"]), data["abc"], data["abc_ids"], data["prefix"],
                   data["timing"], data["truncated"])


@dataclass
class SemanticResult:
    plan: SymbolicPlan
    tokens: list[int]
    timing: dict
    truncated: bool


@dataclass
class SongResult:
    audio: np.ndarray
    sample_rate: int
    semantic: SemanticResult
    latents: np.ndarray
    config: dict
    weights: dict
    timing: dict
    request_identity: str

    @property
    def abc(self):
        return self.semantic.plan.abc

    @property
    def truncated(self):
        return {"abc": self.semantic.plan.truncated, "semantic": self.semantic.truncated}

    def save(self, path):
        """Write audio; use save_artifacts(directory) to retain a reproducible run."""
        import soundfile as sf
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() not in {".flac", ".wav"}:
            raise ValueError("Use .flac or .wav; MP3 is an optional delivery conversion")
        sf.write(path, self.audio, self.sample_rate, subtype="PCM_24" if path.suffix.lower() == ".flac" else "FLOAT")
        return str(path)

    def save_artifacts(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.semantic.plan.save(directory)
        self.save(directory / "audio.flac")
        np.save(directory / "semantic.npy", np.asarray(self.semantic.tokens, dtype=np.int32))
        np.save(directory / "latent.npy", self.latents.astype(np.float32))
        write_json(directory / "request.json", self.semantic.plan.request.to_dict())
        write_json(directory / "config.json", self.config)
        result = {"status": "complete", "identity": self.request_identity,
                  "truncated": self.truncated, "sample_rate": self.sample_rate,
                  "audio_seconds": len(self.audio) / self.sample_rate,
                  "weights": self.weights, "timing": self.timing,
                  "artifacts": collect_hashes(directory)}
        write_json(directory / "result.json", result)
        return result


class YuE2Pipeline:
    #: Default engine for instances built without __init__ (test doubles).
    engine = "torch"

    def __init__(self, model_dir, vae_dir, *, device="auto", memory_budget_gib=24,
                 generation_config=None, verify_hashes=True,
                 vae_core_frames=None, quantization="auto", offload_ar=False,
                 progress=True, engine="auto"):
        require_apple_silicon("YuE2Pipeline")
        if not isinstance(progress, bool):
            raise TypeError("progress must be True or False")
        self.progress = progress
        # Engine: "auto" prefers the native MLX path on Apple Silicon —
        # int4 weights stay resident and decode ~3-4x faster than the
        # torch-MPS int8 path. "torch" keeps the MPS/CPU fallback whole.
        if engine not in {"auto", "mlx", "torch"}:
            raise ValueError("engine must be auto, mlx, or torch")
        if engine == "auto":
            try:
                import mlx.core  # noqa: F401
                engine = "mlx"
            except ImportError:
                engine = "torch"
        if engine == "mlx":
            try:
                import mlx.core  # noqa: F401
            except ImportError as error:
                raise RuntimeError(
                    "engine='mlx' needs the mlx package — pip install mlx") from error
        self.engine = engine
        # Apple Silicon only — Metal (MPS) or CPU. No CUDA anywhere.
        if quantization not in {"none", "int4", "int8", "auto"}:
            raise ValueError("quantization must be none, int4, int8, or auto")
        if quantization == "int4" and engine != "mlx":
            raise ValueError("int4 quantization requires engine='mlx'")
        if str(device).startswith("cuda"):
            raise ValueError("CUDA is unsupported — this build targets Apple Silicon (mps/cpu)")
        if not 0 < memory_budget_gib:
            raise ValueError("memory_budget_gib must be positive")
        if self.engine == "mlx" and quantization == "auto":
            # Best fidelity that fits with headroom — quality first, speed
            # second. bf16 is reference; int8 is transparent; int4 is the
            # last resort before refusing. Thresholds cover the whole
            # working set (weights + KV + torch + Metal pool), not just the
            # weight file — a bf16 load that only fits its weights swaps
            # the moment the KV cache allocates. Resolve BEFORE device=auto
            # so its int8 remap can't claim "auto".
            free = _free_gib()
            quantization = ("none" if free is not None and free >= MLX_BF16_MIN_FREE_GIB
                            else "int8" if free is None or free >= MLX_INT8_MIN_FREE_GIB
                            else "int4")
        if device == "auto":
            if torch.backends.mps.is_available():
                free = _free_gib()
                if free is None or free >= MPS_AUTO_MIN_FREE_GIB:
                    device = "mps"
                elif quantization in {"auto", "int8"} and free >= MPS_INT8_MIN_FREE_GIB \
                        and _int8_mps_supported():
                    device, quantization = "mps", "int8"
                else:
                    device = "cpu"
            else:
                device = "cpu"
        self.device = torch.device(device)
        if quantization == "auto":
            quantization = "none"
            if self.device.type == "mps":
                free = _free_gib()
                if free is not None and MPS_INT8_MIN_FREE_GIB <= free < MPS_AUTO_MIN_FREE_GIB \
                        and _int8_mps_supported():
                    quantization = "int8"
        if quantization == "int8" and engine == "torch" \
                and not _int8_mps_supported():
            # The op also exists on CPU — only reject truly absent kernels.
            if not _int8_cpu_supported():
                raise RuntimeError(
                    "int8 quantization needs aten._weight_int8pack_mm — use 'none'")
        # fp32 matmul precision also governs the int8-packed path's
        # fp32 activations — keep it exact.
        torch.set_float32_matmul_precision("highest")
        self.model_dir, self.vae_dir = Path(model_dir), Path(vae_dir)
        self.quantization = quantization
        self.memory_budget_gib = float(memory_budget_gib)
        self.vae_core_frames = vae_core_frames if vae_core_frames is not None else (512 if memory_budget_gib <= 12 else 1024)
        self.offload_ar = offload_ar
        self.generation_config = generation_config or GenerationConfig()
        self.tokenizer = YuE2TextTokenizer(self.model_dir / "qwen.tiktoken")
        self._verify_hashes = verify_hashes
        self._weights_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._weights = None
        self._weights_error = None
        with self._status("Verifying model files"):
            for required_dir in (self.model_dir, self.vae_dir):
                if not list(required_dir.glob("*.safetensors")):
                    raise FileNotFoundError(
                        f"No safetensors weights in {required_dir}")
            # Full sha256 identity streams in the background — ~7 GB of reads
            # overlap model load instead of blocking startup on slow drives.
            threading.Thread(target=self._hash_weights,
                             name="yue2-weights-id", daemon=True).start()
        self.runtime_sha256 = identity({p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
        self._model, self._vae = None, None
        self.load_timing = {}

    @classmethod
    def from_pretrained(cls, model="m-a-p/YuE2-3B", *, vae="m-a-p/YuE2-Vae",
                        revision=None, vae_revision=None, local_files_only=False,
                        token=None, cache_dir=None, progress=True, **kwargs):
        """Load a song pipeline with English progress on stderr; set progress=False to hide it."""
        if not isinstance(progress, bool):
            raise TypeError("progress must be True or False")
        start = time.perf_counter()
        saved = Path(model) / "pipeline.json"
        if saved.is_file():
            metadata = json.loads(saved.read_text())
            parent = Path(model)
            model = parent / metadata["model"]
            if vae == "m-a-p/YuE2-Vae":
                vae = parent / metadata["vae"]
            kwargs.setdefault("generation_config", GenerationConfig.from_dict(metadata["generation_config"]))
        hub = dict(local_files_only=local_files_only, token=token, cache_dir=cache_dir)
        with Progress(enabled=progress).stage("Resolving model files"):
            model_path = resolve_model(model, revision=revision, **hub)
            vae_path = resolve_model(vae, revision=vae_revision, **hub)
        result = cls(model_path, vae_path, progress=progress, **kwargs)
        result.load_timing["resolve_and_integrity_seconds"] = time.perf_counter() - start
        return result

    @contextmanager
    def _status(self, label, *, total=None, unit=None):
        # Display state never enters the request/config identity or model RNG.
        with Progress(enabled=self.progress) as reporter:
            with reporter.stage(label, total=total, unit=unit) as stage:
                yield stage

    def _hash_weights(self):
        try:
            weights = {"mot": model_identity(self.model_dir, self._verify_hashes),
                       "vae": model_identity(self.vae_dir, self._verify_hashes)}
        except Exception as error:
            with self._weights_lock:
                self._weights_error = error
            return
        with self._weights_lock:
            self._weights = weights

    @property
    def weights(self):
        """Weight-file identity — hashed in a background thread at
        construction; the first access waits for it if still running."""
        while True:
            with self._weights_lock:
                if self._weights is not None:
                    return self._weights
                error = self._weights_error
            if error is not None:
                raise error
            time.sleep(0.05)

    def save_pretrained(self, directory):
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("save_pretrained needs an empty pipeline destination")
        for name, source in (("YuE2-3B", self.model_dir), ("YuE2-Vae", self.vae_dir)):
            destination = directory / name
            if destination.resolve() == source.resolve():
                continue
            copy_model_files(source, destination)
        write_json(directory / "pipeline.json", {"model": "YuE2-3B", "vae": "YuE2-Vae",
                   "generation_config": self.generation_config.to_dict(), "source_weights": self.weights})

    @staticmethod
    def _load_hint(error):
        """Translate a raw load failure into the action that fixes it."""
        text = f"{type(error).__name__} {error}".lower()
        if "safetensors" in text or "header" in text or "incomplete file" in text:
            return ("weight files appear corrupt or incomplete — "
                    "delete the model folder and re-download")
        if "out of memory" in text or ("mps" in text and "alloc" in text) \
                or "defaultcpuallocator" in text:
            return ("not enough memory to load the model — quit heavy apps, "
                    "or switch the compute device to CPU in Settings")
        return None

    def warmup(self):
        """Preload model weights so the first request doesn't pay the cost."""
        self._load_model()

    def _load_model(self, for_nar=False):
        # Serialized — a warmup thread and a job must never double-load.
        with self._load_lock:
            return self._load_model_inner(for_nar)

    def _load_model_inner(self, for_nar=False):
        if self.engine == "mlx":
            # Unified memory — loaded once, never moved, never reloaded.
            if self._model is None:
                import mlx.core as mx
                from .mlx_yue2 import load_weights
                # Cap Metal's buffer cache — without a bound it retains
                # every freed KV/activation block and grows ~GiBs per stem
                # until the machine swaps.
                mx.metal.set_cache_limit(4 * 1024 ** 3)
                bits = {"int4": 4, "int8": 8}.get(self.quantization)
                # Hard floor at the point of commitment — the server's
                # min-free check only gates job creation; a direct pipeline
                # call (or memory eaten after the check) must not be able
                # to force the whole system into swap. Full working-set
                # floors (weights + KV + torch + Metal pool), not just
                # the weight footprint.
                need_gib = {4: MLX_INT4_MIN_FREE_GIB, 8: MLX_INT8_MIN_FREE_GIB,
                            None: MLX_BF16_MIN_FREE_GIB}[bits]
                free = _free_gib()
                if free is not None and free < need_gib:
                    raise RuntimeError(
                        f"Not enough free memory to load the model — need "
                        f"~{need_gib:.0f} GiB, have {free:.1f} GiB. Quit "
                        f"memory-heavy apps and retry.")
                with self._status("Loading model"):
                    start = time.perf_counter()
                    self._model = load_weights(self.model_dir, bits=bits)
                    self.load_timing["mot_load_seconds"] = time.perf_counter() - start
            return self._model
        loading = self._model is None or next(self._model.parameters()).device != self.device
        with self._status("Loading model") if loading else nullcontext():
            try:
                if self._model is None:
                    from .modeling_yue2 import YuE2ForCausalLM
                    start = time.perf_counter()
                    self._model = YuE2ForCausalLM.from_pretrained(self.model_dir, local_files_only=True,
                                  dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
                    self.load_timing["mot_load_seconds"] = time.perf_counter() - start
                if self.quantization == "int8":
                    # Pack on CPU before .to(device): halves both the
                    # unified-memory peak and the GPU transfer.
                    from .quantization import prepare_int8_ar
                    prepare_int8_ar(self._model)
                self._model.to(self.device)
                # Drop the mmap'd bf16 file pages and freed CPU copies so
                # they stop inflating the footprint under memory pressure.
                import gc
                gc.collect()
                if self.device.type == "mps" and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception as error:
                # GPU allocation failed with a constructed model — move it
                # to CPU and keep going. Slow beats dead.
                # A .to(mps) that OOMs mid-move may leave the first parameter
                # on CPU — check the intended device, not just the current one.
                if self._model is not None and _looks_like_oom(error) \
                        and (self.device.type == "mps" or _on_mps(self._model)):
                    print("[YuE2] GPU out of memory — falling back to CPU "
                          "(slower, but it will run)", file=sys.stderr)
                    self.device = torch.device("cpu")
                    self._model.to(self.device)
                else:
                    hint = self._load_hint(error)
                    if hint:
                        raise RuntimeError(
                            f"{type(error).__name__}: {error} — {hint}") from error
                    raise
        return self._model

    def _request(self, style=None, lyrics=None, *, tags=None, **kwargs):
        if style is not None and tags is not None and style != tags:
            raise ValueError("style and tags are aliases and cannot disagree")
        style = tags if style is None else style
        if style is None or lyrics is None:
            raise ValueError("Provide style and lyrics")
        return SongRequest(style=style, lyrics=lyrics, **kwargs)

    def _generate(self, prefix, sampling, seed, phase, **kwargs):
        cancelled = kwargs.get("cancelled")
        if cancelled is not None and cancelled():
            raise InterruptedError("cancelled")
        model = self._load_model()
        callback = kwargs.pop("on_token", None)
        label = "Planning score" if phase == "abc" else "Generating song"
        # An MPS OOM mid-generation is not fatal: the seed makes the output
        # deterministic, so moving to CPU and restarting yields the same
        # result — slower, but it finishes.
        while True:
            with self._status(label, unit="tokens") as status:
                def on_output(token_phase, token):
                    # The backend emits each real output once, including its end token.
                    # Prefixes, provided scores and extra CFG branches are not output.
                    status.advance()
                    if callback is not None:
                        callback(token_phase, token)
                observed = on_output if self.progress else callback
                try:
                    if self.engine == "mlx":
                        from .mlx_generate import generate_tokens_mlx
                        result = generate_tokens_mlx(
                            model, prefix, sampling, seed, phase,
                            on_token=observed, **kwargs)
                    else:
                        result = generate_tokens(model, prefix, sampling, seed, phase,
                                                 on_token=observed, **kwargs)
                except Exception as error:
                    if model is not None and self.engine == "torch" \
                            and _looks_like_oom(error) and _on_mps(model):
                        print("[YuE2] GPU out of memory mid-generation — "
                              "retrying on CPU (deterministic, just slower)",
                              file=sys.stderr)
                        self.device = torch.device("cpu")
                        model.to(self.device)
                        continue
                    raise
                if result[2]:
                    status.finish(status="truncated")
                return result

    def plan(self, style=None, lyrics=None, *, tags=None, request=None, abc_sampling=None,
             cancelled=None, on_token=None, **kwargs):
        request = request or self._request(style, lyrics, tags=tags, **kwargs)
        if request.cot == "off":
            return SymbolicPlan(request, None, [], token_prefixes(request, self.tokenizer))
        if request.abc is not None:
            with self._status("Using provided score"):
                ids = self.tokenizer.encode(request.abc)
                return SymbolicPlan(request, request.abc, ids, token_prefixes(request, self.tokenizer, ids),
                                    {"seconds": 0., "output_tokens": 0, "external_prefix_tokens": len(ids)})
        sampling = resolve_sampling(abc_sampling, self.generation_config.abc)
        ids, timing, truncated = self._generate(token_prefixes(request, self.tokenizer), sampling,
                            request.seed, "abc", cancelled=cancelled, on_token=on_token)
        return SymbolicPlan(request, self.tokenizer.decode(ids), ids,
                            token_prefixes(request, self.tokenizer, ids), timing, truncated)

    def generate_semantic(self, plan, *, sampling=None, cancelled=None, on_token=None):
        if not isinstance(plan, SymbolicPlan):
            raise TypeError("Pass the SymbolicPlan returned by pipe.plan()")
        request = plan.request
        expected = token_prefixes(request, self.tokenizer, plan.abc_ids)
        if expected != plan.prefix:
            raise ValueError("Plan prefix disagrees with request/exact ABC IDs")
        sampling = resolve_sampling(sampling, self.generation_config.semantic)
        negative = negative_prefix(request, self.tokenizer, plan.abc_ids) if request.guidance != 1 else None
        ids, timing, truncated = self._generate(plan.prefix, sampling, request.seed, "semantic",
                        negative=negative, cfg_scale=request.guidance, legacy_off=request.cot == "off",
                        cancelled=cancelled, on_token=on_token)
        return SemanticResult(plan, [int(t) - CODEC_OFFSET for t in ids], timing, truncated)

    def synthesize(self, semantic, *, cancelled=None, on_progress=None):
        if not isinstance(semantic, SemanticResult):
            raise TypeError("Pass the SemanticResult returned by generate_semantic()")
        if token_prefixes(semantic.plan.request, self.tokenizer, semantic.plan.abc_ids) != semantic.plan.prefix:
            raise ValueError("Semantic result does not retain the request's exact prefix")
        model = self._load_model(for_nar=True)
        unit = "evals" if self.engine == "mlx" else "steps"
        with self._status("Synthesizing audio", unit=unit) as status:
            report = None
            if self.progress or on_progress is not None:
                def report(completed, total):
                    if self.progress:
                        status.update(completed, total=total)
                    if on_progress:
                        on_progress(completed, total)
            ode_steps = (semantic.plan.request.ode_steps
                         or self.generation_config.ode_steps)
            ode_method = (semantic.plan.request.ode_method
                          or self.generation_config.ode_method)
            if self.engine == "mlx":
                from .mlx_nar import synthesize as mlx_synthesize
                result = mlx_synthesize(
                    model, semantic.plan.prefix, semantic.tokens,
                    semantic.plan.request.seed,
                    steps=ode_steps, method=ode_method,
                    context=self.generation_config.context,
                    cancelled=cancelled, on_progress=report)
            else:
                from .nar import synthesize
                result = synthesize(model, semantic.plan.prefix, semantic.tokens,
                                    semantic.plan.request.seed, steps=ode_steps,
                                    context=self.generation_config.context, offload_ar=self.offload_ar,
                                    cancelled=cancelled, on_progress=report)
            return result.detach().float().cpu().numpy()

    def close(self):
        self._model, self._vae = None, None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def decode(self, latents, *, full=False, vae=None, cancelled=None):
        from .modeling_vae import YuE2VAE
        with self._status("Loading audio decoder"):
            # The VAE needs ~1 GiB of workspace — on unified memory it
            # decodes fine alongside a resident AR model, so don't pay two
            # multi-GiB transfers per stem unless memory is genuinely tight
            # (low free RAM, or the user opted into offload).
            needs_ar_offload = (self.engine == "torch" and
                                (self.offload_ar or (_free_gib() or 0.0) < 3.0))
            if self._model is not None and needs_ar_offload:
                self._model.to("cpu")
            if needs_ar_offload and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if vae is not None:
                model = YuE2VAE.from_pretrained(vae, decoder_only=True, device=self.device)
            else:
                if self._vae is None:
                    self._vae = YuE2VAE.from_pretrained(self.vae_dir, decoder_only=True, device="cpu",
                                                        local_files_only=True)
                model = self._vae.to(self.device)
        z = torch.as_tensor(latents, dtype=torch.float32)
        if z.ndim == 2 and z.shape[1] == 64:
            z = z.T.unsqueeze(0)
        if z.ndim != 3 or z.shape[0] != 1 or z.shape[1] != 64:
            raise ValueError("Expected latents [T,64] or [1,64,T]")
        try:
            tiles = 1 if full else (z.shape[-1] + self.vae_core_frames - 1) // self.vae_core_frames
            with self._status("Decoding audio", total=tiles, unit="chunks") as status:
                report = None
                if self.progress or cancelled is not None:
                    def report(completed, total):
                        if cancelled is not None and cancelled():
                            raise InterruptedError("Cancelled during VAE decode")
                        if self.progress:
                            status.update(completed, total=total)
                with torch.inference_mode():
                    if full:
                        audio = model.decode(z.to(self.device)).cpu()
                        status.update(1)
                    else:
                        audio = model.decode_tiled(z, core_frames=self.vae_core_frames, halo_frames=16,
                                                   output_device="cpu", on_progress=report)
                if not torch.isfinite(audio).all():
                    raise ValueError("VAE produced non-finite audio")
                return audio[0].float().clamp(-1, 1).T.contiguous().numpy()
        finally:
            model.to("cpu")
            if self._model is not None and needs_ar_offload:
                # Bring the AR model back to the accelerator it lives on —
                # leaving it stranded on CPU forces a multi-GiB transfer
                # before every subsequent stem.
                try:
                    self._model.to(self.device)
                except Exception:
                    pass  # _load_model retries with full OOM handling
            if torch.backends.mps.is_available():
                # Release the VAE workspace so the next stem's KV/logits
                # allocations aren't squeezed by a cached decoder block.
                torch.mps.empty_cache()

    def effective_config(self, request, abc_sampling=None, semantic_sampling=None):
        config = self.generation_config.to_dict()
        for key, sampling in (("abc", abc_sampling), ("semantic", semantic_sampling)):
            if sampling is not None:
                config[key] = dataclasses.asdict(resolve_sampling(sampling, getattr(self.generation_config, key)))
        defaults = GenerationConfig().to_dict()
        overrides = {k: v for k, v in config.items() if defaults.get(k) != v}
        if request.guidance != (1.01 if request.cot == "off" else 1.0):
            overrides["cfg_scale"] = request.guidance
        return {"generation": config, "overrides": overrides,
                "cot": request.cot, "cfg_scale": request.guidance,
                "cfg_negative": "instruction_only" if request.cot == "off" else "same_instruction_and_exact_abc",
                "quantization": self.quantization,
                "model_dtype": "bfloat16", "vae_dtype": "float32", "vae_decode": "halo_crop",
                "vae_core_frames": self.vae_core_frames, "vae_halo_frames": 16,
                "device": str(self.device), "engine": self.engine,
                "memory_budget_gib": self.memory_budget_gib,
                "offload_ar": self.offload_ar, "runtime_sha256": self.runtime_sha256,
                "decoder_release": json.loads((self.vae_dir / "config.json").read_text()).get("release_variant"),
                "validation_status": "unvalidated"}

    def __call__(self, style=None, lyrics=None, *, tags=None, abc_sampling=None,
                 semantic_sampling=None, cancelled=None, on_token=None,
                 on_progress=None, **kwargs):
        request = self._request(style, lyrics, tags=tags, **kwargs)
        config = self.effective_config(request, abc_sampling, semantic_sampling)
        request_id = identity({"request": request.to_dict(), "config": config, "weights": self.weights})
        start = time.perf_counter()
        plan = self.plan(request=request, abc_sampling=abc_sampling, cancelled=cancelled, on_token=on_token)
        semantic = self.generate_semantic(plan, sampling=semantic_sampling, cancelled=cancelled, on_token=on_token)
        nar_start = time.perf_counter()
        latents = self.synthesize(semantic, cancelled=cancelled,
                                  on_progress=on_progress)
        nar_seconds = time.perf_counter() - nar_start
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before VAE")
        vae_start = time.perf_counter()
        audio = self.decode(latents, cancelled=cancelled)
        timing = {"abc": plan.timing, "semantic": semantic.timing, "nar_seconds": nar_seconds,
                  "vae_seconds": time.perf_counter() - vae_start, "load": dict(self.load_timing),
                  "e2e_seconds": time.perf_counter() - start}
        Progress(enabled=self.progress).complete(len(audio) / 48000, timing["e2e_seconds"],
                                                truncated=plan.truncated or semantic.truncated)
        return SongResult(audio, 48000, semantic, latents, config, self.weights, timing, request_id)
