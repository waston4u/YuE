"""Local HTTP/SSE API for the YuE2 Studio app — stdlib only.

127.0.0.1 only, one job at a time, sessions resumable via artifact
verification. The engine is pluggable so the API is testable without a GPU:
``StudioServer(engine=...)`` accepts any object with ``__call__`` producing
a SongResult-like object with ``audio``, ``sample_rate`` and
``save_artifacts()``, plus ``plan()`` for score generation.
"""
from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ..arrange import (ENSEMBLE_PRESETS, INSTRUMENT_CATALOG, KIT_PRESETS,
                       arrangement_from_picks, derive_arrangement,
                       expand_leaves, load_arrangement, make_kit, make_track,
                       save_arrangement)
from ..deliver import available_audio_formats, deliver
from ..mix import default_session, render_session, save_session
from ..score import parse
from ..stems import StemSession
from ..storage import identity, write_json

DEFAULT_PORT = 8787

#: Free RAM required before the engine may touch the model. The 3B bf16
#: weights (~6.2 GiB) move to MPS as non-evictable memory; below this the
#: machine swaps to death instead of generating.
MIN_FREE_GIB = 4.0  # hard floor — below this even starting is hopeless;
                    # above it we attempt and let MPS OOM cleanly if needed


_mem_cache = {"at": 0.0, "value": None}


def free_memory_gib() -> float | None:
    """Free + inactive + speculative RAM in GiB via vm_stat (macOS)."""
    now = time.time()
    if now - _mem_cache["at"] < 5:
        return _mem_cache["value"]
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
        _mem_cache.update(at=now, value=pages * page_size / 2**30)
    except Exception:
        _mem_cache.update(at=now, value=None)
    return _mem_cache["value"]


#: With weights already resident, a render's marginal cost is the KV cache,
#: activations and the VAE workspace — not another model load. A warm engine
#: may proceed with far less free RAM than a cold start needs.
WARM_MIN_FREE_GIB = 1.5

#: A cold int8-on-MPS load peaks near ~2.4 GiB — safetensors weights mmap,
#: so the bf16 pages never fully fault in and the packed copy is ~1 GiB.
INT8_LOAD_MIN_FREE_GIB = 3.0


def engine_warm(state):
    engine = getattr(state, "_engine", None)
    return engine is not None and getattr(engine, "_model", None) is not None


def engine_loading(state):
    """A model load is already in flight (startup warmup or an earlier job) —
    its memory is committed either way, so a new render should wait on it,
    not be refused."""
    engine = getattr(state, "_engine", None)
    if engine is None or engine_warm(state):
        return False
    lock = getattr(engine, "_load_lock", None)
    return bool(lock is not None and lock.locked())


def needed_free_gib(state):
    engine = getattr(state, "_engine", None)
    if engine_warm(state) or engine_loading(state):
        return min(state.min_free_gib, WARM_MIN_FREE_GIB)
    if engine is not None and getattr(engine, "engine", None) == "mlx":
        # MLX cold-load floors match _load_model_inner — the shared
        # working-set constants (weights + KV + torch + Metal pool), so a
        # bf16 pick can't start with only its weight footprint available.
        from ..pipeline import MLX_MIN_FREE_GIB
        need = MLX_MIN_FREE_GIB.get(
            getattr(engine, "quantization", None), state.min_free_gib)
        return max(state.min_free_gib, need)
    if engine is not None and getattr(engine, "quantization", None) == "int8" \
            and str(getattr(engine, "device", "")).startswith("mps"):
        return min(state.min_free_gib, INT8_LOAD_MIN_FREE_GIB)
    return state.min_free_gib


def check_memory(state, free_gib=None):
    free = free_memory_gib() if free_gib is None else free_gib
    needed = needed_free_gib(state)
    if free is not None and free < needed:
        raise RuntimeError(
            f"Not enough free memory to render — need ~{needed:.0f} GiB "
            f"available, have {free:.1f} GiB. Quit memory-heavy apps "
            "(VMs, IDEs) and retry.")


MIN_DISK_GIB = 2.0


def check_disk(directory):
    free = shutil.disk_usage(directory).free / 2**30
    if free < MIN_DISK_GIB:
        raise RuntimeError(
            f"Not enough disk space — need ~{MIN_DISK_GIB:.0f} GiB free "
            f"for stems and renders, have {free:.1f} GiB.")


def readiness_problems(state, directory=None):
    """Everything that would make a render/session fail — reported all at
    once instead of one cryptic exception at a time."""
    problems = []
    if state.require_weights:
        status = model_status()
        corrupt = [
            f"{e['repo'].split('/')[-1]} ({', '.join(e['corrupt'])})"
            for e in status["models"].values() if e.get("corrupt")]
        if corrupt:
            problems.append("model weights corrupt: " + ", ".join(corrupt)
                            + " — delete the model folder and re-download")
        missing = [
            f"{e['repo'].split('/')[-1]} "
            f"({e['bytes'] / 1e9:.1f}/{(e['expected'] or 0) / 1e9:.1f} GB)"
            for e in status["models"].values()
            if not e["downloaded"] and not e.get("corrupt")]
        if missing:
            problems.append("model weights incomplete: "
                            + ", ".join(missing)
                            + " — finish the download first")
    free = free_memory_gib()
    needed = needed_free_gib(state)
    if free is not None and free < needed:
        problems.append(f"low memory: {free:.1f} GiB free "
                        f"(need ~{needed:.0f}) — quit heavy apps")
    if directory is not None:
        disk = shutil.disk_usage(directory).free / 2**30
        if disk < MIN_DISK_GIB:
            problems.append(f"low disk: {disk:.1f} GiB free "
                            f"(need ~{MIN_DISK_GIB:.0f})")
    return problems


def check_ready(state, directory=None):
    problems = readiness_problems(state, directory)
    if problems:
        raise RuntimeError("Preflight failed: " + " | ".join(problems))


def _leaf_done_fast(stems_dir: Path, name: str) -> bool:
    """Light completion check for snapshots — leaf_done() hashes every
    artifact, far too heavy to run per-broadcast over N stems."""
    try:
        result = json.loads((stems_dir / name / "result.json").read_text())
        return (result.get("status") == "complete"
                and (stems_dir / name / "audio.flac").is_file())
    except Exception:
        return False


def _engine_label(state):
    """Honest load label — the startup warmup shares the same model load,
    so a job arriving mid-warmup is waiting on it, not reloading."""
    if engine_warm(state):
        return "model ready"
    if engine_loading(state):
        return "waiting for model warmup"
    return "loading model"


def _plan_and_parse(state, request, progress, cancelled, attempts=3):
    """Plan until the model emits a score that parses — one malformed ABC
    must not kill a session. The seed nudge keeps every attempt
    deterministic; the first attempt uses the user's seed unchanged."""
    from ..protocol import SongRequest
    from ..score import ScoreError
    last = None
    for attempt in range(attempts):
        plan = state.engine.plan(request=SongRequest(
            style=request["style"], lyrics=request["lyrics"],
            cot=request["cot"], seed=request["seed"] + attempt,
            ode_steps=request.get("ode_steps"),
            ode_method=request.get("ode_method")),
            cancelled=cancelled, on_token=_token_progress(progress))
        score_text = plan.abc or ""
        if not score_text:
            raise ScoreError(
                "planning produced no score — 'cot: off' cannot drive orchestration")
        try:
            return score_text, parse(score_text)
        except ScoreError as error:
            last = error
            progress("score", 0, 1,
                     f"score failed to parse — replanning ({attempt + 1}/{attempts})")
    raise last


def _token_progress(progress):
    """Adapt pipe's per-token callback into Job.progress updates —
    throttled so SSE isn't flooded at generation speed. Counts are
    per-item: each stem gets its own index/rate/ETA against its own
    total — a cumulative job counter makes the bar overflow and the
    ETA climb at every new stem."""
    state = {"n": 0, "t0": 0.0, "item": None}

    def on_token(phase, _token, item="engine", total=0):
        if item != state["item"]:
            state["item"], state["n"], state["t0"] = item, 0, time.time()
        state["n"] += 1
        if state["n"] % 8:
            return
        elapsed = max(0.1, time.time() - state["t0"])
        rate = state["n"] / elapsed
        label = "planning" if phase == "abc" else "generating"
        # total=0 → indeterminate; stems pass an estimate (25 tok/s audio)
        # so the bar and ETA become real instead of a bare token count.
        eta = max(0, int((total - state["n"]) / rate)) if total else None
        progress(item, state["n"], total, f"{label} · {rate:.1f} tok/s", eta=eta)

    return on_token


def default_sessions_root() -> Path:
    root = Path.home() / "Library" / "Application Support" / "YuE2 Studio" / "Sessions"
    return root


#: Model-weight repos the engine needs. Downloaded once into the HF cache
#: (or wherever --models-dir / HF_HOME points, e.g. an external SSD).
MODEL_REPOS = {"ar": "m-a-p/YuE2-3B", "vae": "m-a-p/YuE2-Vae"}

_expected_cache: dict = {}


def hub_cache_dir() -> Path:
    """Where HF weights live: HF_HUB_CACHE, else HF_HOME/hub, else default."""
    import os
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    base = Path(os.environ.get("HF_HOME",
                              Path.home() / ".cache/huggingface")).expanduser()
    return base / "hub"


def _repo_dir(repo: str) -> Path:
    return hub_cache_dir() / ("models--" + repo.replace("/", "--"))


def _blob_bytes(repo_dir: Path) -> int:
    """Bytes in the blob store — includes .incomplete so progress moves
    during a file download, not just per-file."""
    blobs = repo_dir / "blobs"
    if not blobs.is_dir():
        return 0
    return sum(p.stat().st_size for p in blobs.rglob("*") if p.is_file())


def _expected_bytes(repo: str, allow_network: bool = False) -> int | None:
    """Total weight size per HF metadata; cached (None when unresolved).

    allow_network=False on the request path: model_info() is a real
    HTTPS call that can stall a handler for tens of seconds under a
    slow/filtered network — and it runs inside snapshot() → broadcast(),
    so one hang wedges every start_job behind self.lock. The download
    job and the startup warmer resolve it in the background instead."""
    if repo in _expected_cache:
        return _expected_cache[repo]
    if not allow_network:
        return None
    total = None
    try:
        from huggingface_hub import HfApi
        # files_metadata → real byte sizes; safetensors.total is a
        # *parameter count*, not bytes.
        info = HfApi().model_info(repo, files_metadata=True, timeout=10)
        total = sum(s.size or 0 for s in info.siblings) or None
    except Exception:
        pass
    _expected_cache[repo] = total
    return total


def _safetensors_errors(repo_dir: Path) -> list:
    """Weight files whose headers can't be parsed — a killed or corrupted
    download can leave blobs that pass the size check but die ~seconds into
    model load. Header reads are milliseconds; tensor data is not touched."""
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return []
    try:
        from safetensors import safe_open
    except ImportError:
        return []
    bad = []
    for path in snapshots.rglob("*.safetensors"):
        try:
            with safe_open(str(path.resolve()), framework="pt"):
                pass
        except Exception:
            bad.append(path.name)
    return bad


def model_status() -> dict:
    """Per-repo download state for the setup sheet / settings panel."""
    out = {"cache_dir": str(hub_cache_dir()), "complete": True, "models": {}}
    for key, repo in MODEL_REPOS.items():
        repo_dir = _repo_dir(repo)
        blobs = repo_dir / "blobs"
        size = _blob_bytes(repo_dir)
        incomplete = blobs.is_dir() and any(
            p.name.endswith(".incomplete") for p in blobs.rglob("*"))
        snapshots = repo_dir / "snapshots"
        has_weights = snapshots.is_dir() and any(
            snapshots.rglob("*.safetensors"))
        corrupt = _safetensors_errors(repo_dir) \
            if has_weights and not incomplete else []
        expected = _expected_bytes(repo)
        done = has_weights and not incomplete and not corrupt and \
            (expected is None or size >= expected * 0.99)
        out["models"][key] = {"repo": repo, "downloaded": done,
                              "bytes": size, "expected": expected,
                              "corrupt": corrupt}
        out["complete"] = out["complete"] and done
    return out


def _download_models(progress, cancelled):
    """Job fn: snapshot-download every required repo, polling blob size.

    Each repo downloads in a child process — snapshot_download can't be
    interrupted cooperatively, but a process can be killed. That's what
    makes Cancel actually stop a multi-GB download instead of waiting
    for the current repo to finish."""
    for repo in MODEL_REPOS.values():
        if cancelled():
            raise InterruptedError()
        # This job owns the network — resolve the real total so progress
        # is honest, and warm the cache for every later request-path call.
        expected = _expected_bytes(repo, allow_network=True)
        stop = threading.Event()

        def _watch(repo=repo, expected=expected):
            while not stop.wait(1.0):
                progress(repo.split("/")[-1],
                         _blob_bytes(_repo_dir(repo)), expected or 0,
                         "downloading")
        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
        code = ("from huggingface_hub import snapshot_download;"
                f"snapshot_download({repo!r})")
        proc = subprocess.Popen([sys.executable, "-c", code])
        try:
            while proc.poll() is None:
                if cancelled():
                    proc.kill()
                    raise InterruptedError()
                stop.wait(0.5)
        finally:
            stop.set()
            watcher.join(timeout=2)
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if proc.returncode:
            raise RuntimeError(
                f"model download failed ({repo}, exit {proc.returncode})")
    return model_status()


class Job:
    """One background job at a time; status broadcast over SSE."""

    def __init__(self, kind: str, fn, notify=None):
        self.kind = kind
        self.fn = fn
        self.notify = notify or (lambda: None)
        self.cancelled = threading.Event()
        self.directory = None
        self.status = {"job": kind, "stage": "queued", "detail": {},
                       "started": time.time(), "error": None}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.done = threading.Event()

    def _run(self):
        try:
            # Don't overwrite "cancelling" if cancel beat the thread start.
            if self.cancelled.is_set():
                self.status["stage"] = "cancelling"
            else:
                self.status["stage"] = "running"
            self.notify()
            self.result = self.fn(self.progress, self.cancelled.is_set)
            # Work that returns early on cancel (without raising) is still
            # a cancelled job — never report it as done.
            self.status["stage"] = ("cancelled" if self.cancelled.is_set()
                                    else "done")
        except InterruptedError:
            self.status["stage"] = "cancelled"
        except Exception as error:  # surfaced to clients, never swallowed
            self.status["stage"] = "error"
            self.status["error"] = f"{type(error).__name__}: {error}"
        finally:
            self.done.set()
            self.notify()

    def progress(self, name, index, total, stage="", eta=None):
        detail = {"item": name, "index": index, "total": total, "stage": stage}
        if eta is not None:
            detail["eta"] = eta
        self.status["detail"] = detail
        self.notify()


class StudioState:
    """Session registry + current-job bookkeeping."""

    def __init__(self, sessions_root, engine_factory=None,
                 min_free_gib: float = MIN_FREE_GIB,
                 require_weights: bool = True):
        self.root = Path(sessions_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine_factory = engine_factory
        self.min_free_gib = min_free_gib
        self.require_weights = require_weights
        self._engine = None
        self._engine_lock = threading.Lock()
        self._session_dir: Path | None = None
        self.job: Job | None = None
        self.lock = threading.Lock()
        self._ready_lock = threading.Lock()
        self.listeners: list[queue.Queue] = []
        self._restore_open_session()

    @property
    def session_dir(self):
        return self._session_dir

    @session_dir.setter
    def session_dir(self, directory):
        # Persist the open-session pointer — an engine restart (watchdog,
        # crash, manual) must come back with the same project open, not a
        # dead "no session" state while the files sit on disk.
        self._session_dir = directory
        marker = self.root / ".current"
        try:
            if directory is None:
                marker.unlink(missing_ok=True)
            else:
                marker.write_text(Path(directory).name)
        except OSError:
            pass

    def _restore_open_session(self):
        marker = self.root / ".current"
        try:
            name = marker.read_text().strip()
        except OSError:
            return
        if not name:
            return
        candidate = (self.root / name).resolve()
        if candidate.is_dir() and candidate.is_relative_to(self.root.resolve()):
            self._session_dir = candidate

    @property
    def engine(self):
        # Locked — the warmup thread and a job must never build two engines.
        with self._engine_lock:
            if self._engine is None:
                if self.engine_factory is None:
                    raise RuntimeError("No engine configured — pass --model or set YUE2_MODEL")
                self._engine = self.engine_factory()
            return self._engine

    def broadcast(self):
        payload = self.snapshot()
        for listener in list(self.listeners):
            try:
                listener.put_nowait(payload)
            except queue.Full:
                pass

    def snapshot(self) -> dict:
        # A session deleted outside the API (Finder, rm) must not keep
        # reporting itself — the directory is the source of truth.
        if self.session_dir is not None and not self.session_dir.is_dir():
            self.session_dir = None
        session = None
        if self.session_dir:
            session = {"dir": str(self.session_dir), "id": self.session_dir.name}
            try:
                name = json.loads(
                    (self.session_dir / "request.json").read_text()).get("name")
                if name:
                    session["name"] = name
            except (OSError, ValueError):
                pass
            arrangement_path = self.session_dir / "arrangement.json"
            if arrangement_path.is_file():
                leaves = expand_leaves(load_arrangement(arrangement_path))
                stems_dir = self.session_dir / "stems"
                session["leaves"] = [
                    {"name": leaf["name"], "track": leaf.get("track"),
                     "done": _leaf_done_fast(stems_dir, leaf["name"])}
                    for leaf in leaves]
            # Which deliverables exist on disk — the UI disables play/download
            # for missing files instead of letting clicks 404.
            session["outputs"] = {
                "master": (self.session_dir / "master.flac").is_file(),
                "binaural": (self.session_dir / "immersive" / "binaural.flac").is_file(),
                "midi": (self.session_dir / "midi" / "song.mid").is_file(),
            }
        compute = None
        if self._engine is not None:
            device = getattr(self._engine, "device", None)
            compute = {"device": str(device) if device is not None else None,
                       "quantization": getattr(self._engine, "quantization", "none")}
        return {"session": session,
                "job": self.job.status if self.job else None,
                "mem_free_gib": round(free_memory_gib() or 0, 1),
                "compute": compute,
                "problems": self._readiness()}

    def _readiness(self):
        """Cached readiness check — model_status() walks the HF cache on
        disk, too expensive to run on every broadcast. Non-blocking: a
        concurrent caller serves the stale value rather than queueing a
        request handler behind another thread's disk scan."""
        now = time.monotonic()
        cached = getattr(self, "_ready_cache", None)
        if cached is not None and now - cached[0] <= 5:
            return cached[1]
        if not self._ready_lock.acquire(blocking=False):
            return cached[1] if cached else []
        try:
            cached = getattr(self, "_ready_cache", None)
            if cached is not None and now - cached[0] <= 5:
                return cached[1]
            try:
                problems = readiness_problems(self, self.session_dir)
            except Exception:
                problems = []
            self._ready_cache = (now, problems)
            return problems
        finally:
            self._ready_lock.release()

    def sessions(self) -> list[dict]:
        out = []
        for child in sorted(self.root.iterdir()):
            if child.is_dir() and (child / "request.json").is_file():
                entry = {"id": child.name, "dir": str(child)}
                try:
                    name = json.loads(
                        (child / "request.json").read_text()).get("name")
                    if name:
                        entry["name"] = name
                except (OSError, ValueError):
                    pass
                out.append(entry)
        return out

    def open_session(self, session_id: str):
        candidate = (self.root / session_id).resolve()
        if not candidate.is_dir() or not candidate.is_relative_to(self.root.resolve()):
            raise ValueError(f"Unknown session {session_id!r}")
        self.session_dir = candidate
        return candidate

    def clear_terminal_job(self):
        """Drop a finished/cancelled/errored job from the status — its
        result was already seen; it must not haunt the UI forever."""
        if self.job and self.job.done.is_set():
            self.job = None

    def start_job(self, kind: str, fn, directory=None) -> Job:
        with self.lock:
            if self.job and not self.job.done.is_set():
                # A terminal stage may still be settling — done.set() runs a
                # breath after the stage flips to cancelled/error. Give a
                # finishing job a moment rather than rejecting the user's
                # next action with a stale "already running". A genuinely
                # running job is rejected immediately.
                if self.job.status.get("stage") in {"cancelled", "error", "done"}:
                    self.job.done.wait(timeout=2.0)
            if self.job and not self.job.done.is_set():
                stage = self.job.status.get("stage", "running")
                raise RuntimeError(
                    f"A job is already running ({stage}) — "
                    + ("it is still stopping, try again in a moment"
                       if stage == "cancelling" else "cancel it first"))
            self.job = Job(kind, fn, notify=self.broadcast)
            self.job.directory = directory  # session dir the job owns
            self.job.thread.start()
            job = self.job
        # Broadcast outside the lock — snapshot() does file I/O per leaf,
        # and holding the lock across it serializes unrelated handlers
        # behind disk work.
        self.broadcast()
        return job


def _json_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    if not length:
        return {}
    return json.loads(handler.rfile.read(length))


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", str(name)).strip("-_").lower()
    return re.sub(r"-{2,}", "-", slug)[:40]


def make_handler(state: StudioState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "YuE2Studio/0.1"

        def log_message(self, *args):
            pass

        # ── helpers ─────────────────────────────────────────────────────
        def _send(self, code: int, body: dict | list | None = None,
                  content_type="application/json", raw=None):
            data = raw if raw is not None else (
                json.dumps(body).encode() if body is not None else b"")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, error: Exception, code=400):
            self._send(code, {"error": f"{type(error).__name__}: {error}"})

        def _session(self) -> Path:
            if state.session_dir is None:
                raise ValueError("No session open")
            return state.session_dir

        def _resolve(self, rel: str) -> Path:
            root = self._session().resolve()
            path = (root / rel).resolve()
            if not path.is_file() or not path.is_relative_to(root):
                raise ValueError(f"Not a session file: {rel!r}")
            return path

        def _add_track(self):
            """Append catalog instrument / ensemble / kit tracks to the open
            session's arrangement and rebuild the session leaf list."""
            body = _json_body(self)
            path = self._session() / "arrangement.json"
            arrangement = load_arrangement(path)
            tracks = arrangement.setdefault("tracks", [])
            names = {t["name"] for t in tracks}

            def unique(name: str) -> str:
                base, i = name, 2
                while name in names:
                    name = f"{base}_{i}"
                    i += 1
                names.add(name)
                return name

            if body.get("instrument"):
                instrument = body["instrument"]
                if instrument not in INSTRUMENT_CATALOG:
                    raise ValueError(f"Unknown instrument {instrument!r}")
                tracks.append(make_track(instrument, name=unique(instrument)))
            elif body.get("kit"):
                kit = body["kit"]
                if kit not in KIT_PRESETS:
                    raise ValueError(f"Unknown kit {kit!r}")
                tracks.append(make_kit(kit, name=unique(f"drums_{kit}")))
            elif body.get("ensemble"):
                preset = ENSEMBLE_PRESETS.get(body["ensemble"])
                if preset is None:
                    raise ValueError(f"Unknown ensemble {body['ensemble']!r}")
                for instrument in preset:
                    tracks.append(make_track(instrument,
                                             name=unique(instrument)))
            else:
                raise ValueError("track add needs instrument|kit|ensemble")
            save_arrangement(path, arrangement)
            save_session(self._session(), default_session(arrangement))
            state.broadcast()
            self._send(200, arrangement)

        # ── routing ─────────────────────────────────────────────────────
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/health":
                    self._send(200, {"ok": True, "engine": state._engine is not None})
                elif parsed.path == "/api/status":
                    if "text/event-stream" in (self.headers.get("Accept") or ""):
                        self._sse()
                    else:
                        self._send(200, state.snapshot())
                elif parsed.path == "/api/sessions":
                    self._send(200, state.sessions())
                elif parsed.path == "/api/session":
                    self._send(200, state.snapshot()["session"] or {})
                elif parsed.path == "/api/arrangement":
                    self._send(200, load_arrangement(self._session() / "arrangement.json"))
                elif parsed.path == "/api/catalog":
                    self._send(200, {
                        "instruments": {name: {"family": spec["family"],
                                               "line": spec["line"]}
                                        for name, spec in
                                        INSTRUMENT_CATALOG.items()},
                        "ensembles": ENSEMBLE_PRESETS,
                        "kits": KIT_PRESETS})
                elif parsed.path == "/api/mix-state":
                    self._send(200, json.loads(
                        (self._session() / "session.json").read_text()))
                elif parsed.path == "/api/formats":
                    self._send(200, {"audio": available_audio_formats(),
                                     "sample_rates": [48000, 44100],
                                     "midi_types": [0, 1]})
                elif parsed.path == "/api/models":
                    self._send(200, model_status())
                elif parsed.path == "/api/ready":
                    problems = readiness_problems(state, state.session_dir)
                    self._send(200, {"ready": not problems,
                                     "problems": problems})
                elif parsed.path.startswith("/api/files/"):
                    rel = parsed.path[len("/api/files/"):]
                    path = self._resolve(rel)
                    # A real content type — AVPlayer refuses octet-stream.
                    content_type = {
                        ".flac": "audio/flac", ".wav": "audio/wav",
                        ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
                        ".mid": "audio/midi", ".midi": "audio/midi",
                        ".json": "application/json", ".abc": "text/plain",
                        ".md": "text/markdown", ".txt": "text/plain",
                    }.get(path.suffix.lower(), "application/octet-stream")
                    self._send(200, content_type=content_type,
                               raw=path.read_bytes())
                else:
                    self._send(404, {"error": "not found"})
            except Exception as error:
                self._error(error)

        def _sse(self):
            listener = queue.Queue(maxsize=64)
            state.listeners.append(listener)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                self.wfile.write(b"data: " + json.dumps(state.snapshot()).encode() + b"\n\n")
                self.wfile.flush()
                while True:
                    try:
                        payload = listener.get(timeout=15)
                    except queue.Empty:
                        payload = state.snapshot()
                    self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                state.listeners.remove(listener)

        def do_PUT(self):
            try:
                if self.path == "/api/arrangement":
                    arrangement = _json_body(self)
                    save_arrangement(self._session() / "arrangement.json", arrangement)
                    session = default_session(arrangement)
                    save_session(self._session(), session)
                    state.broadcast()
                    self._send(200, {"ok": True})
                elif self.path == "/api/arrangement/tracks":
                    self._add_track()
                elif self.path == "/api/mix-state":
                    body = _json_body(self)
                    for key in ("stems", "buses", "fx_sends", "master"):
                        if key not in body:
                            raise ValueError(f"mix state missing {key!r}")
                    write_json(self._session() / "session.json", body)
                    state.broadcast()
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {"error": "not found"})
            except Exception as error:
                self._error(error)

        def do_POST(self):
            try:
                if self.path == "/api/session":
                    self._create_session()
                elif self.path == "/api/open":
                    # One-session rule: no switching while a job is active.
                    if state.job and not state.job.done.is_set():
                        raise RuntimeError(
                            "a job is running — wait for it to finish or "
                            "cancel it before switching sessions")
                    state.open_session(_json_body(self)["id"])
                    state.clear_terminal_job()
                    state.broadcast()
                    self._send(200, state.snapshot())
                elif self.path == "/api/delete":
                    self._delete()
                elif self.path == "/api/dismiss":
                    # User dismissed the result — a terminal job must not
                    # keep reporting itself in every snapshot.
                    state.clear_terminal_job()
                    state.broadcast()
                    self._send(200, {"ok": True})
                elif self.path == "/api/render":
                    self._render()
                elif self.path == "/api/mix":
                    self._mix()
                elif self.path == "/api/models/download":
                    job = state.start_job("models", _download_models)
                    self._send(202, job.status)
                elif self.path == "/api/export":
                    self._export()
                elif self.path == "/api/cancel":
                    if state.job and not state.job.done.is_set():
                        state.job.cancelled.set()
                        state.job.status["stage"] = "cancelling"
                        state.broadcast()
                        self._send(200, {"ok": True})
                    else:
                        # Honest answer — a Cancel click that hits nothing
                        # must not look like it did something.
                        self._send(400, {"error": "no job is running"})
                else:
                    self._send(404, {"error": "not found"})
            except Exception as error:
                self._error(error)

        # ── endpoints ───────────────────────────────────────────────────
        def _create_session(self):
            body = _json_body(self)
            request = {"style": body["style"], "lyrics": body.get("lyrics", ""),
                       "cot": body.get("cot", "full"),
                       "seed": int(body.get("seed", 831001))}
            if body.get("ode_steps"):
                request["ode_steps"] = int(body["ode_steps"])
            if body.get("ode_method"):
                request["ode_method"] = str(body["ode_method"])
            stamp = time.strftime("%Y%m%d-%H%M%S")
            slug = _slug(body.get("name", ""))
            session_id = f"{slug}-{stamp}" if slug else f"{stamp}-{identity(request)[:8]}"
            directory = state.root / session_id
            directory.mkdir(parents=True, exist_ok=True)
            request["name"] = slug or session_id
            write_json(directory / "request.json", request)

            def work(progress, cancelled):
                score = None
                if body.get("abc"):
                    score_text = body["abc"]
                else:
                    check_ready(state, directory)
                    progress("engine", 1, 1, _engine_label(state))
                    score_text, score = _plan_and_parse(
                        state, request, progress, cancelled)
                if cancelled():
                    raise InterruptedError("cancelled")
                progress("score", 0, 1, "writing score")
                (directory / "score.abc").write_text(score_text)
                if score is None:
                    score = parse(score_text)
                progress("orchestration", 0, 1, "building orchestration")
                arrangement = (body.get("arrangement")
                               or (arrangement_from_picks(body["instruments"], request)
                                   if body.get("instruments") else None)
                               or derive_arrangement(score, request))
                save_arrangement(directory / "arrangement.json", arrangement)
                save_session(directory, default_session(arrangement))
                state.broadcast()  # leaves appear in the UI immediately
                progress("session", 1, 1, "ready")
                return {"session": session_id}

            previous = state.session_dir
            state.session_dir = directory
            state.clear_terminal_job()
            if body.get("abc") or state.engine_factory:
                try:
                    state.start_job("session", work, directory=directory)
                except RuntimeError as error:
                    if "already running" not in str(error):
                        # Genuine failure — don't leave an orphaned dir.
                        state.session_dir = previous
                        shutil.rmtree(directory, ignore_errors=True)
                        raise
                    # Engine busy on another job: keep the session spec —
                    # the first Render plans it (same as a parked session).
                    self._send(202, {"session": session_id, "dir": str(directory),
                                     "pending": "score"})
                else:
                    self._send(202, {"session": session_id, "dir": str(directory)})
            else:
                # No engine and no provided score: park the session empty.
                self._send(202, {"session": session_id, "dir": str(directory),
                                 "pending": "score"})

        def _delete(self):
            session_id = _json_body(self)["id"]
            candidate = (state.root / session_id).resolve()
            if not candidate.is_dir() or not candidate.is_relative_to(state.root.resolve()):
                raise ValueError(f"Unknown session {session_id!r}")
            if (state.job and not state.job.done.is_set()
                    and getattr(state.job, "directory", None) is not None
                    and Path(state.job.directory).resolve() == candidate):
                raise RuntimeError("a job is still using this session — "
                                   "cancel it first")
            if state.session_dir and state.session_dir.resolve() == candidate:
                state.session_dir = None
            shutil.rmtree(candidate)
            state.clear_terminal_job()
            state.broadcast()
            self._send(200, {"ok": True})

        def _render(self):
            body = _json_body(self)
            directory = self._session()
            stems = StemSession(directory)

            def work(progress, cancelled):
                request = json.loads((directory / "request.json").read_text())
                arrangement_path = directory / "arrangement.json"
                score_path = directory / "score.abc"
                score = None
                if not score_path.is_file():
                    # Parked/incomplete session — plan the score now instead
                    # of forcing the user to delete and recreate.
                    score_text, score = _plan_and_parse(
                        state, request, progress, cancelled)
                    progress("score", 0, 1, "writing score")
                    score_path.write_text(score_text)
                    if cancelled():
                        raise InterruptedError("cancelled")
                if not arrangement_path.is_file():
                    score = score or parse(score_path.read_text())
                    progress("orchestration", 0, 1, "building orchestration")
                    arrangement = derive_arrangement(score, request)
                    save_arrangement(arrangement_path, arrangement)
                    save_session(directory, default_session(arrangement))
                    state.broadcast()
                    if cancelled():
                        raise InterruptedError("cancelled")
                else:
                    arrangement = load_arrangement(arrangement_path)
                check_ready(state, directory)
                progress("engine", 1, 1, _engine_label(state))
                engine = state.engine
                if cancelled():
                    raise InterruptedError("cancelled")
                return stems.render(engine, request, arrangement,
                                    score_path.read_text(),
                                    tracks=body.get("tracks"), cancelled=cancelled,
                                    on_token=_token_progress(progress),
                                    on_progress=lambda n, i, t, s: progress(n, i, t, s),
                                    cleanup=body.get("cleanup", "role-eq"))

            state.start_job("render", work, directory=directory)
            self._send(202, {"ok": True})

        def _mix(self):
            directory = self._session()

            def work(progress, cancelled):
                if cancelled():
                    raise InterruptedError("cancelled")
                return render_session(directory, on_progress=lambda n, i, t, s: progress(n, i, t, s))

            state.start_job("mix", work, directory=directory)
            self._send(202, {"ok": True})

        def _export(self):
            directory = self._session()
            body = _json_body(self)
            zip_path = deliver(directory,
                               bundle=body.get("bundle"),
                               files=body.get("files"),
                               audio_format=body.get("audio_format", "flac-24"),
                               sample_rate=body.get("sample_rate", 48000),
                               midi_type=body.get("midi_type", 1),
                               out=directory / "exports" /
                               f"deliver_{body.get('bundle') or 'custom'}.zip")
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition",
                             f'attachment; filename="{zip_path.name}"')
            data = zip_path.read_bytes()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


class _DeadPipeSafe:
    """Stream proxy: a closed stdout/stderr pipe (app restarted or quit while
    the engine was mid-write) drops the write instead of raising
    BrokenPipeError into an unrelated render job."""
    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        try:
            return self._stream.write(data)
        except (OSError, ValueError):
            return len(data)

    def flush(self):
        try:
            self._stream.flush()
        except (OSError, ValueError):
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _warmup_engine(state: StudioState):
    """Load the model at startup — silently, in the background — so the
    first render doesn't pay the load cost. Skipped when weights are
    missing (the setup flow owns that state) or in test/fake-engine mode."""
    try:
        # Resolve expected sizes here — the one place a network call can
        # never stall a request handler.
        for repo in MODEL_REPOS.values():
            _expected_bytes(repo, allow_network=True)
        if not state.require_weights or not model_status()["complete"]:
            return
        warmup = getattr(state.engine, "warmup", None)
        if warmup is not None:
            warmup()
        state.broadcast()
    except Exception as error:
        print(f"[YuE2] Background model warmup failed: {error} — "
              "it will retry on first render", file=sys.stderr)


def serve(port: int = DEFAULT_PORT, sessions_root=None, engine_factory=None,
          min_free_gib: float = MIN_FREE_GIB):
    """Run the local API server (blocking)."""
    from .._platform import require_apple_silicon
    require_apple_silicon("YuE2 Studio")
    sys.stdout = _DeadPipeSafe(sys.stdout)
    sys.stderr = _DeadPipeSafe(sys.stderr)
    state = StudioState(sessions_root or default_sessions_root(), engine_factory,
                        min_free_gib=min_free_gib)
    if engine_factory is not None:
        threading.Thread(target=_warmup_engine, args=(state,),
                         daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    # The launcher parses this marker to learn the bound port — with
    # --port 0 the OS assigns a free port, so a stale engine from a dead
    # app instance can never shadow a new one.
    print(f"YUE2_PORT={server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
