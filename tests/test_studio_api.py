"""Studio local API: session, arrangement, render, mix, export, cancel."""
import json
import sys
import threading
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yue2.studio.server import StudioState, make_handler  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCORE_TEXT = (EXAMPLES / "score.abc").read_text()


class FakePlan:
    abc = SCORE_TEXT


class FakeResult:
    def __init__(self):
        self.sample_rate = 48000
        t = np.arange(48000) / 48000
        self.audio = np.stack([0.2 * np.sin(2 * np.pi * 440 * t)] * 2,
                              axis=1).astype(np.float32)
        self.request_identity = "fake" + "0" * 60
        self.truncated = {"abc": False, "semantic": False}

    def save_artifacts(self, directory):
        import soundfile as sf
        from yue2.storage import collect_hashes, write_json
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sf.write(directory / "audio.flac", self.audio, self.sample_rate,
                 subtype="PCM_24")
        for name in ("prefix.npy", "semantic.npy", "latent.npy",
                     "abc_tokens.npy"):
            np.save(directory / name, np.zeros(4, dtype=np.int32))
        write_json(directory / "request.json",
                   {"style": "x", "lyrics": "y"})
        write_json(directory / "config.json", {})
        write_json(directory / "result.json",
                   {"status": "complete", "identity": self.request_identity,
                    "truncated": self.truncated, "sample_rate": 48000,
                    "audio_seconds": 1.0, "weights": {}, "timing": {},
                    "artifacts": collect_hashes(directory)})
        return {"identity": self.request_identity}


class FakeEngine:
    def __init__(self):
        self.calls = 0

    def plan(self, **kwargs):
        return FakePlan()

    def __call__(self, **kwargs):
        self.calls += 1
        return FakeResult()


@pytest.fixture()
def server(tmp_path):
    state = StudioState(tmp_path / "sessions", engine_factory=lambda: FakeEngine(),
                        min_free_gib=0, require_weights=False)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", state
    httpd.shutdown()


def _req(base, path, method="GET", body=None, raw=False):
    request = urllib.request.Request(
        base + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = response.read()
        return response.status, (payload if raw else json.loads(payload or b"{}"))


def _wait_done(base):
    for _ in range(100):
        _, status = _req(base, "/api/status")
        if not status.get("job") or status["job"]["stage"] in {
                "done", "error", "cancelled"}:
            return status["job"]
        import time
        time.sleep(0.05)
    raise TimeoutError("job never finished")


def test_health(server):
    base, _ = server
    code, body = _req(base, "/api/health")
    assert code == 200 and body["ok"]


def test_expected_bytes_no_network_on_request_path(monkeypatch):
    """model_status() runs inside snapshot() → broadcast() → every request.
    A live HfApi call there once stalled POST /api/session past the client
    timeout under a filtered network. Cache-miss must return None, never
    dial out; only the download job / startup warmer may (allow_network)."""
    from yue2.studio import server as srv
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("model_info on request path")))
    srv._expected_cache.clear()
    assert srv._expected_bytes("m-a-p/YuE2-3B") is None
    srv.model_status()  # snapshot path — must not raise via model_info
    # The sanctioned path resolves then caches — here it fails closed to
    # None (the monkeypatched call raises inside the try/except).
    assert srv._expected_bytes("m-a-p/YuE2-3B", allow_network=True) is None
    assert "m-a-p/YuE2-3B" in srv._expected_cache
    srv._expected_cache.clear()


def test_readiness_serves_stale_instead_of_blocking(tmp_path):
    """A concurrent _readiness computation must not queue a request thread
    behind another thread's disk scan — serve the stale cache."""
    from yue2.studio import server as srv
    state = srv.StudioState(tmp_path / "s", min_free_gib=0,
                            require_weights=False)
    state._ready_cache = (0.0, ["stale"])  # expired
    state._ready_lock.acquire()            # another thread is computing
    try:
        assert state._readiness() == ["stale"]
    finally:
        state._ready_lock.release()


def test_session_create_never_stalls_on_readiness(tmp_path, monkeypatch):
    """The real reported failure: POST /api/session timed out because
    snapshot→readiness→model_status made a live HF call inside self.lock.
    Run the REAL readiness path (require_weights=True — real model_status,
    real safetensors-header scan, cold _expected_cache) with the network
    poisoned — the POST must still answer immediately."""
    from yue2.studio import server as srv
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("network on request path")))
    srv._expected_cache.clear()
    state = srv.StudioState(tmp_path / "sessions",
                            engine_factory=lambda: FakeEngine(),
                            min_free_gib=0, require_weights=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        import time
        t0 = time.time()
        code, _ = _req(base, "/api/session", "POST",
                       {"style": "pop", "lyrics": "la", "abc": SCORE_TEXT})
        assert code == 202
        assert time.time() - t0 < 9.0  # _req itself times out at 10
        _wait_done(base)
    finally:
        httpd.shutdown()
        srv._expected_cache.clear()


def test_session_flow(server):
    base, _ = server
    code, body = _req(base, "/api/session", "POST",
                      {"style": "pop, piano", "lyrics": "la", "abc": SCORE_TEXT})
    assert code == 202
    job = _wait_done(base)
    assert job["stage"] == "done", job
    _, arrangement = _req(base, "/api/arrangement")
    assert arrangement["tracks"]
    _, snap = _req(base, "/api/status")
    assert snap["session"]["leaves"]


def test_arrangement_put(server):
    base, _ = server
    _req(base, "/api/session", "POST",
         {"style": "pop", "lyrics": "la", "abc": SCORE_TEXT})
    _wait_done(base)
    _, arrangement = _req(base, "/api/arrangement")
    arrangement["tracks"] = arrangement["tracks"][:1]
    code, _ = _req(base, "/api/arrangement", "PUT", arrangement)
    assert code == 200


def test_render_and_mix(server):
    base, state = server
    _req(base, "/api/session", "POST",
         {"style": "pop", "lyrics": "la", "abc": SCORE_TEXT})
    _wait_done(base)
    _, arrangement = _req(base, "/api/arrangement")
    arrangement["tracks"] = arrangement["tracks"][:2]
    _req(base, "/api/arrangement", "PUT", arrangement)
    code, _ = _req(base, "/api/render", "POST", {})
    assert code == 202
    job = _wait_done(base)
    assert job["stage"] == "done", job
    code, _ = _req(base, "/api/mix", "POST", {})
    job = _wait_done(base)
    assert job["stage"] == "done", job
    session_dir = state.session_dir
    assert (session_dir / "master.flac").is_file()


def test_export_zip(server):
    base, _ = server
    _req(base, "/api/session", "POST",
         {"style": "pop", "lyrics": "la", "abc": SCORE_TEXT})
    _wait_done(base)
    _, arrangement = _req(base, "/api/arrangement")
    arrangement["tracks"] = arrangement["tracks"][:1]
    _req(base, "/api/arrangement", "PUT", arrangement)
    _req(base, "/api/render", "POST", {})
    _wait_done(base)
    _req(base, "/api/mix", "POST", {})
    _wait_done(base)
    code, payload = _req(base, "/api/export", "POST",
                         {"bundle": "everything"}, raw=True)
    assert code == 200
    names = set(zipfile.ZipFile(BytesIO(payload)).namelist())
    assert "MANIFEST.json" in names and "master.flac" in names


def test_file_download_and_traversal(server):
    base, _ = server
    _req(base, "/api/session", "POST",
         {"style": "pop", "lyrics": "la", "abc": SCORE_TEXT})
    _wait_done(base)
    code, data = _req(base, "/api/files/score.abc", raw=True)
    assert code == 200 and data == SCORE_TEXT.encode()
    with pytest.raises(urllib.error.HTTPError):
        _req(base, "/api/files/../secret.txt")


def test_one_job_at_a_time(server):
    base, _ = server
    _req(base, "/api/session", "POST",
         {"style": "pop", "lyrics": "la", "abc": SCORE_TEXT})
    _wait_done(base)
    code, _ = _req(base, "/api/render", "POST", {})
    assert code == 202
    with pytest.raises(urllib.error.HTTPError) as err:
        _req(base, "/api/mix", "POST", {})
    assert err.value.code == 400
    _wait_done(base)


def test_session_create_while_job_running_parks_pending(server):
    """Create during a busy engine must NOT error + orphan a session dir —
    the spec is kept and planning defers to the first Render."""
    base, state = server
    gate = threading.Event()

    def hold(progress, cancelled):
        gate.wait(10)
        return {}

    state.start_job("render", hold)
    try:
        code, body = _req(base, "/api/session", "POST",
                          {"style": "pop", "lyrics": "la", "name": "busy"})
        assert code == 202 and body.get("pending") == "score"
        dirs = [p for p in state.root.iterdir() if p.is_dir()]
        assert len(dirs) == 1                       # kept, not rolled back
        _, snap = _req(base, "/api/status")
        assert snap["session"]["id"] == dirs[0].name
    finally:
        gate.set()
    _wait_done(base)
    # The parked session plans + renders once the engine is free.
    code, _ = _req(base, "/api/render", "POST", {})
    assert code == 202
    job = _wait_done(base)
    assert job["stage"] == "done", job
    _, snap = _req(base, "/api/status")
    assert snap["session"]["leaves"]


def test_open_session_refused_while_job_running(server):
    """One-session rule: no switching mid-job."""
    base, state = server
    _req(base, "/api/session", "POST",
         {"style": "pop", "lyrics": "la", "abc": SCORE_TEXT})
    _wait_done(base)
    _req(base, "/api/session", "POST",
         {"style": "jazz", "lyrics": "da", "abc": SCORE_TEXT})
    _wait_done(base)
    gate = threading.Event()

    def hold(progress, cancelled):
        gate.wait(10)
        return {}

    state.start_job("render", hold)
    try:
        _, snap = _req(base, "/api/status")
        other = next(s["id"] for s in _req(base, "/api/sessions")[1]
                     if s["id"] != snap["session"]["id"])
        with pytest.raises(urllib.error.HTTPError) as err:
            _req(base, "/api/open", "POST", {"id": other})
        assert err.value.code == 400
        # Still on the original session.
        _, snap = _req(base, "/api/status")
        assert snap["session"]["id"] != other
    finally:
        gate.set()


def test_malformed_plan_replans_until_valid(server):
    """One malformed ABC from the model must not kill the session — the
    planner retries with a nudged seed until the score parses."""
    base, state = server
    calls = {"n": 0}

    class FlakyPlan:
        def __init__(self, abc):
            self.abc = abc

    def flaky_plan(**kwargs):
        calls["n"] += 1
        return FlakyPlan("X:1\nT:\ngarbage\n" if calls["n"] == 1 else SCORE_TEXT)

    state.engine.plan = flaky_plan
    code, _ = _req(base, "/api/session", "POST",
                   {"style": "pop", "lyrics": "la"})
    assert code == 202
    job = _wait_done(base)
    assert job["stage"] == "done", job
    assert calls["n"] == 2
    _, snap = _req(base, "/api/status")
    assert snap["session"]["leaves"]


def test_unparseable_plan_fails_after_bounded_retries(server):
    """A plan that never parses must fail honestly after the retry cap,
    not loop forever."""
    base, state = server
    calls = {"n": 0}

    class BadPlan:
        abc = "X:1\nT:\ngarbage\n"

    def bad_plan(**kwargs):
        calls["n"] += 1
        return BadPlan()

    state.engine.plan = bad_plan
    code, _ = _req(base, "/api/session", "POST",
                   {"style": "pop", "lyrics": "la"})
    assert code == 202
    job = _wait_done(base)
    assert job["stage"] == "error", job
    assert "ScoreError" in job["error"]
    assert calls["n"] == 3


def test_sessions_listing(server):
    base, _ = server
    _req(base, "/api/session", "POST",
         {"style": "pop", "lyrics": "la", "abc": SCORE_TEXT})
    _wait_done(base)
    code, sessions = _req(base, "/api/sessions")
    assert code == 200 and sessions


def test_needed_free_gib_tracks_engine_state(tmp_path):
    """The memory gate must follow the engine's real load state, not a
    static floor — this is what blocked renders at 3 GiB free."""
    from types import SimpleNamespace
    from yue2.studio.server import (needed_free_gib, engine_warm,
                                    engine_loading, WARM_MIN_FREE_GIB,
                                    INT8_LOAD_MIN_FREE_GIB)
    state = StudioState(tmp_path / "sessions", min_free_gib=4.0,
                        require_weights=False)
    assert needed_free_gib(state) == 4.0            # no engine → full floor
    engine = SimpleNamespace(_model=None, _load_lock=threading.Lock(),
                             quantization="int8", device="mps")
    state._engine = engine
    assert needed_free_gib(state) == INT8_LOAD_MIN_FREE_GIB  # cold int8 load
    engine._load_lock.acquire()
    assert engine_loading(state) and not engine_warm(state)
    assert needed_free_gib(state) == WARM_MIN_FREE_GIB       # wait, don't refuse
    engine._load_lock.release()
    engine._model = object()
    assert engine_warm(state) and not engine_loading(state)
    assert needed_free_gib(state) == WARM_MIN_FREE_GIB       # resident: marginal only


def test_open_session_marker_survives_restart(tmp_path):
    """session_dir must be on disk — a new StudioState (engine restart)
    restores the same open project instead of 'No session open'."""
    state = StudioState(tmp_path / "sessions", min_free_gib=0,
                        require_weights=False)
    session = state.root / "abc-20260101"
    session.mkdir()
    (session / "request.json").write_text("{}")
    state.session_dir = session
    assert (state.root / ".current").read_text() == session.name
    restored = StudioState(tmp_path / "sessions", min_free_gib=0,
                           require_weights=False)
    assert restored.session_dir == session
    # Deleting/clearing the session clears the marker too.
    restored.session_dir = None
    again = StudioState(tmp_path / "sessions", min_free_gib=0,
                        require_weights=False)
    assert again.session_dir is None
    # A vanished directory restores nothing instead of zombie state.
    state.session_dir = session
    (session / "request.json").unlink()
    session.rmdir()
    gone = StudioState(tmp_path / "sessions", min_free_gib=0,
                       require_weights=False)
    assert gone.session_dir is None


def test_dead_pipe_safe_drops_writes():
    import os
    from yue2.studio.server import _DeadPipeSafe

    read_fd, write_fd = os.pipe()
    os.close(read_fd)  # reader gone — writes would raise BrokenPipeError
    stream = _DeadPipeSafe(os.fdopen(write_fd, "w", closefd=True))
    assert stream.write("progress text") == len("progress text")
    stream.flush()  # must not raise
    assert stream.isatty() in (True, False)  # attr passthrough works
