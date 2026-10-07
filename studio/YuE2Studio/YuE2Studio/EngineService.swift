import Foundation
import Combine

/// Owns the Python engine process and the local HTTP/SSE API.
@MainActor
final class EngineService: ObservableObject {
    @Published var connected = false
    @Published var status = StatusSnapshot(session: nil, job: nil) {
        didSet { syncMixer() }
    }
    /// Global transport — one synchronized multi-stem player for the whole
    /// app; the mixer strips and the bottom transport bar both drive it.
    let mixer = MixerEngine()
    @Published var catalog: Catalog?
    @Published var sessions: [SessionInfo] = []
    @Published var arrangement: Arrangement?
    @Published var showingNewSession = false
    @Published var showingModelSetup = false
    @Published var modelStatus: ModelStatus?
    @Published var engineLog: String = ""
    @Published var lastError: String?
    private var errorClearTask: Task<Void, Never>?

    /// Surface an error, then auto-dismiss it — a transient failure must
    /// not haunt the banner forever; a recurring problem re-reports itself.
    func reportError(_ message: String) {
        lastError = message
        errorClearTask?.cancel()
        errorClearTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(12))
            guard !Task.isCancelled, let self, self.lastError == message else { return }
            self.lastError = nil
        }
    }

    /// Assigned when the engine announces its bound port (YUE2_PORT=…).
    /// Starts at 0 so health checks can't accidentally reach a stale
    /// engine from a dead app instance holding the old fixed port.
    var port = 0

    /// One-session rule: while any engine job is active, creating a new
    /// session and switching sessions are both locked. The UI disables the
    /// controls; these guards cover direct calls.
    var jobActive: Bool {
        ["queued", "running", "cancelling"].contains(status.job?.stage ?? "")
    }

    var executableURL: URL?  // user-configured `yue2` binary
    private var process: Process?
    private var sseTask: Task<Void, Never>?
    private var pollTask: Task<Void, Never>?
    private var base: URL { URL(string: "http://127.0.0.1:\(port)")! }

    /// First existing `yue2` install, or nil to fall back to /usr/bin/env lookup.
    private func resolvedExecutable() -> (url: URL, args: [String]) {
        if let executableURL { return (executableURL, []) }
        // Self-contained distribution: the engine bundled inside the .app.
        if let bundled = Bundle.main.resourceURL?
            .appendingPathComponent("engine/bin/yue2"),
           FileManager.default.isExecutableFile(atPath: bundled.path) {
            return (bundled, [])
        }
        if let env = ProcessInfo.processInfo.environment["YUE2"], !env.isEmpty {
            return (URL(fileURLWithPath: env), [])
        }
        let home = NSHomeDirectory()
        for path in ["/opt/homebrew/bin/yue2",
                     "/usr/local/bin/yue2",
                     "\(home)/.local/bin/yue2",
                     "\(home)/Library/Python/3.14/bin/yue2",
                     "\(home)/Library/Python/3.13/bin/yue2"]
        where FileManager.default.fileExists(atPath: path) {
            return (URL(fileURLWithPath: path), [])
        }
        return (URL(fileURLWithPath: "/usr/bin/env"), ["yue2"])
    }

    func start() {
        guard process == nil else { return }
        // Orphaned engines from a dead app instance hold ~6 GiB of model —
        // they've reparented to launchd (ppid 1), which is provably
        // abandoned. A user's intentional CLI engine has a shell parent
        // and is never matched.
        let reap = Process()
        reap.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
        reap.arguments = ["-P", "1", "-f", "studio --serve --port"]
        try? reap.run()
        reap.waitUntilExit()
        let proc = Process()
        let (url, prefix) = resolvedExecutable()
        proc.executableURL = url
        let defaults = UserDefaults.standard
        // Engine memory budget scaled to physical RAM — the default (24 GiB)
        // is fantasy on a small Mac; ~half of RAM, capped by RAM-8, leaves
        // the system breathing room (16GB→8, 64GB→32).
        let gib = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        let budget = defaults.object(forKey: "engine.budget") != nil
            ? defaults.double(forKey: "engine.budget")
            : Double(Int(max(6, min(gib - 8, gib * 0.5))))
        let device = defaults.string(forKey: "engine.device") ?? "auto"
        let threads = defaults.object(forKey: "engine.threads") != nil
            ? defaults.integer(forKey: "engine.threads") : 6
        let minFree = defaults.object(forKey: "engine.minfree") != nil
            ? defaults.double(forKey: "engine.minfree") : 4.0
        // --port 0: the OS assigns a free port; the engine prints
        // YUE2_PORT=<n> and the pipe parser below adopts it. No fixed
        // port = no EADDRINUSE and no zombie-engine takeover, ever.
        var arguments = prefix + ["studio", "--serve", "--port", "0",
                                  "--budget", "\(Int(budget))",
                                  "--device", device,
                                  "--min-free", "\(minFree)"]
        // Low-memory mode: AR weights offload to CPU between stages —
        // slower, but a tight machine can actually render.
        if defaults.bool(forKey: "engine.lowmem") {
            arguments += ["--offload-ar"]
        }
        // Optional external model cache (e.g. a fast NVMe/SSD volume).
        if let dir = defaults.string(forKey: "models.dir"), !dir.isEmpty {
            arguments += ["--models-dir", dir]
        }
        proc.arguments = arguments
        // Hard boundaries so generation can't freeze the Mac:
        // - OMP thread cap keeps cores free for the rest of the system
        // - No MPS watermark cap: torch's default (1.7x recommended) lets the
        //   allocator over-commit into swap. A hard cap made tight machines
        //   OOM on load — swap is slower, but it runs.
        var env = ProcessInfo.processInfo.environment
        env["OMP_NUM_THREADS"] = "\(threads)"
        env.removeValue(forKey: "PYTORCH_MPS_HIGH_WATERMARK_RATIO")
        // Any op MPS can't run natively falls back to CPU instead of crashing.
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        proc.environment = env
        let output = Pipe()
        proc.standardOutput = output
        proc.standardError = output
        // The engine writes progress bars to stderr — the pipe MUST be drained
        // continuously or it fills (16KB) and the engine deadlocks mid-render.
        output.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8)
            else { return }
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.engineLog = String((self.engineLog + text).suffix(50_000))
                // Scan the accumulated log — the marker can split across
                // pipe reads, and "YUE2_PORT=" is 10 chars, not 9.
                if self.port == 0,
                   let range = self.engineLog.range(
                       of: #"YUE2_PORT=(\d+)"#, options: .regularExpression),
                   let bound = Int(self.engineLog[range].dropFirst(10)) {
                    self.port = bound
                }
            }
        }
        try? proc.run()
        process = proc
        Task { await waitForEngine() }
    }

    func stop() {
        guard let proc = process else { return }
        proc.terminate()
        process = nil
        port = 0  // the next engine announces its own port
        sseTask?.cancel()
        pollTask?.cancel()
        connected = false
        // SIGTERM only lands when Python returns to the interpreter loop —
        // during long native calls (model load) that can take many seconds.
        // Retaining `proc` keeps the output pipe's read end open so late
        // writes don't hit EPIPE, then SIGKILL settles it for real.
        DispatchQueue.global().asyncAfter(deadline: .now() + 3) {
            if proc.isRunning {
                kill(proc.processIdentifier, SIGKILL)
            }
        }
    }

    /// Restart with the current Settings values (device/budget/threads).
    func restart() {
        stop()
        engineLog = ""
        start()
    }

    private func waitForEngine() async {
        for _ in 0..<60 {
            if await health() {
                connected = true
                await refreshSessions()
                await fetchModels()
                listen()
                startPolling()
                return
            }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        // Engine never came up — say so instead of sitting "offline" forever.
        let tail = engineLog.split(separator: "\n").suffix(3)
            .joined(separator: " ")
        reportError("Engine failed to start"
                    + (tail.isEmpty ? "" : " — \(tail)"))
    }

    // ── HTTP helpers ────────────────────────────────────────────────────

    private func request(_ path: String, method: String = "GET",
                         body: Encodable? = nil,
                         timeout: TimeInterval = 30) async throws -> (Data, HTTPURLResponse) {
        var req = URLRequest(url: base.appendingPathComponent(path))
        req.httpMethod = method
        req.timeoutInterval = timeout
        if let body {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONEncoder().encode(AnyEncodable(body))
        }
        let (data, response) = try await URLSession.shared.data(for: req)
        return (data, response as! HTTPURLResponse)
    }

    func health() async -> Bool {
        guard port > 0 else { return false }  // engine hasn't announced yet
        guard let (_, response) = try? await request("/api/health", timeout: 3)
        else { return false }
        return response.statusCode == 200
    }

    func refreshSessions() async {
        guard let (data, r) = try? await request("/api/sessions"), r.statusCode == 200,
              let list = try? JSONDecoder().decode([SessionInfo].self, from: data) else { return }
        sessions = list
    }

    /// One-shot status pull — SSE can lag or drop, so views that gate on
    /// job state (e.g. the download button) refresh explicitly.
    func refreshStatus() async {
        guard let (data, r) = try? await request("/api/status", timeout: 8),
              r.statusCode == 200,
              let snap = try? JSONDecoder().decode(StatusSnapshot.self, from: data)
        else { return }
        applyStatus(snap)
    }

    /// Write a snapshot only when something the UI shows actually changed.
    /// The 2.5s poll fires constantly and mem_free_gib jitters — without this
    /// gate every view re-renders each poll and open menus/popovers dismiss.
    private var lastStatusKey: Int = 0
    private func applyStatus(_ snap: StatusSnapshot) {
        var hasher = Hasher()
        snap.session.hash(into: &hasher)
        snap.job.hash(into: &hasher)
        snap.compute.hash(into: &hasher)
        snap.problems.hash(into: &hasher)
        // Memory is shown at 0.1 GiB precision — ignore sub-display jitter.
        Int((snap.mem_free_gib ?? 0) * 10).hash(into: &hasher)
        let key = hasher.finalize()
        guard key != lastStatusKey else { return }
        lastStatusKey = key
        status = snap
    }

    func createSession(name: String, style: String, lyrics: String,
                       cot: String, seed: Int,
                       instruments: [String] = [], odeSteps: Int? = nil,
                       odeMethod: String? = nil) async {
        guard !jobActive else {
            reportError("A session is generating — finish or cancel it first")
            return
        }
        struct Body: Encodable {
            var name: String; var style: String
            var lyrics: String; var cot: String; var seed: Int
            var instruments: [String]?; var ode_steps: Int?
            var ode_method: String?
        }
        lastError = nil
        await post("/api/session", body: Body(
            name: name, style: style, lyrics: lyrics, cot: cot, seed: seed,
            instruments: instruments.isEmpty ? nil : instruments,
            ode_steps: odeSteps, ode_method: odeMethod))
        // Reflect the new session immediately — don't wait for SSE/poll.
        await refreshStatus()
        await refreshSessions()
    }

    func openSession(id: String) async {
        guard !jobActive else { return }  // one session at a time while generating
        struct Body: Encodable { var id: String }
        lastError = nil
        do {
            let (data, r) = try await request("/api/open", method: "POST", body: Body(id: id))
            if r.statusCode == 200,
               let snap = try? JSONDecoder().decode(StatusSnapshot.self, from: data) {
                applyStatus(snap)
            } else {
                reportError(serverError(data, r))
            }
        } catch {
            reportError(error.localizedDescription)
        }
        await fetchArrangement()
    }

    func deleteSession(id: String) async {
        struct Body: Encodable { var id: String }
        await post("/api/delete", body: Body(id: id))
        await refreshSessions()
        await refreshStatus()  // the deleted session may have been open —
                               // don't wait for SSE to clear the view
    }

    func fetchArrangement() async {
        guard let (data, r) = try? await request("/api/arrangement"), r.statusCode == 200,
              let decoded = try? JSONDecoder().decode(Arrangement.self, from: data) else { return }
        arrangement = decoded
    }

    func saveArrangement() async {
        guard let arrangement else { return }
        _ = try? await request("/api/arrangement", method: "PUT", body: arrangement)
    }

    func fetchCatalog() async {
        guard catalog == nil,
              let (data, r) = try? await request("/api/catalog"), r.statusCode == 200,
              let decoded = try? JSONDecoder().decode(Catalog.self, from: data) else { return }
        catalog = decoded
    }

    /// Add a catalog instrument, ensemble preset, or drum kit track.
    /// The server expands presets and rebuilds the leaf list.
    func addTrack(instrument: String? = nil, ensemble: String? = nil,
                  kit: String? = nil) async {
        struct Body: Encodable {
            var instrument: String?; var ensemble: String?; var kit: String?
        }
        do {
            let (data, r) = try await request("/api/arrangement/tracks",
                method: "PUT",
                body: Body(instrument: instrument, ensemble: ensemble, kit: kit))
            if r.statusCode == 200,
               let decoded = try? JSONDecoder().decode(Arrangement.self, from: data) {
                arrangement = decoded
                lastError = nil
            } else {
                reportError(serverError(data, r))
            }
        } catch {
            reportError(error.localizedDescription)
        }
    }

    /// Mutate one arrangement track then persist — the leaf list and the
    /// render queue rebuild from the saved arrangement.
    func updateTrack(_ name: String,
                     _ mutate: (inout Arrangement.Track) -> Void) async {
        guard let index = arrangement?.tracks.firstIndex(where: { $0.name == name })
        else { return }
        mutate(&arrangement!.tracks[index])
        await saveArrangement()
    }

    func removeTrack(_ name: String) async {
        arrangement?.tracks.removeAll { $0.name == name }
        await saveArrangement()
    }

    // ── mixer sync ──────────────────────────────────────────────────────

    private var lastSyncedKey = ""
    private var lastMixSession = ""

    /// Keep the global transport stocked with the session's finished stems —
    /// fires on every status snapshot (SSE + poll), loads only what's new.
    private func syncMixer() {
        let done = (status.session?.leaves ?? [])
            .filter { $0.done ?? false }.map(\.name).sorted()
        let key = (status.session?.id ?? "") + "|" + done.joined(separator: ",")
        guard key != lastSyncedKey else { return }
        lastSyncedKey = key
        let session = status.session?.id ?? ""
        Task {
            await mixer.load(names: done, session: session) { [self] rel in
                try await rawRequest("/api/files/\(rel)").0
            }
            // Persisted strip settings apply once per session so the
            // transport plays the saved mix even before the Mixer tab opens.
            if session != lastMixSession {
                lastMixSession = session
                await applySavedMix()
            }
        }
    }

    private func applySavedMix() async {
        guard let (data, r) = try? await rawRequest("/api/mix-state"),
              r.statusCode == 200,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let stems = json["stems"] as? [String: [String: Any]] else { return }
        for (name, s) in stems {
            mixer.set(name, .init(
                gainDb: s["gain_db"] as? Double ?? 0,
                pan: s["pan"] as? Double ?? 0,
                mute: s["mute"] as? Bool ?? false,
                solo: s["solo"] as? Bool ?? false))
        }
    }

    /// POST that surfaces failures into `lastError` instead of swallowing.
    /// A successful action clears a stale error — the user moved on.
    private func post(_ path: String, body: Encodable? = nil) async {
        do {
            let (data, response) = try await request(path, method: "POST", body: body)
            if response.statusCode >= 400 {
                reportError(serverError(data, response))
            } else {
                lastError = nil
            }
        } catch {
            reportError(error.localizedDescription)
        }
    }

    /// Dismiss a finished job's status — clears the server-side job record
    /// so terminal errors/cancelled states stop appearing in snapshots.
    func dismissJob() async {
        lastError = nil
        await post("/api/dismiss")
        await refreshStatus()
    }

    /// Pull the {"error": "..."} message out of an error response body.
    private func serverError(_ data: Data, _ response: HTTPURLResponse) -> String {
        if let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let message = dict["error"] as? String {
            return message
        }
        return String(data: data, encoding: .utf8) ?? "HTTP \(response.statusCode)"
    }

    func render(tracks: [String]? = nil, cleanup: String = "none") async {
        struct Body: Encodable { var tracks: [String]?; var cleanup: String }
        lastError = nil
        await post("/api/render", body: Body(tracks: tracks, cleanup: cleanup))
    }

    func mix() async {
        lastError = nil
        await post("/api/mix")
    }

    func cancel() async {
        // Bypass post() — a "no job is running" 400 is a stale-snapshot
        // race (the job ended between polls), not an error worth a banner.
        var noJob = false
        do {
            let (data, response) = try await request("/api/cancel", method: "POST",
                                                     timeout: 6)
            if response.statusCode >= 400 {
                let message = serverError(data, response)
                noJob = message.contains("no job")
                if !noJob { reportError(message) }
            }
        } catch {
            // A timed-out cancel means the engine is too busy to answer —
            // exactly the case the force-restart watchdog exists for.
            reportError("Cancel request failed — \(error.localizedDescription)")
        }
        // Cooperative cancel has blind spots — a blocking native call
        // (model load, VAE decode) can't observe the flag, and Python
        // can't kill a thread mid-C-call. If the job is still alive after
        // a grace period, kill the engine outright: the stuck work dies
        // with the process and the engine reboots clean. The watchdog arms
        // unless the server explicitly said there's nothing to cancel —
        // its poll loop exits itself once the job leaves an active stage.
        guard !noJob else { return }
        let activeStages = ["queued", "running", "cancelling"]
        Task { [weak self] in
            for _ in 0..<4 {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                await self?.refreshStatus()
                guard let current = self?.status.job?.stage,
                      activeStages.contains(current) else { return }
            }
            self?.engineLog += "\n[app] Job ignored cancel — restarting engine\n"
            self?.restart()
        }
    }

    func export(bundle: String?, files: [String]?, audioFormat: String,
                sampleRate: Int, midiType: Int, to destination: URL) async throws {
        struct Body: Encodable {
            var bundle: String?; var files: [String]?
            var audio_format: String; var sample_rate: Int; var midi_type: Int
        }
        let (data, response) = try await request(
            "/api/export", method: "POST",
            body: Body(bundle: bundle, files: files,
                       audio_format: audioFormat, sample_rate: sampleRate,
                       midi_type: midiType))
        guard response.statusCode == 200 else {
            let detail = String(data: data, encoding: .utf8) ?? "HTTP \(response.statusCode)"
            throw NSError(domain: "YuE2Studio", code: response.statusCode,
                          userInfo: [NSLocalizedDescriptionKey: detail])
        }
        try data.write(to: destination)
    }

    // ── Model weights ───────────────────────────────────────────────────

    func fetchModels() async {
        guard let (data, r) = try? await request("/api/models"),
              r.statusCode == 200,
              let s = try? JSONDecoder().decode(ModelStatus.self, from: data)
        else { return }
        modelStatus = s
        // First-run: prompt setup when weights are missing.
        if !s.complete { showingModelSetup = true }
    }

    func downloadModels() async {
        lastError = nil
        do {
            let (data, r) = try await request("/api/models/download",
                                              method: "POST")
            if r.statusCode == 202,
               let job = try? JSONDecoder().decode(JobStatus.self, from: data) {
                // Lock immediately from the response — don't wait on SSE.
                status.job = job
            } else if r.statusCode >= 400 {
                reportError(serverError(data, r))
            }
        } catch {
            reportError(error.localizedDescription)
        }
    }

    func fetchFormats() async -> Formats? {
        guard let (data, r) = try? await request("/api/formats"), r.statusCode == 200 else { return nil }
        return try? JSONDecoder().decode(Formats.self, from: data)
    }

    func fileURL(_ rel: String) -> URL {
        base.appendingPathComponent("/api/files/\(rel)")
    }

    func rawRequest(_ path: String, method: String = "GET",
                    body: Data? = nil) async throws -> (Data, HTTPURLResponse) {
        var req = URLRequest(url: base.appendingPathComponent(path))
        req.httpMethod = method
        if let body {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = body
        }
        let (data, response) = try await URLSession.shared.data(for: req)
        return (data, response as! HTTPURLResponse)
    }

    func saveMixState(_ json: [String: Any]) async {
        guard let data = try? JSONSerialization.data(withJSONObject: json) else { return }
        _ = try? await rawRequest("/api/mix-state", method: "PUT", body: data)
    }

    // ── SSE ─────────────────────────────────────────────────────────────

    /// Status safety net — SSE is the fast path, but if an event drops or
    /// the stream silently dies the UI would freeze on stale state forever.
    /// A cheap 2.5s poll guarantees the snapshot always converges.
    private func startPolling() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 2_500_000_000)
                await self?.refreshStatus()
            }
        }
    }

    private func listen() {
        sseTask?.cancel()
        sseTask = Task {
            // Reconnect forever — a dropped stream must not leave the UI blind.
            while !Task.isCancelled {
                var req = URLRequest(url: base.appendingPathComponent("/api/status"))
                req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                var buffer = ""
                do {
                    let (bytes, _) = try await URLSession.shared.bytes(for: req)
                    for try await line in bytes.lines {
                        if Task.isCancelled { return }
                        if line.hasPrefix("data: ") {
                            buffer += String(line.dropFirst(6))
                        } else if line.isEmpty, !buffer.isEmpty {
                            if let data = buffer.data(using: .utf8),
                               let snap = try? JSONDecoder().decode(
                                   StatusSnapshot.self, from: data) {
                                applyStatus(snap)
                                // Refresh model status when the download
                                // job finishes (done, error or cancelled).
                                if snap.job?.job == "models",
                                   ["done", "error", "cancelled"]
                                       .contains(snap.job?.stage ?? "") {
                                    Task { await fetchModels() }
                                }
                            }
                            buffer = ""
                        }
                    }
                } catch { /* connection closed — fall through to retry */ }
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }
}

private struct AnyEncodable: Encodable {
    private let encodeClosure: (Encoder) throws -> Void
    init(_ wrapped: Encodable) { encodeClosure = wrapped.encode }
    func encode(to encoder: Encoder) throws { try encodeClosure(encoder) }
}
