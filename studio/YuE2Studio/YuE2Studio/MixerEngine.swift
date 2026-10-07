import AVFoundation
import AudioToolbox

/// Synchronized multi-stem playback — the actual mixer. One AVAudioPlayerNode
/// per leaf stem into the main mixer; gain/pan/mute/solo apply live so the
/// strips audition exactly what "Mix Session" will render. Stems download
/// through the local engine API into a temp cache (AVAudioFile needs files).
@MainActor
final class MixerEngine: ObservableObject {
    @Published private(set) var playing = false
    @Published private(set) var position: Double = 0
    @Published private(set) var duration: Double = 0
    @Published private(set) var loaded: Set<String> = []
    @Published private(set) var loading = false

    private let engine = AVAudioEngine()
    private var limiter: AVAudioUnit?
    private var nodes: [String: AVAudioPlayerNode] = [:]
    private var files: [String: AVAudioFile] = [:]
    private var frames: [String: AVAudioFramePosition] = [:]
    private var state: [String: ChannelState] = [:]
    private var baseFrame: AVAudioFramePosition = 0
    private var sessionKey = ""
    private var ticker: Timer?
    /// Strip-play audition: solos one stem without touching mix state.
    @Published private(set) var auditioning: String?

    struct ChannelState {
        var gainDb: Double = 0
        var pan: Double = 0
        var mute = false
        var solo = false
    }

    // ── loading ────────────────────────────────────────────────────────

    /// Fetch + attach every finished stem for the session. Safe to call
    /// repeatedly — already-loaded stems are skipped, new ones join.
    /// `fetch` maps a session-relative path to bytes (engine.rawRequest).
    func load(names: [String], session: String,
              fetch: (String) async throws -> Data) async {
        if session != sessionKey {
            await unload()
            sessionKey = session
        }
        let pending = names.filter { !files.keys.contains($0) }
        guard !pending.isEmpty else { return }
        loading = true
        defer { loading = false }
        let cache = FileManager.default.temporaryDirectory
            .appendingPathComponent("YuE2MixCache").appendingPathComponent(session)
        try? FileManager.default.createDirectory(at: cache,
                                                 withIntermediateDirectories: true)
        for name in pending {
            do {
                let data = try await fetch("stems/\(name)/audio.flac")
                let local = cache.appendingPathComponent("\(name).flac")
                try data.write(to: local, options: .atomic)
                let file = try AVAudioFile(forReading: local)
                files[name] = file
                frames[name] = file.length
                let node = AVAudioPlayerNode()
                nodes[name] = node
                engine.attach(node)
                engine.connect(node, to: engine.mainMixerNode,
                               format: file.processingFormat)
                loaded.insert(name)
            } catch {
                // A stem that vanished or is still being written just
                // doesn't join the mix — it will on the next load().
            }
        }
        duration = files.values.map { Double($0.length) / $0.processingFormat.sampleRate }.max() ?? 0
        guard !nodes.isEmpty else { return }
        routeOutput()
        insertLimiter()
        engine.mainMixerNode.outputVolume = AudioSettings.volume
        engine.prepare()
        try? engine.start()
        applyAll()
    }

    private func unload() async {
        stop()
        for node in nodes.values { engine.detach(node) }
        nodes = [:]; files = [:]; frames = [:]; loaded = []
        duration = 0
    }

    // ── transport ──────────────────────────────────────────────────────

    func play() {
        guard !nodes.isEmpty, !playing else { return }
        let startAt = futureTime(0.15)
        for (name, node) in nodes {
            guard let file = files[name], let total = frames[name],
                  baseFrame < total else { continue }
            node.stop()
            node.scheduleSegment(file, startingFrame: baseFrame,
                                 frameCount: AVAudioFrameCount(total - baseFrame),
                                 at: nil) { }
            node.play(at: startAt)
        }
        playing = true
        startTicker()
    }

    func pause() {
        guard playing else { return }
        baseFrame = currentFrame()
        for node in nodes.values { node.pause() }
        playing = false
        ticker?.invalidate()
    }

    func stop() {
        for node in nodes.values { node.stop() }
        baseFrame = 0
        position = 0
        auditioning = nil
        applyAll()
        playing = false
        ticker?.invalidate()
    }

    func seek(to seconds: Double) {
        let wasPlaying = playing
        if playing {
            for node in nodes.values { node.stop() }
            // play() refuses to run while `playing` is set — clear it so
            // the re-schedule below isn't a silent no-op.
            playing = false
            ticker?.invalidate()
        }
        let sr = files.values.first?.processingFormat.sampleRate ?? 48000
        baseFrame = AVAudioFramePosition(max(0, min(seconds, duration)) * sr)
        position = seconds
        if wasPlaying { play() }
    }

    /// Stop everything and drop loaded stems (session switch).
    func teardown() {
        for node in nodes.values { node.stop(); engine.detach(node) }
        nodes = [:]; files = [:]; frames = [:]; loaded = []
        sessionKey = ""; duration = 0; position = 0; playing = false
        auditioning = nil
        ticker?.invalidate()
        engine.stop()
    }

    // ── live channel state ─────────────────────────────────────────────

    func set(_ name: String, _ s: ChannelState) {
        state[name] = s
        apply(name)
    }

    /// Audition one stem in isolation — tap again to rejoin the mix.
    /// Pure playback override; the committed mix state never changes.
    func audition(_ name: String) {
        auditioning = auditioning == name ? nil : name
        applyAll()
        if auditioning != nil && !playing { play() }
    }

    private func applyAll() { for name in nodes.keys { apply(name) } }

    private func apply(_ name: String) {
        guard let node = nodes[name] else { return }
        let s = state[name] ?? ChannelState()
        let audible: Bool
        if let soloName = auditioning {
            audible = name == soloName
        } else {
            let anySolo = state.values.contains { $0.solo }
            audible = !s.mute && (!anySolo || s.solo)
        }
        node.volume = audible ? Float(pow(10.0, s.gainDb / 20.0)) : 0
        node.pan = Float(max(-1, min(1, s.pan)))
    }

    // ── internals ──────────────────────────────────────────────────────

    private func currentFrame() -> AVAudioFramePosition {
        // playerTime.sampleTime is ALREADY the absolute file position for
        // a file-backed player — adding baseFrame double-counts the seek
        // offset, overshoots duration, and the ticker force-stops.
        for node in nodes.values {
            if let nodeTime = node.lastRenderTime,
               let playerTime = node.playerTime(forNodeTime: nodeTime) {
                return playerTime.sampleTime
            }
        }
        return baseFrame
    }

    private func futureTime(_ ahead: Double) -> AVAudioTime? {
        guard let render = engine.outputNode.lastRenderTime else { return nil }
        let sr = engine.outputNode.outputFormat(forBus: 0).sampleRate
        return AVAudioTime(sampleTime: render.sampleTime
                           + AVAudioFramePosition(ahead * sr),
                           atRate: sr)
    }

    private func startTicker() {
        ticker?.invalidate()
        ticker = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) {
            [weak self] _ in Task { @MainActor [weak self] in
                guard let self else { return }
                let sr = self.files.values.first?.processingFormat.sampleRate ?? 48000
                self.position = Double(self.currentFrame()) / sr
                if self.duration > 0 && self.position >= self.duration {
                    self.stop()
                }
            }
        }
    }

    /// Apple's PeakLimiter between the mix sum and the output — stems
    /// each at unity gain clip the moment two or more overlap; the render
    /// path ends in a limiter too, so the live mix should behave the same.
    private func insertLimiter() {
        guard limiter == nil else { return }
        var desc = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: kAudioUnitSubType_PeakLimiter,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0, componentFlagsMask: 0)
        guard let unit = AVAudioUnitEffect(audioComponentDescription: desc) as AVAudioUnit? else { return }
        limiter = unit
        engine.attach(unit)
        engine.disconnectNodeOutput(engine.mainMixerNode)
        engine.connect(engine.mainMixerNode, to: unit, format: nil)
        engine.connect(unit, to: engine.outputNode, format: nil)
    }

    /// Route to the configured output device — the engine's output node
    /// is an AudioUnit we can point at a specific HAL device.
    private func routeOutput() {
        guard let deviceID = AudioSettings.outputDeviceID,
              let unit = engine.outputNode.audioUnit else { return }
        var id = deviceID
        AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice,
                             kAudioUnitScope_Global, 0, &id,
                             UInt32(MemoryLayout<AudioDeviceID>.size))
    }
}
