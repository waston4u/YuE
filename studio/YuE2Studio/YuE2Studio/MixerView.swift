import SwiftUI

/// Channel strips for every leaf stem + bus strips + master.
struct MixerView: View {
    @EnvironmentObject var engine: EngineService
    @StateObject private var player = PreviewPlayer()
    private var mixer: MixerEngine { engine.mixer }
    @State private var mixState: [String: StemState] = [:]
    @State private var fullState: [String: Any] = [:]
    @State private var masterGain: Double = 0

    struct StemState: Codable {
        var gain_db: Double = 0
        var pan: Double = 0
        var mute: Bool = false
        var solo: Bool = false
    }

    private var leaves: [StatusSnapshot.Leaf] {
        engine.status.session?.leaves ?? []
    }

    var body: some View {
        Group {
            if leaves.isEmpty {
                ContentUnavailableView("No stems to mix",
                    systemImage: "slider.vertical.3",
                    description: Text("Render a session first."))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 0) {
                    ScrollView(.horizontal) {
                        HStack(alignment: .bottom, spacing: 10) {
                            ForEach(leaves, id: \.name) { leaf in
                                ChannelStrip(name: leaf.name, state: binding(for: leaf.name),
                                             done: leaf.done ?? false,
                                             playing: mixer.auditioning == leaf.name,
                                             onPlay: { mixer.audition(leaf.name) },
                                             onCommit: commit,
                                             metering: engine.status.job?.stage == "running")
                                    .transition(.move(edge: .bottom).combined(with: .opacity))
                            }
                            Divider()
                            MasterStrip(gain: $masterGain,
                                        masterReady: engine.status.session?.outputs?.master ?? false,
                                        playing: player.playing == "master",
                                        onPlay: { play("master", rel: "master.flac") },
                                        onCommit: commit,
                                        metering: engine.status.job?.stage == "running")
                        }
                        .padding()
                    }
                }
            }
        }
        .onAppear { loadState() }
        // A different session is a different project — drop the old mix
        // state and load the new session's settings, never show stale strips.
        .onChange(of: engine.status.session?.id) {
            player.stop()
            mixState = [:]
            fullState = [:]
            masterGain = 0
            loadState()
        }
    }

    private func fmt(_ s: Double) -> String {
        String(format: "%d:%04.1f", Int(s) / 60, s.truncatingRemainder(dividingBy: 60))
    }

    private func play(_ name: String, rel: String? = nil) {
        player.play(url: engine.fileURL(rel ?? "stems/\(name)/audio.flac"), key: name)
    }

    private func binding(for leaf: String) -> Binding<StemState> {
        Binding(
            get: { mixState[leaf] ?? StemState() },
            set: { new in
                mixState[leaf] = new
                // Live-audition: the strip IS the mix, not a save-then-render.
                mixer.set(leaf, .init(gainDb: new.gain_db, pan: new.pan,
                                      mute: new.mute, solo: new.solo))
            })
    }

    private func commit() {
        var master = fullState["master"] as? [String: Any] ?? [:]
        master["gain_db"] = masterGain
        fullState["master"] = master
        pushState()
    }

    private func loadState() {
        Task {
            guard let (data, r) = try? await engine.rawRequest("/api/mix-state"),
                  r.statusCode == 200,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let stems = json["stems"] as? [String: [String: Any]] else { return }
            fullState = json
            masterGain = (json["master"] as? [String: Any])?["gain_db"] as? Double ?? 0
            for (name, state) in stems {
                mixState[name] = StemState(
                    gain_db: state["gain_db"] as? Double ?? 0,
                    pan: state["pan"] as? Double ?? 0,
                    mute: state["mute"] as? Bool ?? false,
                    solo: state["solo"] as? Bool ?? false)
            }
            for (name, s) in mixState {
                mixer.set(name, .init(gainDb: s.gain_db, pan: s.pan,
                                      mute: s.mute, solo: s.solo))
            }
        }
    }

    private func pushState() {
        var stems = fullState["stems"] as? [String: [String: Any]] ?? [:]
        for (name, state) in mixState {
            var entry = stems[name] ?? [:]
            entry["gain_db"] = state.gain_db
            entry["pan"] = state.pan
            entry["mute"] = state.mute
            entry["solo"] = state.solo
            stems[name] = entry
        }
        fullState["stems"] = stems
        Task { await engine.saveMixState(fullState) }
    }
}

struct ChannelStrip: View {
    let name: String
    @Binding var state: MixerView.StemState
    var done = false
    var playing = false
    var onPlay: () -> Void = {}
    var onCommit: () -> Void = {}
    var metering = false

    private var color: Color { Studio.familyColor(name) }

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 4) {
                Text(name.components(separatedBy: ".").last ?? name)
                    .font(.caption2.weight(.semibold))
                    .lineLimit(1)
                Button(action: onPlay) {
                    Image(systemName: playing ? "pause.fill" : "play.fill")
                        .font(.caption2)
                        .foregroundStyle(playing ? Studio.accent : .secondary)
                }
                .buttonStyle(.borderless)
                .disabled(!done)
                .help(done ? (playing ? "Stop stem preview" : "Play stem") : "Render this stem first")
            }
            .frame(width: 72)

            RotaryKnob(value: $state.pan, tint: color, onCommit: onCommit)
            Text(state.pan == 0 ? "C"
                 : "\(state.pan < 0 ? "L" : "R")\(Int(abs(state.pan) * 100))")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)

            HStack(alignment: .bottom, spacing: 6) {
                FaderView(value: $state.gain_db, tint: color, onCommit: onCommit)
                LiveMeter(active: metering && !state.mute,
                          seed: Double(abs(name.hashValue % 100)) / 10.0,
                          segments: 14, height: 132)
            }

            Text("\(state.gain_db, specifier: "%+.1f") dB")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)

            HStack(spacing: 4) {
                Toggle("M", isOn: $state.mute).toggleStyle(.button)
                    .controlSize(.mini).tint(.orange)
                    .onChange(of: state.mute) { onCommit() }
                Toggle("S", isOn: $state.solo).toggleStyle(.button)
                    .controlSize(.mini).tint(.yellow)
                    .onChange(of: state.solo) { onCommit() }
            }
        }
        .padding(8)
        .opacity(state.mute ? 0.5 : 1)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(state.solo ? color.opacity(0.8) : .white.opacity(0.06),
                              lineWidth: state.solo ? 1.5 : 1)
        )
        .animation(.spring(duration: 0.2), value: state.mute)
        .animation(.spring(duration: 0.2), value: state.solo)
    }
}

struct MasterStrip: View {
    @Binding var gain: Double
    var masterReady = false
    var playing = false
    var onPlay: () -> Void = {}
    var onCommit: () -> Void = {}
    var metering = false

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 4) {
                Text("MASTER")
                    .font(.caption2.weight(.bold))
                    .tracking(1.5)
                    .foregroundStyle(Studio.accent)
                Button(action: onPlay) {
                    Image(systemName: playing ? "stop.fill" : "play.fill")
                        .font(.caption2)
                        .foregroundStyle(playing ? Studio.accent : .secondary)
                }
                .buttonStyle(.borderless)
                .disabled(!masterReady)
                .help(masterReady ? "Play the stereo master" : "Mix the session first")
            }
            .frame(width: 72)

            Spacer().frame(height: 24)  // aligns with the pan-knob row

            HStack(alignment: .bottom, spacing: 6) {
                FaderView(value: $gain, tint: Studio.accent, onCommit: onCommit)
                HStack(alignment: .bottom, spacing: 4) {
                    LiveMeter(active: metering, seed: 0, height: 132)
                    LiveMeter(active: metering, seed: 0.6, height: 132)
                }
            }

            Text("\(gain, specifier: "%+.1f") dB")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
            Text("-14 LUFS · ceiling -1")
                .font(.system(size: 8).monospacedDigit())
                .foregroundStyle(.tertiary)
        }
        .padding(8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Studio.accent.opacity(0.35), lineWidth: 1)
        )
    }
}

/// Global transport — one synchronized playhead across every finished stem.
/// Lives at the bottom of the window, visible from every tab.
struct TransportBar: View {
    @EnvironmentObject var engine: EngineService
    private var mixer: MixerEngine { engine.mixer }

    var body: some View {
        HStack(spacing: 12) {
            Button { mixer.stop() } label: {
                Image(systemName: "stop.fill")
            }
            .buttonStyle(.borderless).controlSize(.large)
            .disabled(mixer.loaded.isEmpty)

            Button { mixer.playing ? mixer.pause() : mixer.play() } label: {
                Image(systemName: mixer.playing ? "pause.fill" : "play.fill")
            }
            .buttonStyle(.borderless).controlSize(.large)
            .keyboardShortcut(.space, modifiers: [])
            .disabled(mixer.loaded.isEmpty)

            Slider(value: Binding(
                get: { mixer.position },
                set: { mixer.seek(to: $0) }),
                   in: 0...max(mixer.duration, 0.01))
            .disabled(mixer.loaded.isEmpty)

            Text("\(fmt(mixer.position)) / \(fmt(mixer.duration))")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 110, alignment: .trailing)

            if mixer.loading {
                ProgressView().controlSize(.small)
            } else if mixer.loaded.isEmpty {
                Text("No stems rendered")
                    .font(.caption2).foregroundStyle(.tertiary)
            } else {
                Text("\(mixer.loaded.count) stems")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 8)
        .background(.bar)
    }

    private func fmt(_ s: Double) -> String {
        String(format: "%d:%04.1f", Int(s) / 60, s.truncatingRemainder(dividingBy: 60))
    }
}
