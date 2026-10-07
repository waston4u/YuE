import SwiftUI

/// Export panel: named bundles or a custom per-file checklist → ZIP.
struct DeliverablesView: View {
    @EnvironmentObject var engine: EngineService
    @State private var bundle = "everything"
    @State private var audioFormat = "flac-24"
    @State private var sampleRate = 48000
    @State private var midiType = 1
    @State private var formats: Formats?
    @State private var customFiles: Set<String> = []
    @State private var useCustom = false
    @State private var exporting = false
    @State private var lastExport: URL?

    private let bundles: [(name: String, icon: String, tint: Color)] = [
        ("everything", "shippingbox.fill", Studio.accent),
        ("trackout", "square.stack.3d.up.fill", .blue),
        ("stereo", "speaker.wave.2.fill", .cyan),
        ("binaural", "headphones", .mint),
        ("immersive", "hifispeaker.2.fill", .indigo),
        ("buses", "point.3.connected.trianglepath.dotted", .orange),
        ("fx", "wand.and.sparkles", .pink),
        ("midi", "pianokeys", .purple),
        ("session", "doc.zipper", .gray),
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // ── bundle grid ──────────────────────────────────────
                Text("What to export").font(.headline)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 104), spacing: 8)],
                          spacing: 8) {
                    ForEach(bundles, id: \.name) { b in
                        BundleTile(name: b.name, icon: b.icon, tint: b.tint,
                                   selected: !useCustom && bundle == b.name)
                        .onTapGesture {
                            withAnimation(.spring(duration: 0.25)) {
                                bundle = b.name; useCustom = false
                            }
                        }
                    }
                    BundleTile(name: "custom pick", icon: "checklist", tint: .green,
                               selected: useCustom)
                    .onTapGesture {
                        withAnimation(.spring(duration: 0.25)) { useCustom = true }
                    }
                }

                if useCustom {
                    FileChecklist(selection: $customFiles)
                        .frame(minHeight: 180)
                        .transition(.opacity.combined(with: .move(edge: .top)))
                }

                // ── formats ──────────────────────────────────────────
                Text("Formats").font(.headline)
                HStack(spacing: 16) {
                    Picker("Audio", selection: $audioFormat) {
                        ForEach(formats?.audio ?? ["flac-24"], id: \.self) {
                            Text($0).tag($0)
                        }
                    }
                    Picker("Rate", selection: $sampleRate) {
                        ForEach(formats?.sample_rates ?? [48000], id: \.self) {
                            Text("\($0) Hz").tag($0)
                        }
                    }
                    Picker("MIDI", selection: $midiType) {
                        Text("Type 1").tag(1)
                        Text("Type 0").tag(0)
                    }
                    Spacer()
                }
                .pickerStyle(.menu)

                // ── export ───────────────────────────────────────────
                HStack(spacing: 12) {
                    Button { export() } label: {
                        Label(exporting ? "Exporting…" : "Export ZIP",
                              systemImage: "arrow.down.doc.fill")
                    }
                    .buttonStyle(PrimaryGradientButton())
                    .disabled(exporting || (useCustom && customFiles.isEmpty))
                    if exporting {
                        ActivityMeter(color: Studio.accent)
                            .frame(width: 80)
                            .transition(.opacity)
                    }
                    if let lastExport {
                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([lastExport])
                        } label: {
                            Label(lastExport.lastPathComponent, systemImage: "folder")
                        }
                        .buttonStyle(.bordered)
                    }
                    Spacer()
                }

                // ── previews ─────────────────────────────────────────
                Text("Previews").font(.headline)
                Card {
                    VStack(spacing: 4) {
                        PreviewRow(label: "Stereo master", icon: "speaker.wave.2.fill",
                                   rel: "master.flac",
                                   ready: engine.status.session?.outputs?.master ?? false)
                        PreviewRow(label: "Binaural", icon: "headphones",
                                   rel: "immersive/binaural.flac",
                                   ready: engine.status.session?.outputs?.binaural ?? false)
                        PreviewRow(label: "MIDI", icon: "pianokeys",
                                   rel: "midi/song.mid", isAudio: false,
                                   ready: engine.status.session?.outputs?.midi ?? false)
                    }
                }
            }
            .padding(16)
            .padding(.bottom, 40)
        }
        .onAppear { Task { formats = await engine.fetchFormats() } }
        // Session switch = different project — drop stale export state.
        .onChange(of: engine.status.session?.id) {
            lastExport = nil
            customFiles = []
        }
        .animation(.spring(duration: 0.3), value: useCustom)
        .animation(.spring(duration: 0.3), value: exporting)
    }

    private func export() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.zip]
        panel.nameFieldStringValue = "yue2-\(useCustom ? "custom" : bundle).zip"
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        exporting = true
        Task {
            defer { exporting = false }
            do {
                try await engine.export(
                    bundle: useCustom ? nil : bundle,
                    files: useCustom ? Array(customFiles) : nil,
                    audioFormat: audioFormat, sampleRate: sampleRate,
                    midiType: midiType, to: destination)
                lastExport = destination
            } catch {
                engine.reportError("Export failed: \(error.localizedDescription)")
            }
        }
    }
}

/// Selectable export-bundle tile.
struct BundleTile: View {
    let name: String
    let icon: String
    let tint: Color
    let selected: Bool

    var body: some View {
        VStack(spacing: 6) {
            IconTile(symbol: icon, color: tint, size: 34)
            Text(name).font(.caption.weight(.medium)).lineLimit(1)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(selected ? tint.opacity(0.9) : .white.opacity(0.06),
                              lineWidth: selected ? 2 : 1)
        )
        .shadow(color: selected ? tint.opacity(0.35) : .clear, radius: 8)
        .scaleEffect(selected ? 1.04 : 1)
    }
}

struct FileChecklist: View {
    @EnvironmentObject var engine: EngineService
    @Binding var selection: Set<String>
    @State private var files: [String] = []

    var body: some View {
        List(files, id: \.self) { file in
            Toggle(file, isOn: Binding(
                get: { selection.contains(file) },
                set: { on in
                    if on { selection.insert(file) } else { selection.remove(file) }
                }))
            .toggleStyle(.checkbox)
            .font(.caption)
        }
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .onAppear { Task { await load() } }
    }

    private func load() async {
        guard let (data, r) = try? await engine.rawRequest(
            "/api/files/report.json"), r.statusCode == 200,
              let report = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let artifacts = report["artifacts"] as? [String: Any] else { return }
        files = artifacts.keys.sorted()
    }
}

struct PreviewRow: View {
    @EnvironmentObject var engine: EngineService
    @StateObject private var player = PreviewPlayer()
    let label: String
    let icon: String
    let rel: String
    var isAudio = true
    var ready = false

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .foregroundStyle(ready ? Studio.accent : Color.secondary.opacity(0.5))
                .frame(width: 20)
            Text(label)
                .foregroundStyle(ready ? .primary : .secondary)
            if !ready {
                Text("not rendered").font(.caption2).foregroundStyle(.tertiary)
            }
            Spacer()
            if isAudio {
                Button { player.play(url: engine.fileURL(rel), key: label) } label: {
                    Image(systemName: player.playing == label
                          ? "stop.circle.fill" : "play.circle.fill")
                        .font(.title3)
                        .foregroundStyle(player.playing == label ? Studio.accent : .secondary)
                        .symbolEffect(.pulse, isActive: player.playing == label)
                }
                .buttonStyle(.borderless)
                .disabled(!ready)
            }
            Button("Download") { download() }
                .buttonStyle(.bordered).controlSize(.small)
                .disabled(!ready)
        }
    }

    private func download() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = URL(string: rel)!.lastPathComponent
        guard panel.runModal() == .OK, let dest = panel.url else { return }
        Task {
            guard let (data, _) = try? await engine.rawRequest("/api/files/\(rel)") else { return }
            try? data.write(to: dest)
        }
    }
}
