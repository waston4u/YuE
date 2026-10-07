import SwiftUI

/// Editable arrangement: tracks, ensembles, kits, divisi, line conditioning.
/// Every control persists immediately — the saved arrangement IS the render
/// queue: bypassed tracks produce no leaves and are never rendered.
struct OrchestrationView: View {
    @EnvironmentObject var engine: EngineService

    private let lines = ["melody", "counter", "chords", "bass", "rhythm"]

    var body: some View {
        VStack(spacing: 0) {
            if let arrangement = engine.arrangement {
                Table(arrangement.tracks) {
                    TableColumn("On") { track in
                        Toggle("", isOn: enabledBinding(track))
                            .labelsHidden()
                            .toggleStyle(.checkbox)
                            .help("Bypass — unchecked tracks render nothing")
                    }.width(32)
                    TableColumn("Track") { track in
                        HStack(spacing: 8) {
                            IconTile(symbol: Studio.icon(track.instrument ?? track.kit ?? track.name),
                                     color: Studio.familyColor(track.name), size: 26)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(track.name).font(.callout.weight(.medium))
                                Text(track.instrument ?? track.kit ?? track.role ?? "")
                                    .font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                        .opacity((track.enabled ?? true) ? 1 : 0.4)
                    }
                    TableColumn("Line") { track in
                        if track.kit == nil {
                            Picker("", selection: lineBinding(track)) {
                                ForEach(lines, id: \.self) { Text($0).tag($0) }
                            }
                            .labelsHidden()
                            .help("Which score part conditions this stem")
                        }
                    }.width(100)
                    TableColumn("Split") { track in
                        if track.kit != nil {
                            Toggle("parts", isOn: splitKitBinding(track))
                                .font(.caption2)
                                .help("Render each kit part as its own stem — parts share conditioning, expect overlap")
                        }
                    }.width(64)
                    TableColumn("Divisi") { track in
                        if track.kit == nil {
                            Stepper(value: divisiBinding(track), in: 1...4) {
                                Text("\(track.divisi ?? 1)")
                                    .font(.caption.monospacedDigit())
                            }
                            .help("Split the part into N divisi renders")
                        }
                    }.width(90)
                    TableColumn("Stems") { track in
                        let mine = (engine.status.session?.leaves ?? [])
                            .filter { $0.track == track.name }
                        if track.enabled == false {
                            Text("bypassed").font(.caption2).foregroundStyle(.tertiary)
                        } else if mine.isEmpty {
                            Text("—").foregroundStyle(.tertiary)
                        } else {
                            let done = mine.filter { $0.done ?? false }.count
                            let active = engine.status.job?.detail?.item
                            let rendering = mine.contains { $0.name == active }
                            HStack(spacing: 4) {
                                if rendering {
                                    ProgressView().controlSize(.mini)
                                }
                                Text("\(done)/\(mine.count)")
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(done == mine.count ? .green
                                                     : rendering ? Studio.accent : .secondary)
                            }
                        }
                    }.width(80)
                    TableColumn("Style suffix") { track in
                        Text(track.style_suffix ?? "—")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    TableColumn("") { track in
                        Button(role: .destructive) {
                            Task { await engine.removeTrack(track.name) }
                        } label: {
                            Image(systemName: "minus.circle")
                        }
                        .buttonStyle(.borderless)
                        .help("Remove track")
                    }.width(28)
                }
            } else if engine.status.session != nil {
                ContentUnavailableView("No arrangement yet",
                    systemImage: "list.bullet.rectangle",
                    description: Text(engine.jobActive
                        ? "The orchestration appears when planning finishes — edit it before the first stem renders."
                        : "This session has no arrangement yet — render once or add tracks to create it."))
            } else {
                ContentUnavailableView("No arrangement",
                    systemImage: "list.bullet.rectangle",
                    description: Text("Open a session to see its orchestration."))
            }
            Divider()
            HStack(spacing: 10) {
                addMenu
                Button { Task { await engine.fetchArrangement() } } label: {
                    Label("Reload", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.bordered)
                Spacer()
                let total = engine.arrangement?.tracks.count ?? 0
                let active = engine.arrangement?.tracks
                    .filter { $0.enabled ?? true }.count ?? 0
                Label("\(active) of \(total) tracks · \(engine.status.session?.leaves?.count ?? 0) leaf stems",
                      systemImage: "square.stack.3d.up")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(10)
        }
        .onAppear {
            Task {
                await engine.fetchArrangement()
                await engine.fetchCatalog()
            }
        }
        // Session switch or a finished session/render job → reload so the
        // table always shows the current project's orchestration.
        .onChange(of: engine.status.session?.id) {
            Task { await engine.fetchArrangement() }
        }
        .onChange(of: engine.status.job?.stage) {
            if engine.status.job?.stage == "done" {
                Task { await engine.fetchArrangement() }
            }
        }
        // The arrangement lands mid-planning, before the job finishes —
        // leaf list appearing is the signal it's on disk.
        .onChange(of: engine.status.session?.leaves?.count) {
            Task { await engine.fetchArrangement() }
        }
    }

    // ── add track ───────────────────────────────────────────────────────

    private var addMenu: some View {
        Menu {
            if let catalog = engine.catalog {
                ForEach(families(catalog), id: \.self) { family in
                    Menu(family.capitalized) {
                        ForEach(catalog.instruments
                            .filter { $0.value.family == family }
                            .map(\.key).sorted(), id: \.self) { name in
                            Button(name.replacingOccurrences(of: "_", with: " ")) {
                                Task { await engine.addTrack(instrument: name) }
                            }
                        }
                    }
                }
                Divider()
                Menu("Ensembles") {
                    ForEach(catalog.ensembles.keys.sorted(), id: \.self) { name in
                        Button(name.replacingOccurrences(of: "_", with: " ")) {
                            Task { await engine.addTrack(ensemble: name) }
                        }
                    }
                }
                Menu("Drum kits") {
                    ForEach(catalog.kits.keys.sorted(), id: \.self) { name in
                        Button(name) {
                            Task { await engine.addTrack(kit: name) }
                        }
                    }
                }
            } else {
                Text("Catalog unavailable")
            }
        } label: {
            Label("Add Track", systemImage: "plus")
        }
        .menuStyle(.borderedButton)
        .disabled(engine.arrangement == nil || engine.jobActive)
    }

    private func families(_ catalog: Catalog) -> [String] {
        Array(Set(catalog.instruments.values.compactMap(\.family))).sorted()
    }

    // ── bindings ────────────────────────────────────────────────────────

    private func enabledBinding(_ track: Arrangement.Track) -> Binding<Bool> {
        Binding(
            get: { track.enabled ?? true },
            set: { value in Task { await engine.updateTrack(track.name) {
                $0.enabled = value } } })
    }

    private func lineBinding(_ track: Arrangement.Track) -> Binding<String> {
        Binding(
            get: { track.line ?? "melody" },
            set: { value in Task { await engine.updateTrack(track.name) {
                $0.line = value } } })
    }

    private func divisiBinding(_ track: Arrangement.Track) -> Binding<Int> {
        Binding(
            get: { track.divisi ?? 1 },
            set: { value in Task { await engine.updateTrack(track.name) {
                $0.divisi = value } } })
    }

    private func splitKitBinding(_ track: Arrangement.Track) -> Binding<Bool> {
        Binding(
            get: { track.split_kit ?? false },
            set: { value in Task { await engine.updateTrack(track.name) {
                $0.split_kit = value } } })
    }
}

/// Small colored capsule label.
struct Chip: View {
    let text: String
    var color: Color
    init(_ text: String, color: Color = .accentColor) {
        self.text = text; self.color = color
    }
    var body: some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 7).padding(.vertical, 2)
            .background(color.opacity(0.18), in: Capsule())
            .foregroundStyle(color)
    }
}
