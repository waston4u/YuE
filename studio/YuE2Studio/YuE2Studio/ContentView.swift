import SwiftUI
import AppKit

struct ContentView: View {
    @EnvironmentObject var engine: EngineService
    @State private var selection: SessionInfo?
    @State private var tab = Tab.session

    enum Tab: String, CaseIterable, Identifiable {
        case session = "Session"
        case orchestration = "Orchestration"
        case mixer = "Mixer"
        case deliverables = "Deliverables"
        var id: String { rawValue }
        var symbol: String {
            switch self {
            case .session: return "waveform"
            case .orchestration: return "list.bullet.rectangle"
            case .mixer: return "slider.vertical.3"
            case .deliverables: return "shippingbox"
            }
        }
    }

    var body: some View {
        NavigationSplitView {
            VStack(spacing: 0) {
                List(engine.sessions, selection: $selection) { session in
                    HStack(spacing: 10) {
                        IconTile(symbol: "music.note.list",
                                 color: Studio.nameColor(session.displayName), size: 30)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(session.displayName).font(.callout.weight(.medium))
                                .lineLimit(1)
                            if let date = session.displayDate {
                                Text(date)
                                    .font(.caption2).foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }
                        }
                    }
                    .padding(.vertical, 2)
                    .tag(session)
                    .contextMenu {
                        Button(role: .destructive) {
                            Task {
                                await engine.deleteSession(id: session.id)
                                if selection == session { selection = nil }
                            }
                        } label: {
                            Label("Delete Session", systemImage: "trash")
                        }
                    }
                }
                .listStyle(.sidebar)
                // One-session rule: no switching while a job is active.
                .disabled(engine.jobActive)

                Divider()
                Button { engine.showingNewSession = true } label: {
                    Label("New Session", systemImage: "plus.sparkles")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(PrimaryGradientButton())
                .disabled(engine.jobActive)
                .padding(10)
            }
            .navigationTitle("YuE2 Studio")
            .navigationSplitViewColumnWidth(min: 200, ideal: 220)
        } detail: {
            VStack(spacing: 0) {
                HStack {
                    Picker("", selection: $tab) {
                        ForEach(Tab.allCases) { tab in
                            Label(tab.rawValue, systemImage: tab.symbol).tag(tab)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .fixedSize()
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)
                Divider()
                ZStack {
                    switch tab {
                    case .session: SessionView()
                    case .orchestration: OrchestrationView()
                    case .mixer: MixerView()
                    case .deliverables: DeliverablesView()
                    }
                    if engine.status.session == nil && selection == nil {
                        HeroEmptyState()
                            .transition(.opacity)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .animation(.snappy(duration: 0.25), value: tab)
                .animation(.easeInOut(duration: 0.2),
                           value: engine.status.session == nil && selection == nil)
                Divider()
                // Global transport — synchronized stem playback from any tab.
                if engine.status.session != nil {
                    TransportBar()
                    Divider()
                }
                StatusBar()
            }
        }
        .onChange(of: selection) { _, new in
            if engine.jobActive {
                // One-session rule: the selection cannot move while a job
                // runs — snap it back to the session the engine owns.
                syncSelection()
                return
            }
            if let new {
                Task {
                    await engine.openSession(id: new.id)
                    // A failed open must not leave the highlight lying about
                    // which session is actually open.
                    if engine.status.session?.id != new.id { syncSelection() }
                }
            }
        }
        // The server snapshot is the source of truth for the open session:
        // creating, opening, deleting or restoring after an engine restart
        // all surface here — keep the highlight and view in step with it.
        .onChange(of: engine.status.session?.id) { _, _ in syncSelection() }
        .onChange(of: engine.sessions) { _, _ in syncSelection() }
        .sheet(isPresented: $engine.showingNewSession) { NewSessionSheet() }
        .sheet(isPresented: $engine.showingModelSetup) { ModelSetupSheet() }
        .onAppear { engine.start() }
        .onReceive(NotificationCenter.default.publisher(
            for: NSApplication.willTerminateNotification)) { _ in
            engine.stop()
        }
    }

    /// Point the sidebar at the session the engine actually has open and
    /// land on the Session view — a freshly created session appears
    /// immediately instead of leaving the user in the empty state.
    private func syncSelection() {
        if let id = engine.status.session?.id {
            selection = engine.sessions.first { $0.id == id }
            if selection != nil { tab = .session }
        } else {
            selection = nil
        }
    }
}

/// Animated hero shown before any session is open.
struct HeroEmptyState: View {
    @EnvironmentObject var engine: EngineService

    var body: some View {
        VStack(spacing: 18) {
            PulseOrb(size: 110)
            Text("YuE2 Studio")
                .font(.largeTitle.weight(.bold))
                .foregroundStyle(Studio.gradient)
            Text("One prompt in — a full production suite out.\nStems, buses, masters, immersive beds and MIDI.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button { engine.showingNewSession = true } label: {
                Label("Create a Session", systemImage: "plus.sparkles")
            }
            .buttonStyle(PrimaryGradientButton())
            .disabled(engine.jobActive)
            .padding(.top, 6)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.regularMaterial)
        .allowsHitTesting(true)
    }
}

struct StatusBar: View {
    @EnvironmentObject var engine: EngineService

    var body: some View {
        HStack(spacing: 8) {
            StatusDot(on: engine.connected)
            // Done/cancelled jobs return the bar to idle — errors stay
            // visible (red, static) until dismissed or the next action.
            if let job = engine.status.job,
               job.stage != "done", job.stage != "cancelled" {
                let active = ["queued", "running", "cancelling"].contains(job.stage ?? "")
                if active {
                    ActivityMeter(color: job.stage == "cancelling" ? .red : .green)
                        .frame(width: 56)
                }
                Text("\(job.job ?? ""): \(job.stage ?? "")")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(job.stage == "error" ? .red
                                     : active ? .primary : .secondary)
                if let d = job.detail {
                    // detail.stage carries the phase ("planning · 8.9 tok/s",
                    // "loading model"); i/t only makes sense with a real total.
                    if let stage = d.stage, !stage.isEmpty {
                        Text("— \(stage)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    if let item = d.item, let i = d.index, let t = d.total, t > 0 {
                        Text("(\(item) \(i)/\(t))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                if let error = job.error { Text(error).font(.caption).foregroundStyle(.red) }
            } else if let error = engine.lastError {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.caption).foregroundStyle(.red)
                Text(error).font(.caption).foregroundStyle(.red).lineLimit(1)
                if !engine.connected {
                    Button("Retry") { engine.restart() }
                        .buttonStyle(.bordered).controlSize(.small)
                }
            } else {
                Text(engine.connected ? "Engine ready" : "Engine offline")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if let compute = engine.status.compute, let device = compute.device {
                // Truthful compute path: Metal GPU vs CPU, and the weight
                // format the engine actually loaded (bf16 or packed int8).
                let gpu = device.hasPrefix("mps")
                let quant = compute.quantization.flatMap { $0 == "none" ? nil : " \u{00B7} \($0)" } ?? ""
                Label(gpu ? "Metal GPU\(quant)" : "CPU\(quant)",
                      systemImage: gpu ? "gpu" : "cpu")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(gpu ? Color.accentColor : .secondary)
            }
            if let mem = engine.status.mem_free_gib {
                Label(String(format: "%.1f GB free", mem),
                      systemImage: "memorychip")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(mem < 7 ? Color.red
                                     : mem < 10 ? Color.orange : Color.secondary)
            }
            if let session = engine.status.session {
                Label(session.name ?? session.id, systemImage: "folder")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.bar)
    }
}

struct NewSessionSheet: View {
    @EnvironmentObject var engine: EngineService
    @Environment(\.dismiss) var dismiss
    @State private var name = ""
    @State private var preset = "House"
    @State private var style = ""
    @State private var lyrics = ""
    @State private var generateLyrics = false
    @State private var lyricTopic = ""
    @State private var cot = "full"
    @State private var seed = 831001
    @State private var tempo = 124
    @State private var seconds = 180
    @State private var detail = "ab2:24"
    @State private var picked: Set<String> = []
    @State private var creating = false

    /// Style presets — each seeds the prompt; custom text still wins.
    private static let presets = [
        "Custom", "House", "EDM", "Pop", "Hip Hop", "Rock", "Funk",
        "Jazz", "Folk", "Ambient", "Reggae", "Latin", "Lo-Fi", "Ballad",
        "Orchestral",
    ]
    private static let presetStyles: [String: String] = [
        "House": "deep house, four-on-the-floor groove",
        "EDM": "edm, big synths, festival energy",
        "Pop": "pop, catchy hooks, radio polish",
        "Hip Hop": "hip hop, hard beat, 808 bass",
        "Rock": "rock, driving guitars, live drums",
        "Funk": "funk, slap bass, tight groove",
        "Jazz": "jazz, swung groove, extended harmonies",
        "Folk": "folk, acoustic, intimate",
        "Ambient": "ambient, evolving pads, spacious",
        "Reggae": "reggae, offbeat skank, laid back",
        "Latin": "latin, percussion driven, warm",
        "Lo-Fi": "lo-fi, dusty beat, mellow rhodes",
        "Ballad": "pop ballad, emotional, piano led",
        "Orchestral": "cinematic orchestral, sweeping strings",
    ]

    /// Style string sent to the engine — preset text + custom additions
    /// + tempo/duration steering hints the planner reads.
    private var composedStyle: String {
        var parts = [Self.presetStyles[preset] ?? ""]
        let extra = style.trimmingCharacters(in: .whitespaces)
        if !extra.isEmpty { parts.append(extra) }
        parts.append("\(tempo) BPM")
        let m = seconds / 60, s = seconds % 60
        parts.append("approximately \(m) minute\(m == 1 ? "" : "s") \(s > 0 ? "\(s) seconds " : "")long")
        return parts.filter { !$0.isEmpty }.joined(separator: ", ")
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                PulseOrb(size: 44)
                VStack(alignment: .leading) {
                    Text("New Session").font(.headline)
                    Text("Style + lyrics in — production suite out")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding()

            ScrollView {
                Form {
                    TextField("Name", text: $name, prompt: Text("my-song"))
                    Picker("Style", selection: $preset) {
                        ForEach(Self.presets, id: \.self) { Text($0).tag($0) }
                    }
                    TextField("Add to style (optional)", text: $style,
                              prompt: Text("e.g. madonna-inspired, simple memorable melody"))
                    HStack {
                        Stepper("Tempo: \(tempo) BPM", value: $tempo, in: 50...200)
                    }
                    Picker("Length", selection: $seconds) {
                        Text("1:00").tag(60); Text("1:30").tag(90)
                        Text("2:00").tag(120); Text("2:30").tag(150)
                        Text("3:00").tag(180); Text("3:30").tag(210)
                        Text("4:00").tag(240)
                    }
                    Toggle(isOn: $generateLyrics) {
                        Label("Let YuE2 write the lyrics",
                              systemImage: "wand.and.sparkles")
                    }
                    if generateLyrics {
                        TextField("What should the song be about? (optional — empty = freestyle)",
                                  text: $lyricTopic)
                    } else {
                        TextField("Lyrics", text: $lyrics, axis: .vertical)
                            .lineLimit(3...8)
                    }
                    Picker("Planning", selection: $cot) {
                        Text("Full").tag("full"); Text("Melody").tag("melody"); Text("Off").tag("off")
                    }
                    Picker("Render detail", selection: $detail) {
                        Text("Draft — fastest").tag("ab2:12")
                        Text("Balanced").tag("ab2:24")
                        Text("Studio — best").tag("midpoint:32")
                    }
                    .help("Acoustic solver per stem — Draft ~12 evals, Balanced ~24, Studio 64 reference evals")
                    TextField("Seed", value: $seed, format: .number)

                    instrumentPicker
                }
                .formStyle(.grouped)
            }

            if let error = engine.lastError {
                Text(error)
                    .font(.caption).foregroundStyle(.red)
                    .lineLimit(3)
                    .padding(.horizontal)
                    .padding(.bottom, 8)
            }
        }
        .frame(width: 520, height: 620)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Cancel") { dismiss() }
            }
            ToolbarItem(placement: .confirmationAction) {
                Button(creating ? "Creating…" : "Create") {
                    let topic = lyricTopic.trimmingCharacters(in: .whitespaces)
                    let stylePrompt = generateLyrics && !topic.isEmpty
                        ? "\(composedStyle), lyrics about \(topic)" : composedStyle
                    creating = true
                    Task {
                        let parts = detail.split(separator: ":")
                        await engine.createSession(
                            name: name, style: stylePrompt,
                            lyrics: generateLyrics ? "" : lyrics,
                            cot: cot, seed: seed,
                            instruments: Array(picked),
                            odeSteps: parts.count == 2 ? Int(parts[1]) : nil,
                            odeMethod: parts.count == 2 ? String(parts[0]) : nil)
                        await engine.refreshSessions()
                        creating = false
                        // Only dismiss on success — a failed POST must not
                        // close the sheet and lose what the user typed.
                        if engine.lastError == nil { dismiss() }
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(creating)
            }
        }
        .onAppear { Task { await engine.fetchCatalog() } }
    }

    /// Multi-select instruments — empty means "derive from the style".
    @ViewBuilder private var instrumentPicker: some View {
        if let catalog = engine.catalog {
            Section("Instruments (empty = auto from style)") {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 110))],
                        spacing: 6) {
                    ForEach(catalog.instruments.keys.sorted(), id: \.self) { name in
                        Button {
                            if picked.contains(name) { picked.remove(name) }
                            else { picked.insert(name) }
                        } label: {
                            Text(name.replacingOccurrences(of: "_", with: " "))
                                .font(.caption2)
                                .lineLimit(1)
                                .padding(.horizontal, 8).padding(.vertical, 4)
                                .frame(maxWidth: .infinity)
                                .background(picked.contains(name)
                                            ? Studio.accent.opacity(0.3)
                                            : Color.primary.opacity(0.06),
                                            in: Capsule())
                                .foregroundStyle(picked.contains(name)
                                               ? Studio.accent : .secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                if !picked.isEmpty {
                    Text("\(picked.count) selected — these become the tracks")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
    }
}

/// First-run model-weight setup — download ~7 GB once, pick a fast disk.
struct ModelSetupSheet: View {
    @EnvironmentObject var engine: EngineService
    @AppStorage("models.dir") private var modelDir = ""
    @Environment(\.dismiss) private var dismiss
    /// Set the instant Download is tapped — the button must lock before the
    /// first status update arrives, or double-clicks fire duplicate POSTs.
    @State private var downloadRequested = false

    private var job: JobStatus? { engine.status.job }
    /// Any background job is active (download, render, mix…).
    private var jobActive: Bool {
        ["queued", "running", "cancelling"].contains(job?.stage ?? "")
    }
    private var downloading: Bool {
        downloadRequested || (job?.job == "models" && jobActive)
    }
    /// Aggregate bytes downloaded / expected across all repos.
    private var downloadFraction: Double {
        if let models = engine.modelStatus?.models {
            let got = models.values.reduce(0) { $0 + $1.bytes }
            let want = models.values.reduce(0) { $0 + ($1.expected ?? 0) }
            if want > 0 { return min(1, Double(got) / Double(want)) }
        }
        if let d = job?.detail, let i = d.index, let t = d.total, t > 0 {
            return min(1, Double(i) / Double(t))
        }
        return 0
    }

    var body: some View {
        VStack(spacing: 18) {
            if downloading {
                ProgressRing(fraction: downloadFraction)
                    .frame(width: 72, height: 72)
            } else {
                PulseOrb(size: 56, active: false)
            }

            VStack(spacing: 4) {
                Text("Download Model Weights")
                    .font(.title2.weight(.bold))
                Text("YuE2 needs ~7 GB of weights (YuE2-3B + VAE).\n"
                     + "They download once and are cached locally.")
                    .font(.callout).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            // Per-model checklist
            VStack(spacing: 6) {
                ForEach(["ar", "vae"], id: \.self) { key in
                    modelRow(key)
                }
            }
            .padding(10)
            .background(.regularMaterial,
                        in: RoundedRectangle(cornerRadius: 10))

            // Storage location
            HStack {
                Image(systemName: "externaldrive")
                    .foregroundStyle(.secondary)
                Text(modelDir.isEmpty
                     ? (engine.modelStatus?.cache_dir ?? "default cache")
                     : modelDir)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle)
                Spacer()
                Button("Choose…") { pickDir() }
                    .buttonStyle(.bordered).controlSize(.small)
                    .disabled(downloading)
            }

            // Which repo is streaming right now — the ring above shows the %.
            if downloading, let detail = job?.detail,
               let index = detail.index, let total = detail.total, total > 0 {
                HStack {
                    Text(detail.item ?? "")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Text(String(format: "%.1f / %.1f GB",
                                Double(index) / 1e9,
                                Double(total) / 1e9))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            } else if downloading {
                ProgressView()
                    .controlSize(.small)
                Text("Preparing download…")
                    .font(.caption).foregroundStyle(.secondary)
            } else if jobActive {
                // Some other job (render/mix) holds the engine — can't
                // start a download until it finishes.
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Waiting for \(job?.job ?? "job") to finish…")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }

            if let error = job?.error, job?.job == "models" {
                Text(error)
                    .font(.caption).foregroundStyle(.red)
                    .lineLimit(3)
            }
            if let error = engine.lastError {
                Text(error)
                    .font(.caption).foregroundStyle(.red)
                    .lineLimit(3)
            }

            HStack {
                if downloading {
                    Button("Cancel") {
                        downloadRequested = false
                        Task { await engine.cancel() }
                    }
                    .buttonStyle(.bordered)
                } else if engine.modelStatus?.complete == true {
                    Button("Done") { dismiss() }
                        .buttonStyle(.borderedProminent)
                        .keyboardShortcut(.defaultAction)
                } else {
                    Button("Later") { dismiss() }
                        .buttonStyle(.bordered)
                    Button("Download (~7 GB)") {
                        downloadRequested = true  // lock before the POST lands
                        Task { await engine.downloadModels() }
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                    // Locked until every running job finishes — a second
                    // click can only error, never help.
                    .disabled(jobActive)
                }
            }
        }
        .padding(28)
        .frame(width: 460)
        // Only a REAL models job traps the sheet — a stuck optimistic flag
        // must never lock the user out of dismissing it.
        .interactiveDismissDisabled(job?.job == "models" && jobActive)
        // See a job even if SSE lagged or dropped, then clear the optimistic
        // lock once the models job reaches any terminal state.
        .task { await engine.refreshStatus() }
        .onChange(of: engine.status.job?.stage) { stage in
            if ["done", "error", "cancelled"].contains(stage ?? "") {
                downloadRequested = false
            }
        }
        .onChange(of: engine.lastError) { error in
            if error != nil { downloadRequested = false }
        }
        // Download finished → close the sheet. Error/cancel stays open
        // so the user can see what happened and retry.
        .onChange(of: engine.modelStatus?.complete) { complete in
            if complete == true { dismiss() }
        }
        // Checklist bytes tick up while the download runs — and the status
        // pull self-heals a stuck flag: no active models job → release it.
        .onReceive(Timer.publish(every: 2, on: .main, in: .common)
            .autoconnect()) { _ in
            if downloading {
                Task {
                    await engine.fetchModels()
                    await engine.refreshStatus()
                    if downloadRequested
                        && !(engine.status.job?.job == "models"
                             && jobActive) {
                        downloadRequested = false
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func modelRow(_ key: String) -> some View {
        let entry = engine.modelStatus?.models[key]
        HStack {
            Image(systemName: entry?.downloaded == true
                  ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(entry?.downloaded == true
                                 ? Color.green : .secondary)
            Text(key == "ar" ? "YuE2-3B (generator)" : "YuE2-Vae (decoder)")
                .font(.callout)
            Spacer()
            Text(statusText(entry))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }

    private func statusText(_ entry: ModelEntry?) -> String {
        guard let entry else { return "unknown" }
        if entry.downloaded { return "ready" }
        if let expected = entry.expected, expected > 0 {
            return String(format: "%.1f / %.1f GB",
                          Double(entry.bytes) / 1e9,
                          Double(expected) / 1e9)
        }
        return entry.bytes > 0
            ? String(format: "%.1f GB", Double(entry.bytes) / 1e9)
            : "missing"
    }

    private func pickDir() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Select"
        panel.message = "Choose where model weights are stored"
        if panel.runModal() == .OK, let url = panel.url {
            modelDir = url.path
        }
    }
}
