import SwiftUI

/// Per-stem render progress + QC badges + preview.
struct SessionView: View {
    @EnvironmentObject var engine: EngineService
    @StateObject private var player = PreviewPlayer()

    private var runningJob: JobStatus? {
        guard let job = engine.status.job,
              ["queued", "running", "cancelling"].contains(job.stage) else { return nil }
        return job
    }

    private var jobError: String? {
        guard let job = engine.status.job, job.stage == "error" else { return nil }
        return job.error
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 14) {
                // ── action bar ───────────────────────────────────────
                HStack(spacing: 10) {
                    Button { Task { await engine.render() } } label: {
                        Label(runningJob?.job == "render" ? "Rendering…" : "Render All Stems",
                              systemImage: "bolt.fill")
                    }
                    .buttonStyle(PrimaryGradientButton())
                    .disabled(engine.status.session == nil || runningJob != nil)
                    Button { Task { await engine.mix() } } label: {
                        Label(runningJob?.job == "mix" ? "Mixing…" : "Mix Session",
                              systemImage: "slider.vertical.3")
                    }
                    .buttonStyle(.bordered)
                    .disabled(engine.status.session == nil || runningJob != nil)
                    Button { Task { await engine.cancel() } } label: {
                        Label(engine.status.job?.stage == "cancelling" ? "Cancelling…" : "Cancel",
                              systemImage: "xmark")
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                    .disabled(runningJob == nil)
                    Spacer()
                    Button { reveal() } label: {
                        Label("Reveal", systemImage: "folder")
                    }
                    .buttonStyle(.bordered)
                    .disabled(engine.status.session == nil)
                }
                .padding(.horizontal)

                // ── readiness warnings — everything that would fail a
                // render, shown before you click, not after it dies.
                if let problems = engine.status.problems, !problems.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(problems, id: \.self) { problem in
                            HStack(spacing: 8) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundStyle(.orange)
                                Text(problem).font(.callout).lineLimit(2)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
                    .background(Color.orange.opacity(0.12),
                                in: RoundedRectangle(cornerRadius: 10))
                    .padding(.horizontal)
                    .transition(.move(edge: .top).combined(with: .opacity))
                }

                // ── error banner ─────────────────────────────────────
                if let error = engine.lastError ?? jobError {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                        Text(error).font(.callout).lineLimit(3)
                        Spacer()
                        Button { engine.lastError = nil
                                 Task { await engine.dismissJob() } } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.borderless)
                    }
                    .padding(10)
                    .background(Color.red.opacity(0.12),
                                in: RoundedRectangle(cornerRadius: 10))
                    .padding(.horizontal)
                    .transition(.move(edge: .top).combined(with: .opacity))
                }

                // ── generating banner ────────────────────────────────
                if let job = runningJob {
                    GeneratingBanner(job: job)
                        .padding(.horizontal)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }

                // ── stem cards ───────────────────────────────────────
                if let leaves = engine.status.session?.leaves, !leaves.isEmpty {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 230), spacing: 10)],
                              spacing: 10) {
                        ForEach(leaves, id: \.name) { leaf in
                            StemCard(leaf: leaf.name,
                                     done: leaf.done ?? false,
                                     rendering: runningJob?.detail?.item == leaf.name,
                                     player: player,
                                     url: engine.fileURL("stems/\(leaf.name)/audio.flac"))
                        }
                    }
                    .padding(.horizontal)
                } else if runningJob == nil {
                    if engine.status.session != nil {
                        ContentUnavailableView("No orchestration yet",
                            systemImage: "list.bullet.rectangle",
                            description: Text("This session has no arrangement — it needs a score (engine or ABC) first."))
                        .padding(.top, 60)
                    } else {
                        ContentUnavailableView("No session open",
                            systemImage: "music.note.list",
                            description: Text("Create a new session or open one from the sidebar."))
                        .padding(.top, 60)
                    }
                }
            }
            .padding(.vertical, 12)
            .padding(.bottom, 40)
        }
        .animation(.spring(duration: 0.4), value: runningJob?.detail?.item)
    }

    private func reveal() {
        guard let dir = engine.status.session?.dir else { return }
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: dir)
    }
}

/// The live render transport — stage readout, precision progress, meters.
struct GeneratingBanner: View {
    @EnvironmentObject var engine: EngineService
    let job: JobStatus

    /// Last meaningful engine log line (tqdm bars spam \r-separated segments).
    private var logTail: String {
        engine.engineLog
            .split(whereSeparator: \.isNewline)
            .last.map(String.init)?
            .split(separator: "\r").last.map(String.init) ?? ""
    }

    private var stageTitle: String {
        if job.stage == "cancelling" { return "CANCELLING" }
        switch job.job {
        case "mix": return "MIXING"
        case "session": return "PREPARING SESSION"
        case "models": return "DOWNLOADING MODELS"
        case "export": return "EXPORTING"
        default: return "RENDERING"
        }
    }

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            let elapsed = Int(context.date.timeIntervalSince1970
                              - (job.started ?? context.date.timeIntervalSince1970))
            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 10) {
                        Text(stageTitle)
                            .font(.caption.weight(.bold))
                            .foregroundStyle(job.stage == "cancelling" ? Color.red : Studio.accent)
                            .tracking(1.5)
                        if let item = job.detail?.item {
                            Text(item)
                                .font(.callout.weight(.semibold))
                                .foregroundStyle(Studio.familyColor(item))
                        }
                        if let stage = job.detail?.stage, !stage.isEmpty {
                            Text(stage.uppercased())
                                .font(.caption2.weight(.medium))
                                .foregroundStyle(.secondary)
                                .tracking(1)
                        }
                    }
                    if (job.detail?.total ?? 0) > 0 {
                        ProgressView(value: Double(job.detail?.index ?? 0),
                                     total: Double(job.detail?.total ?? 1))
                            .tint(job.stage == "cancelling" ? .red : Studio.accent)
                            .animation(.linear(duration: 0.3),
                                       value: job.detail?.index)
                    } else {
                        // Token-level work has no honest total — an
                        // indeterminate bar looks like a stuck/fake
                        // progress bar. The live token count + rate below
                        // is the real liveness signal.
                        Divider().opacity(0.15)
                    }
                    HStack(spacing: 14) {
                        if let d = job.detail, (d.total ?? 0) > 0 {
                            // Item carries the real name — leaf during stem
                            // renders, "engine" during load/plan steps.
                            Text("\((d.item ?? "step").uppercased()) \(d.index ?? 0) / \(d.total ?? 0)")
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                        } else if (job.detail?.index ?? 0) > 0 {
                            Text("\(job.detail?.index ?? 0) TOKENS")
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        Text(String(format: "%02d:%02d", elapsed / 60, elapsed % 60))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                        // Server-computed ETA — per-item rate × remaining,
                        // so it can't jump when the job crosses stems.
                        if let eta = job.detail?.eta {
                            Text(String(format: "~%d:%02d left", eta / 60, eta % 60))
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }
                    if !logTail.isEmpty {
                        Text(logTail)
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.head)
                    }
                }
                Spacer()
                HStack(spacing: 4) {
                    LiveMeter(active: job.stage == "running", seed: 0, height: 44)
                    LiveMeter(active: job.stage == "running", seed: 0.6, height: 44)
                }
            }
            .padding(14)
        }
        .background {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(.regularMaterial)
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(.white.opacity(0.1), lineWidth: 1)
        }
    }
}

/// One stem — family-colored tile, name, live status (queued/generating/ready), play.
struct StemCard: View {
    let leaf: String
    let done: Bool
    let rendering: Bool
    @ObservedObject var player: PreviewPlayer
    let url: URL

    var body: some View {
        HStack(spacing: 10) {
            IconTile(symbol: Studio.icon(leaf), color: Studio.familyColor(leaf), size: 36)
            VStack(alignment: .leading, spacing: 2) {
                Text(leaf).font(.callout.weight(.medium)).lineLimit(1)
                Text(rendering ? "generating…" : done ? "ready" : "queued")
                    .font(.caption2)
                    .foregroundStyle(rendering ? Studio.accent
                                     : done ? .green : .secondary)
            }
            Spacer()
            if rendering {
                ProgressView().controlSize(.small)
                    .transition(.scale.combined(with: .opacity))
            } else if done {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
            } else {
                Image(systemName: "clock")
                    .foregroundStyle(.tertiary)
            }
            if done {
                Button { player.play(url: url, key: leaf) } label: {
                    Image(systemName: player.playing == leaf
                          ? "stop.circle.fill" : "play.circle.fill")
                        .font(.title3)
                        .foregroundStyle(player.playing == leaf ? Studio.accent : .secondary)
                        .symbolEffect(.pulse, isActive: player.playing == leaf)
                }
                .buttonStyle(.borderless)
            }
        }
        .opacity(done || rendering ? 1 : 0.55)
        .padding(10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(rendering ? Studio.accent.opacity(0.7) : .white.opacity(0.06),
                              lineWidth: rendering ? 1.5 : 1)
        )
        .shadow(color: rendering ? Studio.accent.opacity(0.3) : .clear, radius: 8)
        .animation(.spring(duration: 0.3), value: rendering)
    }
}
