import SwiftUI
import AppKit

/// ⌘, — audio output + engine resource settings.
struct SettingsView: View {
    @EnvironmentObject var engine: EngineService
    @AppStorage(AudioSettings.outputUIDKey) private var outputUID = ""
    @AppStorage(AudioSettings.volumeKey) private var volume = 0.8
    @AppStorage("engine.device") private var device = "auto"
    @AppStorage("engine.budget") private var budget = 8.0
    @AppStorage("engine.threads") private var threads = 6.0
    @AppStorage("engine.minfree") private var minFree = 4.0
    @AppStorage("engine.lowmem") private var lowMem = false
    @AppStorage("models.dir") private var modelDir = ""
    @State private var devices: [AudioOutputDevice] = []

    private var maxBudget: Double {
        max(8, Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824 - 4)
    }
    private var maxThreads: Double {
        Double(ProcessInfo.processInfo.processorCount)
    }

    var body: some View {
        Form {
            Section("Audio") {
                Picker("Output device", selection: $outputUID) {
                    Text("System Default").tag("")
                    ForEach(devices) { device in
                        Text(device.name).tag(device.uid)
                    }
                }
                LabeledContent("Preview volume") {
                    HStack {
                        Slider(value: $volume, in: 0...1)
                        Text("\(Int(volume * 100))%")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(width: 38, alignment: .trailing)
                    }
                }
            }

            Section("Engine") {
                Picker("Compute device", selection: $device) {
                    Text("Auto").tag("auto")
                    Text("Apple GPU (MPS)").tag("mps")
                    Text("CPU").tag("cpu")
                }
                Text("Auto uses the Metal GPU — packing weights to int8 when "
                     + "memory is too tight for full precision — and only "
                     + "drops to CPU when the GPU truly can't fit. Falls back "
                     + "to CPU if the GPU runs out mid-render.")
                    .font(.caption).foregroundStyle(.secondary)
                LabeledContent("Memory budget") {
                    HStack {
                        Slider(value: $budget, in: 4...maxBudget, step: 1)
                        Text("\(Int(budget)) GiB")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(width: 48, alignment: .trailing)
                    }
                }
                LabeledContent("CPU threads") {
                    HStack {
                        Slider(value: $threads, in: 2...maxThreads, step: 1)
                        Text("\(Int(threads))")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(width: 48, alignment: .trailing)
                    }
                }
                Toggle("Low-memory mode (slower)", isOn: $lowMem)
                Text("Offloads model weights to CPU between stages — lets a "
                     + "tight Mac render instead of running out of memory.")
                    .font(.caption).foregroundStyle(.secondary)
                LabeledContent("Free RAM required") {
                    HStack {
                        Slider(value: $minFree, in: 4...16, step: 0.5)
                        Text(String(format: "%.1f GiB", minFree))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(width: 56, alignment: .trailing)
                    }
                }
                Text("Rendering refuses below this much free RAM. The model "
                     + "needs ~6 GiB — lower the floor at your own risk "
                     + "(fail or heavy swap).")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    Text("Engine changes apply on restart")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("Restart Engine") { engine.restart() }
                        .buttonStyle(.bordered)
                }
            }

            Section("Model storage") {
                LabeledContent("Location") {
                    Text(modelDir.isEmpty
                         ? (engine.modelStatus?.cache_dir ?? "default")
                         : modelDir)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.middle)
                }
                HStack {
                    Button("Choose Folder…") { pickModelDir() }
                        .buttonStyle(.bordered)
                    if let s = engine.modelStatus, !s.complete {
                        Spacer()
                        Button("Download Models…") {
                            engine.showingModelSetup = true
                        }
                        .buttonStyle(.borderedProminent)
                    }
                }
                Text("Weights land here (~7 GB). Point this at a fast "
                     + "external SSD/NVMe if you like — applies on engine "
                     + "restart; existing downloads are not moved.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            if let mem = engine.status.mem_free_gib {
                Section("Resources") {
                    LabeledContent("Free memory") {
                        Text(String(format: "%.1f GiB", mem))
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(mem < 7 ? Color.red
                                             : mem < 10 ? Color.orange : .primary)
                    }
                    Text("The model wants ~7 GiB free; it will still try "
                         + "below that and fail cleanly if it truly can't "
                         + "fit. Turn on Low-memory mode on tight machines.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: 500)
        .onAppear { devices = AudioDevices.outputDevices() }
    }

    private func pickModelDir() {
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
