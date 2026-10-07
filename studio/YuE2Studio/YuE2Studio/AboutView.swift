import SwiftUI
import AppKit

/// About window — developer, credits, and the bundled license texts.
struct AboutView: View {
    @State private var tab = "license"

    private var version: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.1"
    }

    var body: some View {
        VStack(spacing: 12) {
            Image(nsImage: NSApplication.shared.applicationIconImage)
                .resizable()
                .frame(width: 72, height: 72)
            VStack(spacing: 2) {
                Text("YuE2 Studio").font(.title2.weight(.bold))
                Text("Version \(version)")
                    .font(.callout).foregroundStyle(.secondary)
                Text("Miro Sedlacek")
                    .font(.callout)
            }
            Text("One prompt in — a full production suite out.\n"
                 + "Per-stem rendering, orchestration, mixing, "
                 + "immersive beds and MIDI.")
                .font(.caption).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            Picker("", selection: $tab) {
                Text("License").tag("license")
                Text("Model License").tag("model")
                Text("Credits").tag("credits")
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            ScrollView {
                HStack {
                    Spacer(minLength: 0)
                    Text(document(tab))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: 500, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(8)
                    Spacer(minLength: 0)
                }
            }
            .background(.black.opacity(0.2),
                        in: RoundedRectangle(cornerRadius: 8))
        }
        .padding(20)
        .frame(width: 600, height: 520)
    }

    private func document(_ key: String) -> String {
        switch key {
        case "model":
            return resource("MODEL_LICENSE")
        case "credits":
            return """
            YuE2 Studio — generative music production suite
            Developer: Miro Sedlacek

            Built on YuE2 (Apache License 2.0) and the YuE2-3B /
            YuE2-Vae checkpoint weights by the m-a-p project,
            licensed CC BY-NC 4.0 (non-commercial use only):
            https://huggingface.co/m-a-p/YuE2-3B
            https://huggingface.co/m-a-p/YuE2-Vae

            Native macOS app — SwiftUI, AVAudioEngine, Metal
            Performance Shaders. Engine: PyTorch / transformers /
            safetensors / tiktoken / soundfile.
            """
        default:
            return resource("LICENSE")
        }
    }

    private func resource(_ name: String) -> String {
        guard let url = Bundle.main.url(forResource: name, withExtension: nil),
              let text = try? String(contentsOf: url, encoding: .utf8)
        else { return "\(name) not bundled." }
        return text
    }
}
