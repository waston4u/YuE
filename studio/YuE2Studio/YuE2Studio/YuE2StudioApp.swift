import SwiftUI

@main
struct YuE2StudioApp: App {
    @StateObject private var engine = EngineService()
    @Environment(\.openWindow) private var openWindow

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(engine)
                .frame(minWidth: 960, minHeight: 620)
        }
        .windowStyle(.titleBar)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About YuE2 Studio") { openWindow(id: "about") }
            }
            CommandGroup(replacing: .newItem) {
                Button("New Session…") { engine.showingNewSession = true }
                    .keyboardShortcut("n")
                    .disabled(engine.jobActive)
            }
        }

        Settings {
            SettingsView()
                .environmentObject(engine)
        }

        Window("About YuE2 Studio", id: "about") {
            AboutView()
        }
        .windowResizability(.contentSize)
    }
}
