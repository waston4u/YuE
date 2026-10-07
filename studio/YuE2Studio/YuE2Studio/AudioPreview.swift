import AVFoundation

/// AVPlayer preview — streams stems from the local engine HTTP endpoint and
/// routes to the configured output device via `audioOutputDeviceUniqueID`.
@MainActor
final class PreviewPlayer: ObservableObject {
    @Published var playing: String?
    private var player: AVPlayer?
    private var endObserver: Any?

    func play(url: URL, key: String? = nil) {
        let key = key ?? url.lastPathComponent
        if playing == key { stop(); return }
        stop()
        let item = AVPlayerItem(url: url)
        let p = AVPlayer(playerItem: item)
        // Route + level from Settings (⌘, → Audio). Resolve the UID first —
        // a remembered device that is unplugged falls back to system default.
        if let uid = UserDefaults.standard
            .string(forKey: AudioSettings.outputUIDKey), !uid.isEmpty,
           AudioDevices.deviceID(forUID: uid) != nil {
            p.audioOutputDeviceUniqueID = uid
        }
        p.volume = AudioSettings.volume
        player = p
        endObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime, object: item, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.stop() }
        }
        p.play()
        playing = key
    }

    func stop() {
        player?.pause()
        player = nil
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        endObserver = nil
        playing = nil
    }
}
