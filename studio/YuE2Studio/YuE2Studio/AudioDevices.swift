import Foundation
import AudioToolbox

/// A CoreAudio output device (headphones, interface, AirPlay…).
struct AudioOutputDevice: Identifiable, Hashable {
    let id: AudioDeviceID
    let uid: String
    let name: String
}

/// HAL device enumeration — no third-party deps.
enum AudioDevices {
    static func outputDevices() -> [AudioOutputDevice] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size
        ) == noErr else { return [] }
        var ids = [AudioDeviceID](
            repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.stride)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids
        ) == noErr else { return [] }
        return ids.compactMap { id in
            guard hasOutputStreams(id),
                  let name = stringProperty(id, kAudioObjectPropertyName),
                  let uid = stringProperty(id, kAudioDevicePropertyDeviceUID)
            else { return nil }
            return AudioOutputDevice(id: id, uid: uid, name: name)
        }
    }

    /// Stored selections persist UIDs (stable across reconnects); resolve
    /// back to the current device ID by re-enumerating.
    static func deviceID(forUID uid: String) -> AudioDeviceID? {
        outputDevices().first { $0.uid == uid }?.id
    }

    private static func hasOutputStreams(_ id: AudioDeviceID) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreams,
            mScope: kAudioDevicePropertyScopeOutput,
            mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        return AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size) == noErr
            && size > 0
    }

    private static func stringProperty(
        _ id: AudioDeviceID, _ selector: AudioObjectPropertySelector
    ) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var value: CFString = "" as CFString
        var size = UInt32(MemoryLayout<CFString>.stride)
        let status = withUnsafeMutablePointer(to: &value) { ptr in
            AudioObjectGetPropertyData(id, &address, 0, nil, &size, ptr)
        }
        return status == noErr ? value as String : nil
    }
}

/// UserDefaults keys shared by SettingsView and PreviewPlayer.
enum AudioSettings {
    static let outputUIDKey = "audio.outputDeviceUID"
    static let volumeKey = "audio.previewVolume"

    static var outputDeviceID: AudioDeviceID? {
        let uid = UserDefaults.standard.string(forKey: outputUIDKey) ?? ""
        return uid.isEmpty ? nil : AudioDevices.deviceID(forUID: uid)
    }

    static var volume: Float {
        let value = UserDefaults.standard.double(forKey: volumeKey)
        return Float(value == 0 ? 0.8 : value)   // unset → 80%
    }
}
