import Foundation

struct SessionInfo: Codable, Hashable, Identifiable {
    var id: String
    var dir: String
    var name: String?

    /// Human title: stored name, else the id minus its "-YYYYMMDD-HHMMSS" tail.
    var displayName: String {
        if let name, !name.isEmpty { return name }
        if let range = id.range(of: #"-\d{8}-\d{6}$"#, options: .regularExpression) {
            return String(id[id.startIndex..<range.lowerBound])
        }
        // Legacy "{stamp}-{hash}" ids: show the hash tail as "session-xxxxxxxx".
        if let range = id.range(of: #"^\d{8}-\d{6}-"#, options: .regularExpression) {
            return "session-\(id[range.upperBound...])"
        }
        return id
    }

    /// "18.09.2026 02:03" parsed from the timestamp embedded in the id.
    var displayDate: String? {
        guard let range = id.range(of: #"\d{8}-\d{6}"#, options: .regularExpression)
        else { return nil }
        let parser = DateFormatter()
        parser.dateFormat = "yyyyMMdd-HHmmss"
        guard let date = parser.date(from: String(id[range])) else { return nil }
        let out = DateFormatter()
        out.dateFormat = "dd.MM.yyyy HH:mm"
        return out.string(from: date)
    }
}

struct JobStatus: Codable, Hashable {
    var job: String?
    var stage: String?
    var detail: Detail?
    var error: String?
    var started: TimeInterval?

    struct Detail: Codable, Hashable {
        var item: String?
        var index: Int?
        var total: Int?
        var stage: String?
        /// Seconds remaining for this item, computed by the engine from its
        /// own per-item rate — never derive it client-side from job elapsed.
        var eta: Int?
    }
}

struct StatusSnapshot: Codable, Hashable {
    var session: SessionRef?
    var job: JobStatus?
    var mem_free_gib: Double?
    var compute: Compute?
    var problems: [String]?

    struct Compute: Codable, Hashable {
        var device: String?
        var quantization: String?
    }

    struct SessionRef: Codable, Hashable {
        var id: String
        var dir: String
        var name: String?
        var leaves: [Leaf]?
        var outputs: Outputs?
    }

    struct Outputs: Codable, Hashable {
        var master: Bool?
        var binaural: Bool?
        var midi: Bool?
    }

    struct Leaf: Codable, Hashable {
        var name: String
        var track: String?
        var done: Bool?
    }
}

struct Arrangement: Codable {
    var version: Int?
    var tracks: [Track]
    var buses: [Bus]
    var fx_sends: [FXSend]?

    struct Track: Codable, Identifiable, Hashable {
        var id: String { name }
        var name: String
        var instrument: String?
        var role: String?
        var ensemble: String?
        var divisi: Int?
        var line: String?
        var register: String?
        var style_suffix: String?
        var gm_program: Int?
        var pan_azimuth: Double?
        var seed_offset: Int?
        var kit: String?
        var parts: [String]?
        var split_kit: Bool?
        var bus_family: String?
        var enabled: Bool?
    }

    struct Bus: Codable, Identifiable, Hashable {
        var id: String { name }
        var name: String
        var members: [String]
        var sends: [String]?
        var gain_db: Double?
        var mute: Bool?
    }

    struct FXSend: Codable, Identifiable, Hashable {
        var id: String { name }
        var name: String
        var type: String
        var rt60: Double?
        var predelay_ms: Double?
        var time_ms: Double?
        var feedback: Double?
    }
}

struct Formats: Codable {
    var audio: [String]
    var sample_rates: [Int]
    var midi_types: [Int]
}

struct ModelEntry: Codable {
    var repo: String
    var downloaded: Bool
    var bytes: Int64
    var expected: Int64?
}

struct ModelStatus: Codable {
    var cache_dir: String
    var complete: Bool
    var models: [String: ModelEntry]
}

/// Instrument catalog for the orchestration "Add Track" picker.
struct Catalog: Codable {
    var instruments: [String: InstrumentInfo]
    var ensembles: [String: [String]]
    var kits: [String: [String]]

    struct InstrumentInfo: Codable {
        var family: String?
        var line: String?
    }
}
