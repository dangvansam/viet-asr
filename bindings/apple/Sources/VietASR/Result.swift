import Foundation

/// A transcription result, parsed from the JSON the C core emits.
///
/// The known fields (`text`, `partial`, `isFinal`, `segments`, `speakers`) are
/// surfaced directly; anything else a module contributes — e.g. `gender`,
/// `emotion` — is reachable through `field(_:)` or the raw `json` string.
public struct TranscriptResult: CustomStringConvertible {
    /// The raw JSON string returned by the C core.
    public let json: String

    private let object: [String: Any]

    init(json rawJSON: String?) {
        let text = rawJSON ?? ""
        self.json = text.isEmpty ? "{}" : text
        self.object = (try? JSONSerialization.jsonObject(with: Data(self.json.utf8)))
            as? [String: Any] ?? [:]
    }

    /// Final/best transcript text.
    public var text: String { object["text"] as? String ?? "" }

    /// In-progress transcript for a streaming session.
    public var partial: String { object["partial"] as? String ?? "" }

    /// Whether this result is an endpoint-finalised segment.
    public var isFinal: Bool { object["is_final"] as? Bool ?? false }

    /// Per-segment detail, as raw JSON objects (shape varies by module set).
    public var segments: [[String: Any]] { object["segments"] as? [[String: Any]] ?? [] }

    /// Diarisation output, as raw JSON objects, when a speaker module ran.
    public var speakers: [[String: Any]] { object["speakers"] as? [[String: Any]] ?? [] }

    /// Access any top-level field by name (`"gender"`, `"emotion"`, …).
    public func field(_ key: String) -> Any? { object[key] }

    public var description: String { "TranscriptResult(text: \"\(text)\")" }
}
