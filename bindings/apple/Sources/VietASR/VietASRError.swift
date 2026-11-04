import CVietASR

/// An error raised by the VietASR C core.
///
/// `message` is taken from `vietasr_last_error()` when available, otherwise a
/// generic description of the failed operation.
public struct VietASRError: Error, CustomStringConvertible {
    /// Human-readable cause, sourced from the C core where possible.
    public let message: String
    /// The `VietasrStatus` code, when the failing call returned one.
    public let code: VietasrStatus?

    public var description: String {
        if let code {
            return "VietASRError(\(code.rawValue)): \(message)"
        }
        return "VietASRError: \(message)"
    }

    /// Builds an error, preferring the C core's last-error string.
    static func make(_ fallback: String, code: VietasrStatus? = nil) -> VietASRError {
        let last = vietasr_last_error().flatMap { String(cString: $0) } ?? ""
        return VietASRError(message: last.isEmpty ? fallback : last, code: code)
    }
}
