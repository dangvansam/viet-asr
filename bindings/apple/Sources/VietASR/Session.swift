import CVietASR

/// Whether a pushed audio chunk closed an endpoint.
public enum FrameStatus {
    /// Recognition is still in progress.
    case partial
    /// The core detected an endpoint — a final result is available.
    case final

    init(_ raw: VietasrFrameStatus) {
        self = raw.rawValue == 1 ? .final : .partial
    }
}

/// An incremental streaming recognition session.
///
/// Feed audio in chunks with `accept(_:)`; read `partialResult` as you go and
/// `finalResult` once `accept` reports `.final` (or when the stream ends).
/// A session keeps its parent ``Pipeline`` alive for its lifetime.
public final class Session {
    private let handle: OpaquePointer
    private let pipeline: Pipeline

    init(handle: OpaquePointer, pipeline: Pipeline) {
        self.handle = handle
        self.pipeline = pipeline
    }

    deinit {
        vietasr_session_free(handle)
    }

    /// Pushes a chunk of 16-bit PCM samples.
    @discardableResult
    public func accept(_ pcm: [Int16]) -> FrameStatus {
        let raw = pcm.withUnsafeBufferPointer { buf in
            vietasr_accept_waveform_s16(handle, buf.baseAddress, Int32(buf.count))
        }
        return FrameStatus(raw)
    }

    /// Pushes a chunk of 32-bit float PCM samples (range -1…1).
    @discardableResult
    public func accept(_ pcm: [Float]) -> FrameStatus {
        let raw = pcm.withUnsafeBufferPointer { buf in
            vietasr_accept_waveform_f32(handle, buf.baseAddress, Int32(buf.count))
        }
        return FrameStatus(raw)
    }

    /// The in-progress transcript.
    public var partialResult: TranscriptResult {
        TranscriptResult(json: vietasr_partial_result(handle).map { String(cString: $0) })
    }

    /// The latest result (partial or final, whichever the core last produced).
    public var result: TranscriptResult {
        TranscriptResult(json: vietasr_result(handle).map { String(cString: $0) })
    }

    /// The final transcript — call once the stream has ended.
    public var finalResult: TranscriptResult {
        TranscriptResult(json: vietasr_final_result(handle).map { String(cString: $0) })
    }

    /// Clears all session state, ready to reuse for a new utterance.
    public func reset() {
        vietasr_session_reset(handle)
    }
}
