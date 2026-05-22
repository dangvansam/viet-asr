import CVietASR
import Foundation

/// Inference backend for the ASR engine.
public enum Backend {
    /// Let the core pick (currently ONNX Runtime everywhere).
    case auto
    /// ONNX Runtime CPU — the only backend implemented today.
    case onnx
    /// CoreML — reserved; the core still stubs this.
    case coreml

    var raw: VietasrBackend {
        switch self {
        case .auto:   return VietasrBackend(rawValue: 0)
        case .onnx:   return VietasrBackend(rawValue: 1)
        case .coreml: return VietasrBackend(rawValue: 2)
        }
    }
}

/// Logging verbosity for the C core.
public enum LogLevel: UInt32 {
    case trace = 0, debug = 1, info = 2, warn = 3, error = 4, off = 5
}

/// A VietASR processing pipeline.
///
/// Create one from a preset — `Pipeline.preset("transcribe")` — or assemble a
/// custom pipeline with `Pipeline.create()`, `add(_:)` and `build()`. Use
/// `transcribe(...)` for whole-clip recognition or `stream(...)` for
/// incremental streaming.
public final class Pipeline {
    let handle: OpaquePointer

    private init(handle: OpaquePointer) {
        self.handle = handle
    }

    deinit {
        vietasr_pipeline_free(handle)
    }

    // MARK: Construction

    /// Builds a ready-to-use pipeline from a named preset
    /// (`"transcribe"`, `"analytics"`, …; see ``listPresets()``).
    public static func preset(_ name: String) throws -> Pipeline {
        guard let handle = vietasr_pipeline_preset(name) else {
            throw VietASRError.make("unknown preset: \(name)")
        }
        return Pipeline(handle: handle)
    }

    /// Creates an empty pipeline. Add modules with `add(_:)`, then `build()`.
    public static func create() throws -> Pipeline {
        guard let handle = vietasr_pipeline_new() else {
            throw VietASRError.make("could not create pipeline")
        }
        return Pipeline(handle: handle)
    }

    // MARK: Configuration (chainable)

    /// Appends a module (`"vad"`, `"vietasr"`, `"punctuation"`, …), optionally
    /// with a JSON config string.
    @discardableResult
    public func add(_ moduleName: String, config: String? = nil) throws -> Pipeline {
        let status: VietasrStatus = config.map {
            vietasr_pipeline_add_module(handle, moduleName, $0)
        } ?? vietasr_pipeline_add_module(handle, moduleName, nil)
        try Self.check(status, "add module: \(moduleName)")
        return self
    }

    /// Selects the inference backend. Optional — defaults to `.auto`.
    @discardableResult
    public func setBackend(_ backend: Backend) throws -> Pipeline {
        try Self.check(vietasr_pipeline_set_backend(handle, backend.raw), "set backend")
        return self
    }

    /// Overrides the model directory (the bundled embedded model is used by
    /// default — most callers never need this).
    @discardableResult
    public func setModelDir(_ path: String) throws -> Pipeline {
        try Self.check(vietasr_pipeline_set_model_dir(handle, path), "set model dir")
        return self
    }

    /// Finalises a custom pipeline. Not needed after `preset(_:)`.
    @discardableResult
    public func build() throws -> Pipeline {
        try Self.check(vietasr_pipeline_build(handle), "build pipeline")
        return self
    }

    // MARK: Batch transcription

    /// Transcribes a whole WAV file.
    public func transcribe(file path: String) throws -> TranscriptResult {
        guard let raw = vietasr_transcribe_file(handle, path) else {
            throw VietASRError.make("transcribe failed: \(path)")
        }
        return TranscriptResult(json: String(cString: raw))
    }

    /// Transcribes a whole clip of 16-bit PCM samples.
    public func transcribe(_ pcm: [Int16], sampleRate: Float = 16_000) throws -> TranscriptResult {
        let raw: UnsafePointer<CChar>? = pcm.withUnsafeBufferPointer { buf in
            vietasr_transcribe_buffer(handle, buf.baseAddress, Int32(buf.count), sampleRate)
        }
        guard let raw else {
            throw VietASRError.make("transcribe(buffer) failed")
        }
        return TranscriptResult(json: String(cString: raw))
    }

    // MARK: Streaming

    /// Opens an incremental streaming session at the given sample rate.
    public func stream(sampleRate: Float = 16_000) throws -> Session {
        guard let session = vietasr_session_new(handle, sampleRate) else {
            throw VietASRError.make("could not open streaming session")
        }
        return Session(handle: session, pipeline: self)
    }

    // MARK: Helpers

    static func check(_ status: VietasrStatus, _ what: String) throws {
        if status.rawValue != 0 {
            throw VietASRError.make("\(what) failed", code: status)
        }
    }
}

// MARK: - Library-level helpers

public enum VietASR {
    /// The C core version string.
    public static var version: String {
        vietasr_version().map { String(cString: $0) } ?? "unknown"
    }

    /// Names of every available preset.
    public static func listPresets() -> [String] {
        decodeStringArray(vietasr_list_presets())
    }

    /// Names of every available module.
    public static func listModules() -> [String] {
        decodeStringArray(vietasr_list_modules())
    }

    /// Sets the C core's logging verbosity.
    public static func setLogLevel(_ level: LogLevel) {
        vietasr_set_log_level(VietasrLogLevel(rawValue: level.rawValue))
    }

    private static func decodeStringArray(_ raw: UnsafePointer<CChar>?) -> [String] {
        guard let raw else { return [] }
        let json = String(cString: raw)
        let parsed = try? JSONSerialization.jsonObject(with: Data(json.utf8))
        return (parsed as? [String]) ?? []
    }
}
