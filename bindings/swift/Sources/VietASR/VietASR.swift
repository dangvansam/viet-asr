import CVietASR
import Foundation

/// Vietnamese automatic speech recognition.
///
/// Wraps the VietASR C ABI. The native library auto-detects the CoreML
/// execution provider on iOS and falls back to CPU.
public final class VietASR {

    /// ONNX Runtime execution provider. `.auto` probes the best provider on the
    /// device (CoreML on iOS) and falls back to CPU.
    public enum ExecutionProvider: Int32 {
        case auto = 0
        case cpu = 1
        case coreML = 4
    }

    /// Execution providers compiled into the bundled ONNX Runtime.
    public static var availableExecutionProviders: [String] {
        guard let c = vietasr_available_execution_providers() else { return [] }
        let json = Data(String(cString: c).utf8)
        return (try? JSONDecoder().decode([String].self, from: json)) ?? []
    }

    /// VietASR native library version.
    public static var version: String {
        guard let c = vietasr_version() else { return "" }
        return String(cString: c)
    }

    /// Pipeline preset names available in the native library
    /// (e.g. "transcribe", "stream", "batch", "vad").
    public static var availablePresets: [String] {
        guard let c = vietasr_list_presets() else { return [] }
        let json = Data(String(cString: c).utf8)
        return (try? JSONDecoder().decode([String].self, from: json)) ?? []
    }

    /// Module names that can be added to a custom pipeline
    /// (e.g. "vad", "vietasr", "itn").
    public static var availableModules: [String] {
        guard let c = vietasr_list_modules() else { return [] }
        let json = Data(String(cString: c).utf8)
        return (try? JSONDecoder().decode([String].self, from: json)) ?? []
    }

    private let pipeline: OpaquePointer

    /// Builds a pipeline from a preset (e.g. "transcribe", "transcribe-rich").
    /// The execution provider is auto-detected (CoreML on iOS, CPU fallback).
    public init?(preset: String) {
        guard let p = vietasr_pipeline_preset(preset) else { return nil }
        pipeline = p
    }

    /// Builds a pipeline from an explicit module list (added in order, e.g.
    /// ["vad", "vietasr"]), forcing a specific execution provider.
    public init?(modules: [String],
                 executionProvider: ExecutionProvider = .auto,
                 deviceID: Int32 = 0) {
        guard let p = vietasr_pipeline_new() else { return nil }
        _ = vietasr_pipeline_set_execution_provider(
            p,
            VietasrExecutionProvider(UInt32(executionProvider.rawValue)),
            deviceID)
        for module in modules {
            if vietasr_pipeline_add_module(p, module, "{}") != VIETASR_OK {
                vietasr_pipeline_free(p)
                return nil
            }
        }
        if vietasr_pipeline_build(p) != VIETASR_OK {
            vietasr_pipeline_free(p)
            return nil
        }
        pipeline = p
    }

    /// Builds a pipeline from modules with a per-module JSON config string,
    /// e.g. `[(name: "vad", config: #"{"backend":"silero"}"#),
    ///        (name: "vietasr", config: #"{"itn":true,"emit_segments":true}"#)]`.
    /// Use this when a preset is not enough — to select a VAD backend, enable
    /// ITN, etc. Pass `"{}"` for a module that needs no config.
    public init?(configuredModules: [(name: String, config: String)],
                 executionProvider: ExecutionProvider = .auto,
                 deviceID: Int32 = 0) {
        guard let p = vietasr_pipeline_new() else { return nil }
        _ = vietasr_pipeline_set_execution_provider(
            p,
            VietasrExecutionProvider(UInt32(executionProvider.rawValue)),
            deviceID)
        for module in configuredModules {
            if vietasr_pipeline_add_module(
                    p, module.name, module.config) != VIETASR_OK {
                vietasr_pipeline_free(p)
                return nil
            }
        }
        if vietasr_pipeline_build(p) != VIETASR_OK {
            vietasr_pipeline_free(p)
            return nil
        }
        pipeline = p
    }

    deinit { vietasr_pipeline_free(pipeline) }

    /// Transcribes a 16-bit PCM buffer. Returns the result JSON.
    public func transcribe(pcm: [Int16], sampleRate: Float) -> String {
        pcm.withUnsafeBufferPointer { buf in
            guard let c = vietasr_transcribe_buffer(
                pipeline, buf.baseAddress, Int32(buf.count), sampleRate)
            else { return "{}" }
            return String(cString: c)
        }
    }

    /// Transcribes a WAV file. Returns the result JSON.
    public func transcribe(wavPath: String) -> String {
        guard let c = vietasr_transcribe_file(pipeline, wavPath) else { return "{}" }
        return String(cString: c)
    }

    /// Opens a streaming session for incremental audio.
    public func makeSession(sampleRate: Float) -> Session? {
        guard let s = vietasr_session_new(pipeline, sampleRate) else { return nil }
        return Session(handle: s)
    }

    /// A streaming recognition session — feed audio chunk by chunk.
    public final class Session {
        private let handle: OpaquePointer

        fileprivate init(handle: OpaquePointer) { self.handle = handle }
        deinit { vietasr_session_free(handle) }

        /// Feeds 16-bit PCM samples.
        public func accept(pcm: [Int16]) {
            pcm.withUnsafeBufferPointer { buf in
                _ = vietasr_accept_waveform_s16(handle, buf.baseAddress, Int32(buf.count))
            }
        }

        /// Partial result JSON for the audio fed so far.
        public func partialResult() -> String {
            guard let c = vietasr_partial_result(handle) else { return "{}" }
            return String(cString: c)
        }

        /// Flushes pending audio and returns the final result JSON.
        public func finalResult() -> String {
            guard let c = vietasr_final_result(handle) else { return "{}" }
            return String(cString: c)
        }

        /// Canonical name of the execution provider this session runs on
        /// (e.g. "CoreMLExecutionProvider", "CPUExecutionProvider").
        public var activeExecutionProvider: String {
            guard let c = vietasr_active_execution_provider(handle) else { return "" }
            return String(cString: c)
        }

        /// Clears all session state.
        public func reset() { vietasr_session_reset(handle) }
    }
}
