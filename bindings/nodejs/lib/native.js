const fs = require("fs");
const os = require("os");
const path = require("path");

class NativeLibrary {
    constructor() {
        const koffi = require("koffi");
        const nativeDir = NativeLibrary.findNativeDirectory();
        NativeLibrary.preloadSiblings(koffi, nativeDir);
        const libraryPath = NativeLibrary.libraryName(nativeDir);
        this.koffi = koffi;
        this.lib = koffi.load(libraryPath);
        this.declare();
    }

    static findNativeDirectory() {
        const override = process.env.VIETASR_NATIVE_DIR;
        if (override && fs.existsSync(override)) {
            return override;
        }
        const candidates = [
            path.resolve(__dirname, "..", "_native"),
            path.resolve(__dirname, "..", "..", "..", "build-core"),
            path.resolve(__dirname, "..", "..", "..", "build-core", "_deps", "onnxruntime-src", "lib"),
        ];
        for (const dir of candidates) {
            if (fs.existsSync(path.join(dir, NativeLibrary.libraryFilename()))) {
                return dir;
            }
        }
        return candidates[0];
    }

    static libraryFilename() {
        if (process.platform === "win32") return "vietasr.dll";
        if (process.platform === "darwin") return "libvietasr.dylib";
        return "libvietasr.so";
    }

    static libraryName(nativeDir) {
        return path.join(nativeDir, NativeLibrary.libraryFilename());
    }

    static preloadSiblings(koffi, dir) {
        if (!fs.existsSync(dir)) return;
        const patterns = process.platform === "win32"
            ? [/^onnxruntime.*\.dll$/]
            : process.platform === "darwin"
                ? [/^libonnxruntime.*\.dylib$/]
                : [/^libonnxruntime\.so(\..*)?$/];
        for (const entry of fs.readdirSync(dir)) {
            for (const re of patterns) {
                if (re.test(entry)) {
                    try {
                        koffi.load(path.join(dir, entry), { global: true });
                    } catch (_) {
                    }
                }
            }
        }
    }

    declare() {
        const koffi = this.koffi;
        this.VietasrPipeline = koffi.opaque("VietasrPipeline");
        this.VietasrSession = koffi.opaque("VietasrSession");
        const Pipeline = koffi.pointer(this.VietasrPipeline);
        const Session = koffi.pointer(this.VietasrSession);

        this.fn = {
            vietasr_pipeline_preset: this.lib.func("vietasr_pipeline_preset", Pipeline, ["str"]),
            vietasr_pipeline_new: this.lib.func("vietasr_pipeline_new", Pipeline, []),
            vietasr_pipeline_add_module: this.lib.func("vietasr_pipeline_add_module", "int",
                [Pipeline, "str", "str"]),
            vietasr_pipeline_set_backend: this.lib.func("vietasr_pipeline_set_backend", "int",
                [Pipeline, "int"]),
            vietasr_pipeline_set_model_dir: this.lib.func("vietasr_pipeline_set_model_dir", "int",
                [Pipeline, "str"]),
            vietasr_pipeline_build: this.lib.func("vietasr_pipeline_build", "int", [Pipeline]),
            vietasr_pipeline_free: this.lib.func("vietasr_pipeline_free", "void", [Pipeline]),

            vietasr_list_modules: this.lib.func("vietasr_list_modules", "str", []),
            vietasr_list_presets: this.lib.func("vietasr_list_presets", "str", []),

            vietasr_session_new: this.lib.func("vietasr_session_new", Session, [Pipeline, "float"]),
            vietasr_session_free: this.lib.func("vietasr_session_free", "void", [Session]),
            vietasr_session_reset: this.lib.func("vietasr_session_reset", "void", [Session]),

            vietasr_accept_waveform_s16: this.lib.func("vietasr_accept_waveform_s16", "int",
                [Session, koffi.pointer("int16_t"), "int"]),
            vietasr_accept_waveform_f32: this.lib.func("vietasr_accept_waveform_f32", "int",
                [Session, koffi.pointer("float"), "int"]),

            vietasr_partial_result: this.lib.func("vietasr_partial_result", "str", [Session]),
            vietasr_result: this.lib.func("vietasr_result", "str", [Session]),
            vietasr_final_result: this.lib.func("vietasr_final_result", "str", [Session]),

            vietasr_transcribe_file: this.lib.func("vietasr_transcribe_file", "str",
                [Pipeline, "str"]),
            vietasr_transcribe_buffer: this.lib.func("vietasr_transcribe_buffer", "str",
                [Pipeline, koffi.pointer("int16_t"), "int", "float"]),

            vietasr_default_cache_dir: this.lib.func("vietasr_default_cache_dir", "str", []),
            vietasr_set_log_level: this.lib.func("vietasr_set_log_level", "void", ["int"]),
            vietasr_version: this.lib.func("vietasr_version", "str", []),
            vietasr_last_error: this.lib.func("vietasr_last_error", "str", []),
        };
    }
}

module.exports = new NativeLibrary();
