const native = require("./native");
const { Result } = require("./result");
const { Session } = require("./session");

const BACKEND = { auto: 0, onnx: 1, coreml: 2 };

class PipelineError extends Error {}

class Pipeline {
    constructor(handle) {
        this.handle = handle;
    }

    static preset(name) {
        const handle = native.fn.vietasr_pipeline_preset(name);
        if (!handle) {
            throw new PipelineError(Pipeline.lastError() || `unknown preset: ${name}`);
        }
        return new Pipeline(handle);
    }

    static new() {
        const handle = native.fn.vietasr_pipeline_new();
        return new Pipeline(handle);
    }

    add(moduleName, config) {
        const json = JSON.stringify(config || {});
        const status = native.fn.vietasr_pipeline_add_module(this.handle, moduleName, json);
        if (status !== 0) {
            throw new PipelineError(Pipeline.lastError() || `add(${moduleName}) failed: ${status}`);
        }
        return this;
    }

    setBackend(backend) {
        if (!(backend in BACKEND)) {
            throw new PipelineError(`unknown backend: ${backend}`);
        }
        const status = native.fn.vietasr_pipeline_set_backend(this.handle, BACKEND[backend]);
        if (status !== 0) {
            throw new PipelineError(Pipeline.lastError() || `setBackend(${backend}) failed`);
        }
        return this;
    }

    setModelDir(dir) {
        const status = native.fn.vietasr_pipeline_set_model_dir(this.handle, dir);
        if (status !== 0) {
            throw new PipelineError(Pipeline.lastError() || "setModelDir failed");
        }
        return this;
    }

    build() {
        const status = native.fn.vietasr_pipeline_build(this.handle);
        if (status !== 0) {
            throw new PipelineError(Pipeline.lastError() || "build failed");
        }
        return this;
    }

    transcribe(source, sampleRate) {
        if (typeof source === "string") {
            const raw = native.fn.vietasr_transcribe_file(this.handle, source);
            if (!raw) {
                throw new PipelineError(Pipeline.lastError() || "transcribe_file failed");
            }
            return Result.fromJson(raw);
        }
        if (source instanceof Int16Array) {
            const sr = sampleRate || 16000;
            const raw = native.fn.vietasr_transcribe_buffer(
                this.handle, source, source.length, sr);
            if (!raw) {
                throw new PipelineError(Pipeline.lastError() || "transcribe_buffer failed");
            }
            return Result.fromJson(raw);
        }
        throw new TypeError("transcribe() expects a file path or Int16Array");
    }

    stream(sampleRate) {
        const sr = sampleRate || 16000;
        const handle = native.fn.vietasr_session_new(this.handle, sr);
        if (!handle) {
            throw new PipelineError(Pipeline.lastError() || "session creation failed");
        }
        return new Session(handle);
    }

    static listModules() {
        return JSON.parse(native.fn.vietasr_list_modules() || "[]");
    }

    static listPresets() {
        return JSON.parse(native.fn.vietasr_list_presets() || "[]");
    }

    static lastError() {
        return native.fn.vietasr_last_error() || "";
    }

    static version() {
        return native.fn.vietasr_version() || "";
    }

    static setLogLevel(level) {
        const levels = { trace: 0, debug: 1, info: 2, warn: 3, error: 4, off: 5 };
        const value = typeof level === "string" ? levels[level] : level;
        native.fn.vietasr_set_log_level(value);
    }

    close() {
        if (this.handle) {
            native.fn.vietasr_pipeline_free(this.handle);
            this.handle = null;
        }
    }
}

module.exports = { Pipeline, PipelineError, Session, Result };
