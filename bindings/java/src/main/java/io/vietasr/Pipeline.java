package io.vietasr;

import com.sun.jna.Pointer;

public final class Pipeline implements AutoCloseable {
    private Pointer handle;

    private Pipeline(Pointer handle) {
        this.handle = handle;
    }

    public static Pipeline preset(String name) {
        Pointer handle = NativeLibrary.INSTANCE.vietasr_pipeline_preset(name);
        if (handle == null) {
            throw new VietasrException(lastError("unknown preset: " + name));
        }
        return new Pipeline(handle);
    }

    public static Pipeline create() {
        return new Pipeline(NativeLibrary.INSTANCE.vietasr_pipeline_new());
    }

    public Pipeline add(String moduleName) {
        return add(moduleName, "{}");
    }

    public Pipeline add(String moduleName, String jsonConfig) {
        int status = NativeLibrary.INSTANCE.vietasr_pipeline_add_module(
                handle, moduleName, jsonConfig == null ? "{}" : jsonConfig);
        if (status != 0) {
            throw new VietasrException(lastError("add(" + moduleName + ") failed: " + status));
        }
        return this;
    }

    public Pipeline setBackend(Backend backend) {
        int status = NativeLibrary.INSTANCE.vietasr_pipeline_set_backend(handle, backend.ordinal());
        if (status != 0) {
            throw new VietasrException(lastError("setBackend failed"));
        }
        return this;
    }

    public Pipeline setModelDir(String dir) {
        int status = NativeLibrary.INSTANCE.vietasr_pipeline_set_model_dir(handle, dir);
        if (status != 0) {
            throw new VietasrException(lastError("setModelDir failed"));
        }
        return this;
    }

    public Pipeline build() {
        int status = NativeLibrary.INSTANCE.vietasr_pipeline_build(handle);
        if (status != 0) {
            throw new VietasrException(lastError("build failed"));
        }
        return this;
    }

    public Result transcribe(String wavPath) {
        String raw = NativeLibrary.INSTANCE.vietasr_transcribe_file(handle, wavPath);
        if (raw == null) {
            throw new VietasrException(lastError("transcribe failed"));
        }
        return Result.fromJson(raw);
    }

    public Result transcribe(short[] pcm, float sampleRate) {
        String raw = NativeLibrary.INSTANCE.vietasr_transcribe_buffer(
                handle, pcm, pcm.length, sampleRate);
        if (raw == null) {
            throw new VietasrException(lastError("transcribe failed"));
        }
        return Result.fromJson(raw);
    }

    public Session stream(float sampleRate) {
        Pointer sessionHandle = NativeLibrary.INSTANCE.vietasr_session_new(handle, sampleRate);
        if (sessionHandle == null) {
            throw new VietasrException(lastError("session creation failed"));
        }
        return new Session(sessionHandle);
    }

    public static String[] listModules() {
        return parseJsonStringArray(NativeLibrary.INSTANCE.vietasr_list_modules());
    }

    public static String[] listPresets() {
        return parseJsonStringArray(NativeLibrary.INSTANCE.vietasr_list_presets());
    }

    public static String version() {
        return NativeLibrary.INSTANCE.vietasr_version();
    }

    public static void setLogLevel(int level) {
        NativeLibrary.INSTANCE.vietasr_set_log_level(level);
    }

    @Override
    public void close() {
        if (handle != null) {
            NativeLibrary.INSTANCE.vietasr_pipeline_free(handle);
            handle = null;
        }
    }

    private static String lastError(String fallback) {
        String e = NativeLibrary.INSTANCE.vietasr_last_error();
        return (e == null || e.isEmpty()) ? fallback : e;
    }

    private static String[] parseJsonStringArray(String raw) {
        if (raw == null || raw.isEmpty()) {
            return new String[0];
        }
        String trimmed = raw.trim();
        if (trimmed.startsWith("[")) {
            trimmed = trimmed.substring(1);
        }
        if (trimmed.endsWith("]")) {
            trimmed = trimmed.substring(0, trimmed.length() - 1);
        }
        if (trimmed.isEmpty()) {
            return new String[0];
        }
        String[] parts = trimmed.split(",");
        String[] out = new String[parts.length];
        for (int i = 0; i < parts.length; ++i) {
            String s = parts[i].trim();
            if (s.startsWith("\"")) {
                s = s.substring(1);
            }
            if (s.endsWith("\"")) {
                s = s.substring(0, s.length() - 1);
            }
            out[i] = s;
        }
        return out;
    }
}
