package io.vietasr;

public final class Pipeline implements AutoCloseable {
    static {
        System.loadLibrary("onnxruntime");
        System.loadLibrary("vietasr");
        System.loadLibrary("vietasr_jni");
    }

    private long handle;

    private Pipeline(long handle) {
        this.handle = handle;
    }

    public static Pipeline preset(String name) {
        long handle = nativePipelinePreset(name);
        if (handle == 0L) {
            throw new VietasrException(lastError("preset failed: " + name));
        }
        return new Pipeline(handle);
    }

    public static Pipeline newPipeline() {
        long handle = nativePipelineNew();
        return new Pipeline(handle);
    }

    public Pipeline add(String moduleName, String jsonConfig) {
        int status = nativePipelineAddModule(handle, moduleName,
                jsonConfig == null ? "{}" : jsonConfig);
        if (status != 0) {
            throw new VietasrException(lastError("add(" + moduleName + ") failed: " + status));
        }
        return this;
    }

    public Pipeline add(String moduleName) {
        return add(moduleName, null);
    }

    public Pipeline setBackend(Backend backend) {
        int status = nativePipelineSetBackend(handle, backend.ordinal());
        if (status != 0) {
            throw new VietasrException(lastError("setBackend failed"));
        }
        return this;
    }

    public Pipeline setModelDir(String dir) {
        int status = nativePipelineSetModelDir(handle, dir);
        if (status != 0) {
            throw new VietasrException(lastError("setModelDir failed"));
        }
        return this;
    }

    public Pipeline build() {
        int status = nativePipelineBuild(handle);
        if (status != 0) {
            throw new VietasrException(lastError("build failed"));
        }
        return this;
    }

    public Result transcribe(String wavPath) {
        String raw = nativeTranscribeFile(handle, wavPath);
        if (raw == null) {
            throw new VietasrException(lastError("transcribe_file failed"));
        }
        return Result.fromJson(raw);
    }

    public Result transcribe(short[] pcm, float sampleRate) {
        String raw = nativeTranscribeBuffer(handle, pcm, sampleRate);
        if (raw == null) {
            throw new VietasrException(lastError("transcribe_buffer failed"));
        }
        return Result.fromJson(raw);
    }

    public Session stream(float sampleRate) {
        long sessionHandle = nativeSessionNew(handle, sampleRate);
        if (sessionHandle == 0L) {
            throw new VietasrException(lastError("session creation failed"));
        }
        return new Session(sessionHandle);
    }

    public static String[] listModules() {
        return parseJsonStringArray(nativeListModules());
    }

    public static String[] listPresets() {
        return parseJsonStringArray(nativeListPresets());
    }

    public static String version() {
        return nativeVersion();
    }

    public static void setLogLevel(int level) {
        nativeSetLogLevel(level);
    }

    @Override
    public void close() {
        if (handle != 0L) {
            nativePipelineFree(handle);
            handle = 0L;
        }
    }

    private static String lastError(String fallback) {
        String e = nativeLastError();
        return (e == null || e.isEmpty()) ? fallback : e;
    }

    private static String[] parseJsonStringArray(String raw) {
        if (raw == null || raw.isEmpty()) return new String[0];
        String trimmed = raw.trim();
        if (trimmed.startsWith("[")) trimmed = trimmed.substring(1);
        if (trimmed.endsWith("]")) trimmed = trimmed.substring(0, trimmed.length() - 1);
        if (trimmed.isEmpty()) return new String[0];
        String[] parts = trimmed.split(",");
        String[] out = new String[parts.length];
        for (int i = 0; i < parts.length; ++i) {
            String s = parts[i].trim();
            if (s.startsWith("\"")) s = s.substring(1);
            if (s.endsWith("\""))   s = s.substring(0, s.length() - 1);
            out[i] = s;
        }
        return out;
    }

    private static native long nativePipelinePreset(String name);
    private static native long nativePipelineNew();
    private static native int  nativePipelineAddModule(long handle, String name, String jsonConfig);
    private static native int  nativePipelineSetBackend(long handle, int backend);
    private static native int  nativePipelineSetModelDir(long handle, String dir);
    private static native int  nativePipelineBuild(long handle);
    private static native void nativePipelineFree(long handle);
    private static native long nativeSessionNew(long pipelineHandle, float sampleRate);
    private static native String nativeTranscribeFile(long handle, String path);
    private static native String nativeTranscribeBuffer(long handle, short[] pcm, float sampleRate);
    private static native String nativeListModules();
    private static native String nativeListPresets();
    private static native String nativeLastError();
    private static native String nativeVersion();
    private static native void   nativeSetLogLevel(int level);
}
