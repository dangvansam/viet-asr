package io.vietasr;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

interface VietasrNative extends Library {
    Pointer vietasr_pipeline_preset(String name);
    Pointer vietasr_pipeline_new();
    int vietasr_pipeline_add_module(Pointer pipeline, String moduleName, String jsonConfig);
    int vietasr_pipeline_set_backend(Pointer pipeline, int backend);
    int vietasr_pipeline_set_model_dir(Pointer pipeline, String modelDir);
    int vietasr_pipeline_build(Pointer pipeline);
    void vietasr_pipeline_free(Pointer pipeline);

    String vietasr_list_modules();
    String vietasr_list_presets();

    Pointer vietasr_session_new(Pointer pipeline, float sampleRate);
    void vietasr_session_free(Pointer session);
    void vietasr_session_reset(Pointer session);

    int vietasr_accept_waveform_s16(Pointer session, short[] pcm, int len);
    int vietasr_accept_waveform_f32(Pointer session, float[] pcm, int len);

    String vietasr_partial_result(Pointer session);
    String vietasr_result(Pointer session);
    String vietasr_final_result(Pointer session);

    String vietasr_transcribe_file(Pointer pipeline, String wavPath);
    String vietasr_transcribe_buffer(Pointer pipeline, short[] pcm, int len, float sampleRate);

    String vietasr_default_cache_dir();
    void vietasr_set_log_level(int level);
    String vietasr_version();
    String vietasr_last_error();
}

final class NativeLibrary {
    static final VietasrNative INSTANCE = load();

    private NativeLibrary() {
    }

    private static VietasrNative load() {
        String dir = resolveNativeDirectory();
        if (dir != null) {
            preloadSiblings(dir);
            File lib = new File(dir, libraryFilename());
            if (lib.exists()) {
                return Native.load(lib.getAbsolutePath(), VietasrNative.class);
            }
        }
        return Native.load("vietasr", VietasrNative.class);
    }

    private static String resolveNativeDirectory() {
        String override = System.getenv("VIETASR_NATIVE_DIR");
        if (override != null && new File(override).isDirectory()) {
            return override;
        }
        List<String> candidates = new ArrayList<>();
        String here = NativeLibrary.class.getProtectionDomain()
                .getCodeSource().getLocation().getPath();
        Path base = Paths.get(here).getParent();
        if (base != null) {
            candidates.add(base.resolve("_native").toString());
            if (base.getParent() != null) {
                candidates.add(base.getParent().resolve("_native").toString());
            }
        }
        candidates.add("_native");
        for (String c : candidates) {
            if (new File(c, libraryFilename()).exists()) {
                return c;
            }
        }
        return null;
    }

    private static String libraryFilename() {
        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("win")) return "vietasr.dll";
        if (os.contains("mac")) return "libvietasr.dylib";
        return "libvietasr.so";
    }

    private static void preloadSiblings(String dir) {
        String os = System.getProperty("os.name").toLowerCase();
        String prefix = os.contains("win") ? "onnxruntime" : "libonnxruntime";
        File folder = new File(dir);
        File[] files = folder.listFiles();
        if (files == null) return;
        for (File f : files) {
            if (f.getName().startsWith(prefix)) {
                try {
                    System.load(f.getAbsolutePath());
                } catch (Throwable ignored) {
                }
            }
        }
    }
}
