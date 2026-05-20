package io.vietasr;

import com.sun.jna.Pointer;

public final class Session implements AutoCloseable {
    private Pointer handle;

    Session(Pointer handle) {
        this.handle = handle;
    }

    public void reset() {
        NativeLibrary.INSTANCE.vietasr_session_reset(handle);
    }

    public boolean accept(short[] pcm) {
        if (handle == null) {
            throw new VietasrException("Session is closed");
        }
        return NativeLibrary.INSTANCE.vietasr_accept_waveform_s16(handle, pcm, pcm.length) == 1;
    }

    public boolean accept(float[] pcm) {
        if (handle == null) {
            throw new VietasrException("Session is closed");
        }
        return NativeLibrary.INSTANCE.vietasr_accept_waveform_f32(handle, pcm, pcm.length) == 1;
    }

    public Result partial() {
        return Result.fromJson(NativeLibrary.INSTANCE.vietasr_partial_result(handle));
    }

    public Result result() {
        return Result.fromJson(NativeLibrary.INSTANCE.vietasr_result(handle));
    }

    public Result finalResult() {
        return Result.fromJson(NativeLibrary.INSTANCE.vietasr_final_result(handle));
    }

    @Override
    public void close() {
        if (handle != null) {
            NativeLibrary.INSTANCE.vietasr_session_free(handle);
            handle = null;
        }
    }
}
