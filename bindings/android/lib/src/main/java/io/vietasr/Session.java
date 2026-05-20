package io.vietasr;

public final class Session implements AutoCloseable {
    private long handle;

    Session(long handle) {
        this.handle = handle;
    }

    public void reset() {
        nativeReset(handle);
    }

    public boolean accept(short[] pcm) {
        return nativeAcceptS16(handle, pcm) == 1;
    }

    public boolean accept(float[] pcm) {
        return nativeAcceptF32(handle, pcm) == 1;
    }

    public Result partial() {
        return Result.fromJson(nativePartial(handle));
    }

    public Result result() {
        return Result.fromJson(nativeResult(handle));
    }

    public Result finalResult() {
        return Result.fromJson(nativeFinal(handle));
    }

    @Override
    public void close() {
        if (handle != 0L) {
            nativeFree(handle);
            handle = 0L;
        }
    }

    private static native void   nativeReset(long handle);
    private static native int    nativeAcceptS16(long handle, short[] pcm);
    private static native int    nativeAcceptF32(long handle, float[] pcm);
    private static native String nativePartial(long handle);
    private static native String nativeResult(long handle);
    private static native String nativeFinal(long handle);
    private static native void   nativeFree(long handle);
}
