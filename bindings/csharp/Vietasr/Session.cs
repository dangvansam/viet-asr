using System;

namespace Vietasr
{
    public sealed class Session : IDisposable
    {
        private IntPtr handle;

        internal Session(IntPtr handle)
        {
            this.handle = handle;
        }

        public void Reset()
        {
            NativeMethods.vietasr_session_reset(handle);
        }

        public bool Accept(short[] pcm)
        {
            if (handle == IntPtr.Zero)
            {
                throw new VietasrException("Session is closed");
            }
            return NativeMethods.vietasr_accept_waveform_s16(handle, pcm, pcm.Length) == 1;
        }

        public bool Accept(float[] pcm)
        {
            if (handle == IntPtr.Zero)
            {
                throw new VietasrException("Session is closed");
            }
            return NativeMethods.vietasr_accept_waveform_f32(handle, pcm, pcm.Length) == 1;
        }

        public Result Partial()
        {
            return Result.FromJson(NativeMethods.PtrToString(
                NativeMethods.vietasr_partial_result(handle)));
        }

        public Result GetResult()
        {
            return Result.FromJson(NativeMethods.PtrToString(
                NativeMethods.vietasr_result(handle)));
        }

        public Result Final()
        {
            return Result.FromJson(NativeMethods.PtrToString(
                NativeMethods.vietasr_final_result(handle)));
        }

        public void Dispose()
        {
            if (handle != IntPtr.Zero)
            {
                NativeMethods.vietasr_session_free(handle);
                handle = IntPtr.Zero;
            }
        }
    }
}
