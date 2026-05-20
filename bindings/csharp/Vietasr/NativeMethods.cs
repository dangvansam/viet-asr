using System;
using System.Runtime.InteropServices;

namespace Vietasr
{
    internal static class NativeMethods
    {
        private const string Lib = "vietasr";

        static NativeMethods()
        {
            NativeLoader.Ensure();
        }

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vietasr_pipeline_preset(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string name);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vietasr_pipeline_new();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int vietasr_pipeline_add_module(
            IntPtr pipeline,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string moduleName,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string jsonConfig);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int vietasr_pipeline_set_backend(IntPtr pipeline, int backend);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int vietasr_pipeline_set_model_dir(
            IntPtr pipeline,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string modelDir);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int vietasr_pipeline_build(IntPtr pipeline);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void vietasr_pipeline_free(IntPtr pipeline);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vietasr_list_modules();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vietasr_list_presets();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vietasr_session_new(IntPtr pipeline, float sampleRate);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void vietasr_session_free(IntPtr session);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void vietasr_session_reset(IntPtr session);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int vietasr_accept_waveform_s16(
            IntPtr session, short[] pcm, int len);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int vietasr_accept_waveform_f32(
            IntPtr session, float[] pcm, int len);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vietasr_partial_result(IntPtr session);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vietasr_result(IntPtr session);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vietasr_final_result(IntPtr session);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vietasr_transcribe_file(
            IntPtr pipeline,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string wavPath);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vietasr_transcribe_buffer(
            IntPtr pipeline, short[] pcm, int len, float sampleRate);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vietasr_default_cache_dir();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void vietasr_set_log_level(int level);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vietasr_version();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vietasr_last_error();

        internal static string PtrToString(IntPtr ptr)
        {
            return ptr == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(ptr) ?? string.Empty;
        }
    }
}
