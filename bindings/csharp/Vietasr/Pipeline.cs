using System;
using System.Collections.Generic;
using System.Text.Json;

namespace Vietasr
{
    public sealed class Pipeline : IDisposable
    {
        private IntPtr handle;

        private Pipeline(IntPtr handle)
        {
            this.handle = handle;
        }

        public static Pipeline Preset(string name)
        {
            IntPtr handle = NativeMethods.vietasr_pipeline_preset(name);
            if (handle == IntPtr.Zero)
            {
                throw new VietasrException(LastError($"unknown preset: {name}"));
            }
            return new Pipeline(handle);
        }

        public static Pipeline Create()
        {
            return new Pipeline(NativeMethods.vietasr_pipeline_new());
        }

        public Pipeline Add(string moduleName, string? jsonConfig = null)
        {
            int status = NativeMethods.vietasr_pipeline_add_module(
                handle, moduleName, jsonConfig ?? "{}");
            if (status != 0)
            {
                throw new VietasrException(LastError($"add({moduleName}) failed: {status}"));
            }
            return this;
        }

        public Pipeline SetBackend(Backend backend)
        {
            int status = NativeMethods.vietasr_pipeline_set_backend(handle, (int)backend);
            if (status != 0)
            {
                throw new VietasrException(LastError("setBackend failed"));
            }
            return this;
        }

        public Pipeline SetModelDir(string dir)
        {
            int status = NativeMethods.vietasr_pipeline_set_model_dir(handle, dir);
            if (status != 0)
            {
                throw new VietasrException(LastError("setModelDir failed"));
            }
            return this;
        }

        public Pipeline Build()
        {
            int status = NativeMethods.vietasr_pipeline_build(handle);
            if (status != 0)
            {
                throw new VietasrException(LastError("build failed"));
            }
            return this;
        }

        public Result Transcribe(string wavPath)
        {
            IntPtr raw = NativeMethods.vietasr_transcribe_file(handle, wavPath);
            if (raw == IntPtr.Zero)
            {
                throw new VietasrException(LastError("transcribe failed"));
            }
            return Result.FromJson(NativeMethods.PtrToString(raw));
        }

        public Result Transcribe(short[] pcm, float sampleRate)
        {
            IntPtr raw = NativeMethods.vietasr_transcribe_buffer(
                handle, pcm, pcm.Length, sampleRate);
            if (raw == IntPtr.Zero)
            {
                throw new VietasrException(LastError("transcribe failed"));
            }
            return Result.FromJson(NativeMethods.PtrToString(raw));
        }

        public Session Stream(float sampleRate)
        {
            IntPtr sessionHandle = NativeMethods.vietasr_session_new(handle, sampleRate);
            if (sessionHandle == IntPtr.Zero)
            {
                throw new VietasrException(LastError("session creation failed"));
            }
            return new Session(sessionHandle);
        }

        public static IReadOnlyList<string> ListModules()
        {
            return ParseJsonStringArray(
                NativeMethods.PtrToString(NativeMethods.vietasr_list_modules()));
        }

        public static IReadOnlyList<string> ListPresets()
        {
            return ParseJsonStringArray(
                NativeMethods.PtrToString(NativeMethods.vietasr_list_presets()));
        }

        public static string Version()
        {
            return NativeMethods.PtrToString(NativeMethods.vietasr_version());
        }

        public static void SetLogLevel(int level)
        {
            NativeMethods.vietasr_set_log_level(level);
        }

        public void Dispose()
        {
            if (handle != IntPtr.Zero)
            {
                NativeMethods.vietasr_pipeline_free(handle);
                handle = IntPtr.Zero;
            }
        }

        private static string LastError(string fallback)
        {
            string e = NativeMethods.PtrToString(NativeMethods.vietasr_last_error());
            return string.IsNullOrEmpty(e) ? fallback : e;
        }

        private static IReadOnlyList<string> ParseJsonStringArray(string raw)
        {
            if (string.IsNullOrEmpty(raw))
            {
                return Array.Empty<string>();
            }
            try
            {
                using var doc = JsonDocument.Parse(raw);
                var list = new List<string>();
                foreach (var element in doc.RootElement.EnumerateArray())
                {
                    var s = element.GetString();
                    if (s != null)
                    {
                        list.Add(s);
                    }
                }
                return list;
            }
            catch (JsonException)
            {
                return Array.Empty<string>();
            }
        }
    }
}
