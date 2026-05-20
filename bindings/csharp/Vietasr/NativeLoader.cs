using System;
using System.IO;
using System.Runtime.InteropServices;

namespace Vietasr
{
    internal static class NativeLoader
    {
        private static bool registered;
        private static readonly object Gate = new object();

        internal static void Ensure()
        {
            if (registered)
            {
                return;
            }
            lock (Gate)
            {
                if (registered)
                {
                    return;
                }
                NativeLibrary.SetDllImportResolver(
                    typeof(NativeLoader).Assembly, Resolve);
                registered = true;
            }
        }

        private static IntPtr Resolve(string libraryName, System.Reflection.Assembly assembly,
            DllImportSearchPath? searchPath)
        {
            if (libraryName != "vietasr")
            {
                return IntPtr.Zero;
            }

            string fileName = LibraryFileName("vietasr");
            string ortName = LibraryFileName("onnxruntime");

            foreach (string dir in CandidateDirectories())
            {
                string ortPath = Path.Combine(dir, ortName);
                if (File.Exists(ortPath))
                {
                    NativeLibrary.TryLoad(ortPath, out _);
                }
                string libPath = Path.Combine(dir, fileName);
                if (File.Exists(libPath) && NativeLibrary.TryLoad(libPath, out IntPtr handle))
                {
                    return handle;
                }
            }
            return IntPtr.Zero;
        }

        private static System.Collections.Generic.IEnumerable<string> CandidateDirectories()
        {
            string? overrideDir = Environment.GetEnvironmentVariable("VIETASR_NATIVE_DIR");
            if (!string.IsNullOrEmpty(overrideDir))
            {
                yield return overrideDir;
            }

            string baseDir = AppContext.BaseDirectory;
            yield return baseDir;
            yield return Path.Combine(baseDir, RuntimeRelativePath());

            string? asmDir = Path.GetDirectoryName(
                typeof(NativeLoader).Assembly.Location);
            if (!string.IsNullOrEmpty(asmDir))
            {
                yield return asmDir;
                yield return Path.Combine(asmDir, RuntimeRelativePath());
            }
        }

        private static string RuntimeRelativePath()
        {
            string rid;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                rid = "win-x64";
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                rid = RuntimeInformation.ProcessArchitecture == Architecture.Arm64
                    ? "osx-arm64" : "osx-x64";
            }
            else
            {
                rid = RuntimeInformation.ProcessArchitecture == Architecture.Arm64
                    ? "linux-arm64" : "linux-x64";
            }
            return Path.Combine("runtimes", rid, "native");
        }

        private static string LibraryFileName(string stem)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return stem + ".dll";
            }
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                return "lib" + stem + ".dylib";
            }
            return "lib" + stem + ".so";
        }
    }
}
