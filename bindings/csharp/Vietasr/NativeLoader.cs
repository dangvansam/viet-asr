using System;
using System.IO;
using System.Reflection;
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

            // Not bundled: download the native (~67 MB) into the per-user cache.
            string? cache = EnsureCacheDir();
            if (!string.IsNullOrEmpty(cache))
            {
                yield return cache;
            }
        }

        private static string? cacheDir;

        // Downloads + extracts the platform native bundle from the matching
        // GitHub Release into ~/.cache/viet-asr/<version>/.
        private static string? EnsureCacheDir()
        {
            if (cacheDir != null)
            {
                return cacheDir;
            }
            string version = Version();
            string dir = Path.Combine(CacheRoot(), "viet-asr", version);
            string lib = Path.Combine(dir, LibraryFileName("vietasr"));
            if (File.Exists(lib))
            {
                cacheDir = dir;
                return dir;
            }
            try
            {
                Directory.CreateDirectory(dir);
                string asset = $"viet-asr-native-{PlatformKey()}.tar.gz";
                string tarball = Path.Combine(dir, asset);
                string[] urls =
                {
                    $"https://github.com/dangvansam/viet-asr/releases/download/v{version}/{asset}",
                    $"https://github.com/dangvansam/viet-asr/releases/latest/download/{asset}",
                };
                foreach (string url in urls)
                {
                    if (Run("curl", "-fSL", "--retry", "3", "-o", tarball, url)
                        && Run("tar", "-xzf", tarball, "-C", dir)
                        && File.Exists(lib))
                    {
                        File.Delete(tarball);
                        cacheDir = dir;
                        return dir;
                    }
                }
            }
            catch
            {
                // fall through — Resolve reports the failure
            }
            return null;
        }

        private static string Version()
        {
            string version = typeof(NativeLoader).Assembly
                .GetCustomAttribute<AssemblyInformationalVersionAttribute>()
                ?.InformationalVersion ?? "";
            int plus = version.IndexOf('+');
            if (plus >= 0)
            {
                version = version.Substring(0, plus);
            }
            return string.IsNullOrEmpty(version) ? "latest" : version;
        }

        private static string PlatformKey()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return "win-x64";
            }
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                return "darwin-universal2";
            }
            return RuntimeInformation.ProcessArchitecture == Architecture.Arm64
                ? "linux-arm64" : "linux-x64";
        }

        private static string CacheRoot()
        {
            string? xdg = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
            if (!string.IsNullOrEmpty(xdg))
            {
                return xdg;
            }
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".cache");
        }

        private static bool Run(string file, params string[] args)
        {
            try
            {
                var psi = new System.Diagnostics.ProcessStartInfo(file)
                {
                    UseShellExecute = false,
                };
                foreach (string arg in args)
                {
                    psi.ArgumentList.Add(arg);
                }
                using var proc = System.Diagnostics.Process.Start(psi);
                if (proc == null)
                {
                    return false;
                }
                proc.WaitForExit();
                return proc.ExitCode == 0;
            }
            catch
            {
                return false;
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
