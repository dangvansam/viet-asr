import os
import sys
from pathlib import Path

from cffi import FFI


class NativeLibrary:
    def __init__(self):
        self.ffi = FFI()
        self.ffi.cdef(self.cdef_source())
        self.lib = self.ffi.dlopen(self.resolve_library_path())

    @staticmethod
    def cdef_source() -> str:
        return """
        typedef struct VietasrPipeline VietasrPipeline;
        typedef struct VietasrSession  VietasrSession;

        VietasrPipeline* vietasr_pipeline_preset(const char* name);
        VietasrPipeline* vietasr_pipeline_new(void);
        int  vietasr_pipeline_add_module(VietasrPipeline*, const char*, const char*);
        int  vietasr_pipeline_set_backend(VietasrPipeline*, int);
        int  vietasr_pipeline_set_execution_provider(VietasrPipeline*, int, int);
        int  vietasr_pipeline_set_model_dir(VietasrPipeline*, const char*);
        int  vietasr_pipeline_build(VietasrPipeline*);
        void vietasr_pipeline_free(VietasrPipeline*);
        const char* vietasr_list_modules(void);
        const char* vietasr_list_presets(void);
        const char* vietasr_available_execution_providers(void);

        VietasrSession* vietasr_session_new(VietasrPipeline*, float);
        void            vietasr_session_free(VietasrSession*);
        void            vietasr_session_reset(VietasrSession*);

        int vietasr_accept_waveform_s16(VietasrSession*, const short*, int);
        int vietasr_accept_waveform_f32(VietasrSession*, const float*, int);

        const char* vietasr_partial_result(VietasrSession*);
        const char* vietasr_result(VietasrSession*);
        const char* vietasr_final_result(VietasrSession*);
        const char* vietasr_active_execution_provider(VietasrSession*);

        const char* vietasr_transcribe_file(VietasrPipeline*, const char*);
        const char* vietasr_transcribe_buffer(VietasrPipeline*, const short*, int, float);

        int  vietasr_ensure_models(VietasrPipeline*);
        const char* vietasr_default_cache_dir(void);

        void        vietasr_set_log_level(int);
        const char* vietasr_version(void);
        const char* vietasr_last_error(void);
        """

    @staticmethod
    def resolve_library_path() -> str:
        override = os.environ.get("VIETASR_LIBRARY_PATH")
        if override:
            return override

        if sys.platform == "win32":
            name = "vietasr.dll"
        elif sys.platform == "darwin":
            name = "libvietasr.dylib"
        else:
            name = "libvietasr.so"

        # Opt-in GPU variant: VIETASR_ONNX_VARIANT=cuda|directml loads a separate
        # native bundle downloaded from the GitHub release, in preference to the
        # CPU native bundled in the wheel. Mirrors the Node/Rust/Go/.NET
        # resolvers — the GPU bundle is too large to ship inside a PyPI wheel.
        variant_dir = NativeLibrary._gpu_variant_dir(name)
        if variant_dir is not None:
            NativeLibrary.preload_siblings(variant_dir)
            return str(variant_dir / name)

        for candidate_dir in NativeLibrary.candidate_native_dirs():
            candidate = candidate_dir / name
            if candidate.exists():
                NativeLibrary.preload_siblings(candidate_dir)
                return str(candidate)
        return name

    @staticmethod
    def _gpu_variant_dir(lib_name: str):
        """Directory of the GPU native bundle for VIETASR_ONNX_VARIANT, fetching
        it on first use. Returns None when no (usable) GPU variant is requested,
        so the caller falls back to the bundled CPU native."""
        variant = os.environ.get("VIETASR_ONNX_VARIANT", "cpu").strip().lower()
        if variant in ("", "cpu"):
            return None

        key = NativeLibrary._gpu_platform_key(variant)
        if key is None:
            print(
                f"viet-asr: VIETASR_ONNX_VARIANT={variant} is not available for "
                "this platform; using the CPU native",
                file=sys.stderr,
            )
            return None

        from vietasr.version import __version__

        cache = (
            Path(NativeLibrary._cache_root()) / "viet-asr" / __version__ / key
        )
        if (cache / lib_name).exists():
            return cache

        try:
            NativeLibrary._download_native(key, __version__, cache)
        except Exception as exc:  # noqa: BLE001 — fall back to CPU on any failure
            print(
                f"viet-asr: could not fetch the {key} native bundle ({exc}); "
                "using the CPU native",
                file=sys.stderr,
            )
            return None
        return cache if (cache / lib_name).exists() else None

    @staticmethod
    def _gpu_platform_key(variant: str):
        import platform

        machine = platform.machine().lower()
        is_x64 = machine in ("x86_64", "amd64")
        if sys.platform.startswith("linux") and is_x64:
            base = "linux-x64"
        elif sys.platform == "win32" and is_x64:
            base = "win-x64"
        else:
            return None  # GPU bundles exist only for linux-x64 / win-x64
        if variant == "cuda":
            return base + "-gpu"
        if variant == "directml" and base == "win-x64":
            return "win-x64-directml"
        return None

    @staticmethod
    def _cache_root() -> str:
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            return xdg
        return str(Path.home() / ".cache")

    @staticmethod
    def _download_native(key: str, version: str, dest: Path) -> None:
        import shutil
        import tarfile
        import tempfile
        import urllib.request

        repo = "dangvansam/viet-asr"
        asset = f"viet-asr-native-{key}.tar.gz"
        urls = [
            f"https://github.com/{repo}/releases/download/v{version}/{asset}",
            f"https://github.com/{repo}/releases/latest/download/{asset}",
        ]
        dest.mkdir(parents=True, exist_ok=True)
        last_error = None
        for url in urls:
            tarball = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".tar.gz", delete=False
                ) as tmp:
                    tarball = tmp.name
                    with urllib.request.urlopen(url) as resp:  # noqa: S310
                        shutil.copyfileobj(resp, tmp)
                with tarfile.open(tarball) as archive:
                    archive.extractall(dest)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            finally:
                if tarball and os.path.exists(tarball):
                    os.unlink(tarball)
        raise RuntimeError(f"download failed for {asset}: {last_error}")

    @staticmethod
    def candidate_native_dirs():
        seen = set()
        for raw in NativeLibrary._candidate_dirs_raw():
            key = str(raw.resolve()) if raw.exists() else str(raw)
            if key in seen:
                continue
            seen.add(key)
            yield raw

    @staticmethod
    def _candidate_dirs_raw():
        here = Path(__file__).parent
        yield here / "_native"
        yield here.parent / "vietasr" / "_native"

        import importlib.util
        spec = importlib.util.find_spec("vietasr")
        if spec is not None and spec.submodule_search_locations:
            for path in spec.submodule_search_locations:
                yield Path(path) / "_native"

        for site_root in [
            Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
            Path(sys.prefix) / "Lib" / "site-packages",
        ]:
            yield site_root / "vietasr" / "_native"

    @staticmethod
    def preload_siblings(directory: Path) -> None:
        import ctypes
        if sys.platform == "win32":
            try:
                os.add_dll_directory(str(directory))
            except (AttributeError, OSError):
                pass
            patterns = ["onnxruntime*.dll"]
        elif sys.platform == "darwin":
            patterns = ["libonnxruntime*.dylib"]
        else:
            patterns = ["libonnxruntime.so*"]
        for pattern in patterns:
            for sibling in directory.glob(pattern):
                try:
                    ctypes.CDLL(str(sibling), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


native = NativeLibrary()
