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
        int  vietasr_pipeline_set_model_dir(VietasrPipeline*, const char*);
        int  vietasr_pipeline_build(VietasrPipeline*);
        void vietasr_pipeline_free(VietasrPipeline*);
        const char* vietasr_list_modules(void);
        const char* vietasr_list_presets(void);

        VietasrSession* vietasr_session_new(VietasrPipeline*, float);
        void            vietasr_session_free(VietasrSession*);
        void            vietasr_session_reset(VietasrSession*);

        int vietasr_accept_waveform_s16(VietasrSession*, const short*, int);
        int vietasr_accept_waveform_f32(VietasrSession*, const float*, int);

        const char* vietasr_partial_result(VietasrSession*);
        const char* vietasr_result(VietasrSession*);
        const char* vietasr_final_result(VietasrSession*);

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

        for candidate_dir in NativeLibrary.candidate_native_dirs():
            candidate = candidate_dir / name
            if candidate.exists():
                NativeLibrary.preload_siblings(candidate_dir)
                return str(candidate)
        return name

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
