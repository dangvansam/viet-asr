import json
from pathlib import Path
from typing import Optional, Union

from vietasr.ffi import native
from vietasr.result import Result
from vietasr.session import Session


class PipelineError(RuntimeError):
    pass


class Pipeline:
    def __init__(self, handle):
        self.handle = handle
        self.ffi = native.ffi

    @classmethod
    def preset(cls, name: str) -> "Pipeline":
        handle = native.lib.vietasr_pipeline_preset(name.encode("utf-8"))
        if handle == native.ffi.NULL:
            raise PipelineError(cls.last_error() or f"unknown preset: {name}")
        return cls(handle)

    @classmethod
    def new(cls) -> "Pipeline":
        return cls(native.lib.vietasr_pipeline_new())

    def add(self, module_name: str, config: Optional[dict] = None) -> "Pipeline":
        json_config = json.dumps(config or {})
        status = native.lib.vietasr_pipeline_add_module(
            self.handle,
            module_name.encode("utf-8"),
            json_config.encode("utf-8"),
        )
        if status != 0:
            raise PipelineError(self.last_error() or f"add({module_name}) failed: {status}")
        return self

    def set_backend(self, backend: str) -> "Pipeline":
        mapping = {"auto": 0, "onnx": 1, "coreml": 2}
        status = native.lib.vietasr_pipeline_set_backend(self.handle, mapping[backend])
        if status != 0:
            raise PipelineError(self.last_error() or f"set_backend({backend}) failed")
        return self

    def set_model_dir(self, model_dir: Union[str, Path]) -> "Pipeline":
        status = native.lib.vietasr_pipeline_set_model_dir(
            self.handle, str(model_dir).encode("utf-8")
        )
        if status != 0:
            raise PipelineError(self.last_error() or "set_model_dir failed")
        return self

    def build(self) -> "Pipeline":
        status = native.lib.vietasr_pipeline_build(self.handle)
        if status != 0:
            raise PipelineError(self.last_error() or "build failed")
        return self

    def transcribe(self, source: Union[str, Path, "np.ndarray"], sample_rate: float = 16000.0) -> Result:
        if isinstance(source, (str, Path)):
            raw = native.lib.vietasr_transcribe_file(self.handle, str(source).encode("utf-8"))
            if raw == native.ffi.NULL:
                raise PipelineError(self.last_error() or "transcribe_file failed")
            return Result.from_json(self.ffi.string(raw).decode("utf-8"))
        import numpy as np
        array = np.asarray(source).astype(np.int16)
        buffer = self.ffi.cast("const short*", self.ffi.from_buffer(array))
        raw = native.lib.vietasr_transcribe_buffer(self.handle, buffer, array.size, sample_rate)
        if raw == native.ffi.NULL:
            raise PipelineError(self.last_error() or "transcribe_buffer failed")
        return Result.from_json(self.ffi.string(raw).decode("utf-8"))

    def stream(self, sample_rate: float = 16000.0) -> Session:
        handle = native.lib.vietasr_session_new(self.handle, sample_rate)
        if handle == native.ffi.NULL:
            raise PipelineError(self.last_error() or "session creation failed")
        return Session(handle, self.ffi)

    @staticmethod
    def last_error() -> str:
        raw = native.lib.vietasr_last_error()
        if raw == native.ffi.NULL:
            return ""
        return native.ffi.string(raw).decode("utf-8")

    def close(self) -> None:
        if self.handle:
            native.lib.vietasr_pipeline_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
