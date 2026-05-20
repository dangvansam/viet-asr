import numpy as np

from vietasr.ffi import native
from vietasr.result import Result


class Session:
    def __init__(self, handle, ffi):
        self.handle = handle
        self.ffi = ffi

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def reset(self) -> None:
        native.lib.vietasr_session_reset(self.handle)

    def accept(self, pcm) -> bool:
        if isinstance(pcm, (bytes, bytearray, memoryview)):
            buffer = self.ffi.from_buffer("short[]", pcm)
            length = len(pcm) // 2
            status = native.lib.vietasr_accept_waveform_s16(self.handle, buffer, length)
            return status == 1
        array = np.asarray(pcm)
        if array.dtype == np.int16:
            buffer = self.ffi.cast("const short*", self.ffi.from_buffer(array))
            status = native.lib.vietasr_accept_waveform_s16(self.handle, buffer, array.size)
        else:
            array = array.astype(np.float32)
            buffer = self.ffi.cast("const float*", self.ffi.from_buffer(array))
            status = native.lib.vietasr_accept_waveform_f32(self.handle, buffer, array.size)
        return status == 1

    def partial(self) -> Result:
        return Result.from_json(self.ffi.string(native.lib.vietasr_partial_result(self.handle)).decode("utf-8"))

    def result(self) -> Result:
        return Result.from_json(self.ffi.string(native.lib.vietasr_result(self.handle)).decode("utf-8"))

    def final(self) -> Result:
        return Result.from_json(self.ffi.string(native.lib.vietasr_final_result(self.handle)).decode("utf-8"))

    def close(self) -> None:
        if self.handle:
            native.lib.vietasr_session_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
