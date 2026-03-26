from typing import Optional, Union

import torch


class DeviceManager:
    def __init__(
        self,
        device: str = "cpu",
        cuda_id: int = -1,
    ):
        self._requested_device = device
        self._cuda_id = cuda_id
        self._device = self._resolve_device()

    def _resolve_device(self) -> torch.device:
        if self._cuda_id >= 0 and torch.cuda.is_available():
            return torch.device(f"cuda:{self._cuda_id}")

        if self._requested_device == "cuda" and self._cuda_id >= 0:
            device_str = f"cuda:{self._cuda_id}"
        else:
            device_str = self._requested_device

        if device_str.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")

        return torch.device(device_str)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def device_str(self) -> str:
        return str(self._device)

    @property
    def is_cuda(self) -> bool:
        return self._device.type == "cuda"

    @property
    def map_location(self) -> torch.device:
        return self._device

    def to_device(self, tensor_or_model: Union[torch.Tensor, torch.nn.Module]):
        return tensor_or_model.to(self._device)

    @classmethod
    def from_config(cls, config) -> "DeviceManager":
        device = getattr(config, "device", "cpu")
        cuda_id = getattr(config, "cuda_id", -1)
        return cls(device=device, cuda_id=cuda_id)

    @classmethod
    def auto(cls, prefer_cuda: bool = True) -> "DeviceManager":
        if prefer_cuda and torch.cuda.is_available():
            return cls(device="cuda", cuda_id=0)
        return cls(device="cpu", cuda_id=-1)
