from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch


class BaseDataset(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        pass


class BaseCollator(ABC):
    @abstractmethod
    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple:
        pass


class BaseMixer(ABC):
    @abstractmethod
    def mix(
        self,
        utterances: List[Dict[str, Any]],
        sample_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[List], float]:
        pass
