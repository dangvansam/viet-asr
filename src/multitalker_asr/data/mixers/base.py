from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseMixer(ABC):
    @abstractmethod
    def mix(
        self,
        utterances: List[Dict[str, Any]],
        sample_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[List], float]:
        pass
