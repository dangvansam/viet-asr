from typing import Any, Dict, List, Tuple

import torch

from .multitalker import MultitalkerCollator


class MultitaskCollator(MultitalkerCollator):
    """Extends MultitalkerCollator to batch task_labels into {task: Tensor[B]}."""

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple:
        # Delegate audio/text/mask collation to parent
        base_result = super().__call__(batch)

        # Collate task_labels
        task_labels = {}
        if batch and "task_labels" in batch[0]:
            task_keys = batch[0]["task_labels"].keys()
            for task in task_keys:
                values = []
                for item in batch:
                    labels = item.get("task_labels", {})
                    values.append(labels.get(task, 0))
                task_labels[task] = torch.tensor(values, dtype=torch.long)

        return base_result + (task_labels,)
