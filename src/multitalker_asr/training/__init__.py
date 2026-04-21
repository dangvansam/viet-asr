from .base import BaseTrainer
from .trainer import MultitalkerTrainer
from .callbacks import PrintLossCallback

__all__ = [
    "BaseTrainer",
    "MultitalkerTrainer",
    "PrintLossCallback",
]

# Lazy imports to avoid circular dependency (models → training.losses → training → models)
def __getattr__(name):
    if name == "CurriculumTrainer":
        from .curriculum_trainer import CurriculumTrainer
        return CurriculumTrainer
    if name == "CurriculumPhase":
        from .curriculum_trainer import CurriculumPhase
        return CurriculumPhase
    if name == "MultiTaskLoss":
        from .losses import MultiTaskLoss
        return MultiTaskLoss
    if name == "DynamicLossWeighting":
        from .losses import DynamicLossWeighting
        return DynamicLossWeighting
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
