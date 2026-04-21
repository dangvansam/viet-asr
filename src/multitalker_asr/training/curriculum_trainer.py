from enum import Enum
from typing import Optional

from loguru import logger

from ..configs import ModelConfig, TrainingConfig
from ..configs.multitask import MultiTaskConfig
from ..data import DataLoaderFactory
from ..models.multitask_model import MultitalkerMultiTaskModel
from .losses import DynamicLossWeighting
from .trainer import MultitalkerTrainer


class CurriculumPhase(str, Enum):
    ASR = "asr"
    MULTITALKER = "multitalker"
    PARALINGUISTIC = "paralinguistic"


# Default configs per curriculum phase
PHASE_DEFAULTS = {
    CurriculumPhase.ASR: {
        "num_frozen_layers": 18,
        "learning_rate": 1e-4,
        "enable_prompt_loss": False,
        "enable_sortformer": False,
    },
    CurriculumPhase.MULTITALKER: {
        "num_frozen_layers": 12,
        "learning_rate": 1e-5,
        "enable_prompt_loss": False,
        "enable_sortformer": True,
    },
    CurriculumPhase.PARALINGUISTIC: {
        "num_frozen_layers": 0,
        "learning_rate": 5e-6,
        "enable_prompt_loss": True,
        "enable_sortformer": True,
    },
}


class CurriculumTrainer(MultitalkerTrainer):
    """Extends MultitalkerTrainer with 3-phase curriculum training.

    Phase 1 (ASR): Freeze lower encoder, RNNT loss only, single-speaker data
    Phase 2 (Multitalker): Activate Sortformer + kernels, synthetic overlap data
    Phase 3 (Paralinguistic): Activate prompt tokens + CE loss, annotated data
    """

    def __init__(
        self,
        model: MultitalkerMultiTaskModel,
        train_cfg: TrainingConfig,
        model_cfg: Optional[ModelConfig] = None,
        multitask_cfg: Optional[MultiTaskConfig] = None,
    ):
        super().__init__(model, train_cfg, model_cfg)
        self._multitask_model = model
        self._multitask_cfg = multitask_cfg or MultiTaskConfig()
        self._curriculum_phase = CurriculumPhase.ASR
        self._dynamic_weighting: Optional[DynamicLossWeighting] = None

        # Parse curriculum phase from train_cfg
        phase_str = getattr(train_cfg, "curriculum_phase", "asr")
        try:
            self._curriculum_phase = CurriculumPhase(phase_str)
        except ValueError:
            logger.warning(f"Unknown curriculum phase '{phase_str}', defaulting to ASR")
            self._curriculum_phase = CurriculumPhase.ASR

    @property
    def curriculum_phase(self) -> CurriculumPhase:
        return self._curriculum_phase

    def setup(self) -> None:
        """Setup training with curriculum phase-specific configuration."""
        super().setup()
        self._configure_curriculum_phase()
        logger.info(f"Curriculum phase: {self._curriculum_phase.value}")

    def _configure_curriculum_phase(self) -> None:
        """Apply phase-specific freezing, loss, and data configuration."""
        phase = self._curriculum_phase
        defaults = PHASE_DEFAULTS[phase]

        if phase == CurriculumPhase.ASR:
            self._configure_phase_asr(defaults)
        elif phase == CurriculumPhase.MULTITALKER:
            self._configure_phase_multitalker(defaults)
        elif phase == CurriculumPhase.PARALINGUISTIC:
            self._configure_phase_paralinguistic(defaults)

    def _configure_phase_asr(self, defaults: dict) -> None:
        """Phase 1: Vietnamese ASR fine-tuning on single-speaker data."""
        num_frozen = getattr(self._train_cfg, "num_frozen_layers", defaults["num_frozen_layers"])
        self._multitask_model.freeze_encoder_layers(num_frozen)

        # Freeze prompt embedding (not used in Phase 1)
        for param in self._multitask_model.prompt_embedding.parameters():
            param.requires_grad = False
        if self._multitask_model._prompt_classifier is not None:
            for param in self._multitask_model._prompt_classifier.parameters():
                param.requires_grad = False

        logger.info("Phase ASR: prompt layers frozen, RNNT loss only")

    def _configure_phase_multitalker(self, defaults: dict) -> None:
        """Phase 2: Multi-talker adaptation with synthetic overlap data."""
        num_frozen = getattr(self._train_cfg, "num_frozen_layers", defaults["num_frozen_layers"])
        self._multitask_model.freeze_encoder_layers(num_frozen)

        # Prompt layers still frozen in Phase 2
        for param in self._multitask_model.prompt_embedding.parameters():
            param.requires_grad = False
        if self._multitask_model._prompt_classifier is not None:
            for param in self._multitask_model._prompt_classifier.parameters():
                param.requires_grad = False

        logger.info("Phase Multitalker: Sortformer active, prompt layers frozen")

    def _configure_phase_paralinguistic(self, defaults: dict) -> None:
        """Phase 3: Paralinguistic SFT with annotated metadata corpora."""
        # Unfreeze prompt layers
        for param in self._multitask_model.prompt_embedding.parameters():
            param.requires_grad = True
        if self._multitask_model._prompt_classifier is not None:
            for param in self._multitask_model._prompt_classifier.parameters():
                param.requires_grad = True

        # Initialize dynamic loss weighting
        use_dynamic = getattr(self._train_cfg, "use_dynamic_weighting", True)
        if use_dynamic:
            self._dynamic_weighting = DynamicLossWeighting(
                task_names=["rnnt", "prompt"]
            )
            device = self._multitask_model.device
            self._dynamic_weighting.to(device)
            logger.info("Dynamic loss weighting enabled for Phase 3")

        # Validate multitask manifest is provided
        multitask_manifest = getattr(self._train_cfg, "multitask_train_manifest", None)
        if multitask_manifest is None:
            logger.warning(
                "Phase 3 (Paralinguistic) requires --multitask_train_manifest "
                "with annotated task labels. Using default train_manifest."
            )

        logger.info("Phase Paralinguistic: prompt layers active, dynamic weighting on")

    def _setup_multitask_data_loaders(self) -> None:
        """Create data loaders using MultitaskStreamingDataset."""
        manifest = getattr(
            self._train_cfg, "multitask_train_manifest", self._train_cfg.train_manifest
        )
        registry = self._multitask_model.registry

        self._train_loader = DataLoaderFactory.create_multitask_dataloader(
            manifest_paths=[manifest],
            tokenizer=self._multitask_model.tokenizer,
            task_registry=registry,
            batch_size=self._train_cfg.batch_size,
            max_speakers=self._train_cfg.max_speakers,
            num_workers=self._train_cfg.synthesis_num_workers,
        )

        self._val_loader = DataLoaderFactory.create_multitask_dataloader(
            manifest_paths=[self._train_cfg.val_manifest],
            tokenizer=self._multitask_model.tokenizer,
            task_registry=registry,
            batch_size=self._train_cfg.batch_size,
            max_speakers=self._train_cfg.max_speakers,
            num_workers=self._train_cfg.synthesis_num_workers,
            max_samples=self._train_cfg.batch_size * 20,
        )
