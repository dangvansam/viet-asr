import os
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict
from loguru import logger

from nemo.collections.asr.models import SortformerEncLabelModel, ASRModel
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
    SpeakerTaggedASR,
    write_seglst_file,
)
from .config import ModelConfig, InferenceConfig, TrainingConfig


class MultitalkerASRModel:
    def __init__(self, model_cfg: ModelConfig = None):
        self.model_cfg = model_cfg or ModelConfig()
        self.device = torch.device('cpu') if self.model_cfg.cuda_id < 0 or not torch.cuda.is_available(
        ) else torch.device(f'cuda:{self.model_cfg.cuda_id}')
        self.map_location = self.device

        self.asr_model = None
        self.diar_model = None

    def load_models(self):
        """Loads both ASR and Diarization models."""
        logger.info(
            f"Loading Diarization Model from {self.model_cfg.diar_model_path}...")
        if os.path.exists(self.model_cfg.diar_model_path):
            self.diar_model = SortformerEncLabelModel.restore_from(
                restore_path=self.model_cfg.diar_model_path,
                map_location=self.map_location
            )
        else:
            logger.warning(
                f"Diarization model file not found at {self.model_cfg.diar_model_path}, trying pretrained name...")
            self.diar_model = SortformerEncLabelModel.from_pretrained(
                self.model_cfg.diar_pretrained_name,
                map_location=self.map_location
            )

        logger.info(
            f"Loading ASR Model from {self.model_cfg.asr_model_path}...")
        if os.path.exists(self.model_cfg.asr_model_path):
            self.asr_model = ASRModel.restore_from(
                restore_path=self.model_cfg.asr_model_path,
                map_location=self.map_location
            )
        else:
            logger.warning(
                f"ASR model file not found at {self.model_cfg.asr_model_path}, trying cloud...")
            self.asr_model = ASRModel.from_pretrained(
                "nvidia/multitalker-parakeet-streaming-0.6b-v1",
                map_location=self.map_location
            )

        self.asr_model.to(self.device).eval()
        self.diar_model.to(self.device).eval()

    def setup_streaming(self, cfg: InferenceConfig):
        """Configures the models for streaming inference."""
        # Diarization streaming setup
        OmegaConf.set_struct(self.diar_model.cfg, False)
        self.diar_model.cfg.stream_params = OmegaConf.create({
            "window_length_s": 0.5,
            "shift_length_s": 0.05,
            "margin_frames": 10,
            "latency_s": 0.5,
        })
        # Note: In newer NeMo, Sortformer might have different streaming setup
        # For simplicity in this refactor, we match the patched logic found earlier
        if hasattr(self.diar_model, 'sortformer_modules'):
            self.diar_model.sortformer_modules.chunk_len = 0
            self.diar_model.sortformer_modules.spkcache_len = 188
            self.diar_model.sortformer_modules.fifo_len = 188

        # ASR streaming setup
        if cfg.att_context_size and hasattr(self.asr_model.encoder, "set_default_att_context_size"):
            self.asr_model.encoder.set_default_att_context_size(
                att_context_size=cfg.att_context_size)

    def transcribe(self, audio_path: str, output_path: str = "output.json", cfg: InferenceConfig = None):
        """Performs multi-talker transcription on a single audio file."""
        if cfg is None:
            cfg = InferenceConfig()

        self.setup_streaming(cfg)

        samples = [{'audio_filepath': audio_path}]
        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=self.asr_model,
            online_normalization=False,  # default for now
            pad_and_drop_preencoded=False,
        )
        streaming_buffer.append_audio_file(
            audio_filepath=audio_path, stream_id=-1)

        # Wrapped NeMo streamer logic
        multispk_asr_streamer = SpeakerTaggedASR(
            cfg, self.asr_model, self.diar_model)

        # Using parallelism as it's the recommended strategy for Multitalker
        autocast = torch.amp.autocast(self.asr_model.device.type, enabled=True)

        for step_num, (chunk_audio, chunk_lengths) in enumerate(iter(streaming_buffer)):
            drop_extra_pre_encoded = (
                0 if step_num == 0 else self.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
            )
            with torch.inference_mode():
                with autocast:
                    multispk_asr_streamer.perform_parallel_streaming_stt_spk(
                        step_num=step_num,
                        chunk_audio=chunk_audio,
                        chunk_lengths=chunk_lengths,
                        is_buffer_empty=streaming_buffer.is_buffer_empty(),
                        drop_extra_pre_encoded=drop_extra_pre_encoded,
                    )

        batch_seglst_list = multispk_asr_streamer.generate_seglst_dicts_from_parallel_streaming(
            samples=samples)

        if output_path:
            write_seglst_file(
                seglst_dict_list=batch_seglst_list, output_path=output_path)
            logger.success(f"Transcription saved to {output_path}")

        return batch_seglst_list

    def finetune(self, train_cfg: TrainingConfig):
        """Fine-tunes the ASR model."""
        if self.asr_model is None:
            self.load_models()

        cfg = self.asr_model.cfg
        with open_dict(cfg):
            cfg.train_ds.manifest_filepath = train_cfg.train_manifest
            cfg.train_ds.batch_size = train_cfg.batch_size
            cfg.validate_ds = cfg.train_ds.copy()
            cfg.validate_ds.manifest_filepath = train_cfg.val_manifest

            # Optimization
            cfg.optim.lr = train_cfg.learning_rate
            cfg.optim.weight_decay = train_cfg.weight_decay

        self.asr_model.setup_training_data(train_data_config=cfg.train_ds)
        self.asr_model.setup_multiple_validation_data(
            val_data_config=cfg.validate_ds)
        self.asr_model.setup_optimization(optim_config=cfg.optim)

        trainer = pl.Trainer(
            devices=1,
            accelerator='gpu' if torch.cuda.is_available(
            ) and self.model_cfg.cuda_id >= 0 else 'cpu',
            max_steps=train_cfg.max_steps,
            accumulate_grad_batches=train_cfg.accumulate_grad_batches,
            precision=train_cfg.precision,
            val_check_interval=train_cfg.val_check_interval,
            enable_checkpointing=True,
            default_root_dir="checkpoints",
        )

        self.asr_model.set_trainer(trainer)

        # Patch annoying PyTorch Lightning version inheritance mismatch
        import pytorch_lightning.trainer.trainer as pl_trainer
        pl_trainer._maybe_unwrap_optimized = lambda x: x

        trainer.fit(self.asr_model)

        save_path = self.model_cfg.asr_model_path.replace(
            ".nemo", "-finetuned.nemo")
        self.asr_model.save_to(save_path)
        logger.success(f"Fine-tuned model saved to {save_path}")
