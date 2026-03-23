import os

import lightning.pytorch as pl
import torch
from loguru import logger
from nemo.collections.asr.models import ASRModel, EncDecMultiTalkerRNNTBPEModel, SortformerEncLabelModel
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
    SpeakerTaggedASR,
    write_seglst_file,
)
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from omegaconf import OmegaConf, open_dict

from .config import InferenceConfig, ModelConfig, TrainingConfig


class MultitalkerASRModel:
    def __init__(self, model_cfg: ModelConfig = None):
        self.model_cfg = model_cfg or ModelConfig()
        self.device = torch.device("cpu") if self.model_cfg.cuda_id < 0 or not torch.cuda.is_available(
        ) else torch.device(f"cuda:{self.model_cfg.cuda_id}")
        self.map_location = self.device

        self.asr_model = None
        self.diar_model = None

    def load_models(self, tokenizer_dir: str = None):
        """Loads both ASR and Diarization models."""
        logger.info(
            f"Loading Diarization Model from {self.model_cfg.diar_model_path}...")
        if os.path.exists(self.model_cfg.diar_model_path):
            self.diar_model = SortformerEncLabelModel.restore_from(
                restore_path=self.model_cfg.diar_model_path, map_location=self.map_location)
        else:
            logger.warning(
                f"Diarization model file not found at {self.model_cfg.diar_model_path}, trying pretrained name...")
            self.diar_model = SortformerEncLabelModel.from_pretrained(
                self.model_cfg.diar_pretrained_name, map_location=self.map_location)

        logger.info(
            f"Loading ASR Model from {self.model_cfg.asr_model_path}...")
        if os.path.exists(self.model_cfg.asr_model_path):
            if self.model_cfg.asr_model_path.endswith(".ckpt"):
                logger.info(
                    "Detected .ckpt file, loading config and weights manually...")
                checkpoint = torch.load(
                    self.model_cfg.asr_model_path, map_location=self.map_location, weights_only=False)
                if 'hyper_parameters' in checkpoint and 'cfg' in checkpoint['hyper_parameters']:
                    cfg = checkpoint['hyper_parameters']['cfg']
                    # Patch tokenizer paths if they use nemo: prefix which fails outside of .nemo package
                    if 'tokenizer' in cfg:
                        with open_dict(cfg):
                            for key in ['model_path', 'vocab_path', 'spe_tokenizer_vocab']:
                                if key in cfg.tokenizer and isinstance(cfg.tokenizer[key], str) and cfg.tokenizer[key].startswith("nemo:"):
                                    # Extract the actual filename
                                    filename = cfg.tokenizer[key].split(
                                        "_", 1)[-1] if "_" in cfg.tokenizer[key] else cfg.tokenizer[key][5:]
                                    local_path = os.path.join(
                                        cfg.tokenizer.get('dir', ''), filename)
                                    if os.path.exists(local_path):
                                        logger.info(
                                            f"Patching {key}: {cfg.tokenizer[key]} -> {local_path}")
                                        cfg.tokenizer[key] = local_path

                    self.asr_model = EncDecMultiTalkerRNNTBPEModel(cfg=cfg)
                    self.asr_model.load_state_dict(checkpoint['state_dict'])
                    self.asr_model.to(self.map_location)
                else:
                    raise ValueError(
                        "Could not find model config in checkpoint.")
            else:
                self.asr_model = ASRModel.restore_from(
                    restore_path=self.model_cfg.asr_model_path, map_location=self.map_location)
        elif hasattr(self.model_cfg, "config_path") and self.model_cfg.config_path and os.path.exists(self.model_cfg.config_path):
            logger.info(
                f"ASR model file {self.model_cfg.asr_model_path} not found. Creating from scratch using config {self.model_cfg.config_path}...")

            from nemo.collections.common.tokenizers import SentencePieceTokenizer

            cfg = OmegaConf.load(self.model_cfg.config_path)
            vocab_size = getattr(self.model_cfg, "vocab_size", 2048)
            new_vocab_size = vocab_size + 1

            if tokenizer_dir and os.path.exists(tokenizer_dir):
                if os.path.isfile(tokenizer_dir):
                    tokenizer_path = tokenizer_dir
                else:
                    tokenizer_path = os.path.join(
                        tokenizer_dir, "tokenizer.model")

                tokenizer = SentencePieceTokenizer(model_path=tokenizer_path)
                if tokenizer.vocab_size != vocab_size:
                    logger.warning(
                        f"Tokenizer vocab size mismatch: {tokenizer.vocab_size} != {vocab_size}")
                    vocab_size = tokenizer.vocab_size
                    new_vocab_size = vocab_size + 1

                with open_dict(cfg):
                    if 'tokenizer' not in cfg:
                        cfg.tokenizer = {}

                    cfg.tokenizer.dir = os.path.dirname(
                        os.path.abspath(tokenizer_path))
                    cfg.tokenizer.type = "bpe"
                    cfg.tokenizer.model_path = os.path.abspath(tokenizer_path)
                    cfg.tokenizer.vocab_size = vocab_size

                    vocab_file = os.path.abspath(
                        tokenizer_path).replace(".model", ".vocab")
                    if os.path.exists(vocab_file):
                        cfg.tokenizer.vocab_path = vocab_file
                        cfg.tokenizer.spe_tokenizer_vocab = vocab_file

                    if 'decoder' in cfg:
                        cfg.decoder.vocab_size = new_vocab_size
                        if 'vocabulary' in cfg.decoder:
                            cfg.decoder.pop('vocabulary')
                    if 'joint' in cfg:
                        cfg.joint.num_classes = new_vocab_size
                        if 'vocabulary' in cfg.joint:
                            cfg.joint.pop('vocabulary')
                    if 'model_defaults' in cfg:
                        if 'vocab_size' in cfg.model_defaults:
                            cfg.model_defaults.vocab_size = vocab_size
                        if 'num_classes' in cfg.model_defaults:
                            cfg.model_defaults.num_classes = new_vocab_size

            # Remove datasets so initialization doesn't choke on missing files
            ds_configs = {}
            with open_dict(cfg):
                for ds in ['train_ds', 'validation_ds', 'test_ds']:
                    if ds in cfg:
                        ds_configs[ds] = cfg.pop(ds)

            self.asr_model = EncDecMultiTalkerRNNTBPEModel(
                cfg=cfg, trainer=None)

            # Sync config and restore datasets
            with open_dict(self.asr_model.cfg):
                if tokenizer_dir and os.path.exists(tokenizer_dir):
                    if 'decoder' in self.asr_model.cfg:
                        self.asr_model.cfg.decoder.vocab_size = new_vocab_size
                    if 'joint' in self.asr_model.cfg:
                        self.asr_model.cfg.joint.num_classes = new_vocab_size
                    if 'model_defaults' in self.asr_model.cfg:
                        self.asr_model.cfg.model_defaults.num_classes = new_vocab_size
                for ds, ds_cfg in ds_configs.items():
                    self.asr_model.cfg[ds] = ds_cfg

            if tokenizer_dir and os.path.exists(tokenizer_dir):
                try:
                    self.asr_model.register_artifact(
                        "tokenizer.model_path", cfg.tokenizer.model_path)
                except Exception as e:
                    logger.warning(f"Could not register artifact: {e}")
            logger.success("Scratch model created successfully in memory!")
        else:
            logger.warning(
                f"ASR model file not found at {self.model_cfg.asr_model_path}, trying cloud...")
            self.asr_model = ASRModel.from_pretrained(
                "nvidia/multitalker-parakeet-streaming-0.6b-v1", map_location=self.map_location)

        self.asr_model.to(self.device).eval()
        self.diar_model.to(self.device).eval()

    def setup_streaming(self, cfg: InferenceConfig):
        """Configures the models for streaming inference."""
        # Diarization streaming setup
        OmegaConf.set_struct(self.diar_model.cfg, False)
        self.diar_model.cfg.stream_params = OmegaConf.create(
            {
                "window_length_s": 0.5,
                "shift_length_s": 0.05,
                "margin_frames": 10,
                "latency_s": 0.5,
            }
        )
        # Note: In newer NeMo, Sortformer might have different streaming setup
        # For simplicity in this refactor, we match the patched logic found earlier
        if hasattr(self.diar_model, "sortformer_modules"):
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

        samples = [{"audio_filepath": audio_path}]
        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=self.asr_model,
            online_normalization=False,  # default for now
            pad_and_drop_preencoded=False,
        )
        streaming_buffer.append_audio_file(
            audio_filepath=audio_path, stream_id=-1)

        # Convert dataclass to a flexible OmegaConf object for NeMo compatibility
        # Using to_container ensures it's a plain dict-based config that allows missing keys
        nemo_cfg = OmegaConf.create(OmegaConf.to_container(
            OmegaConf.structured(cfg), resolve=True))
        multispk_asr_streamer = SpeakerTaggedASR(
            nemo_cfg, self.asr_model, self.diar_model)

        # Using parallelism as it's the recommended strategy for Multitalker
        autocast = torch.amp.autocast(self.asr_model.device.type, enabled=True)

        for step_num, (chunk_audio, chunk_lengths) in enumerate(iter(streaming_buffer)):
            drop_extra_pre_encoded = 0 if step_num == 0 else self.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
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
            try:
                write_seglst_file(
                    seglst_dict_list=batch_seglst_list, output_path=output_path)
                logger.success(f"Transcription saved to {output_path}")
            except ValueError as e:
                logger.warning(f"Failed to write transcript: {e}")

        return batch_seglst_list

    def finetune(self, train_cfg: TrainingConfig):
        """Fine-tunes the ASR model."""
        if self.asr_model == None:
            self.load_models(tokenizer_dir=train_cfg.tokenizer_dir if hasattr(
                train_cfg, "tokenizer_dir") else None)

        is_from_scratch = hasattr(self.model_cfg, "config_path") and self.model_cfg.config_path and not os.path.exists(
            self.model_cfg.asr_model_path)
        if hasattr(train_cfg, "tokenizer_dir") and train_cfg.tokenizer_dir is not None and not is_from_scratch:
            logger.info(
                f"Changing vocabulary to new tokenizer at {train_cfg.tokenizer_dir}...")
            self.asr_model.change_vocabulary(
                new_tokenizer_dir=train_cfg.tokenizer_dir, new_tokenizer_type="bpe")

        cfg = self.asr_model.cfg
        with open_dict(cfg):
            cfg.train_ds.cuts_path = train_cfg.train_manifest
            cfg.train_ds.manifest_filepath = None
            cfg.train_ds.batch_size = train_cfg.batch_size
            cfg.train_ds.text_field = "text"
            cfg.train_ds.use_lhotse = True

            if not hasattr(cfg, "validate_ds") or cfg.validate_ds is None:
                cfg.validate_ds = cfg.train_ds.copy()

            cfg.validate_ds.cuts_path = train_cfg.val_manifest
            cfg.validate_ds.manifest_filepath = None
            cfg.validate_ds.text_field = "text"
            cfg.validate_ds.batch_size = train_cfg.batch_size
            cfg.validate_ds.use_lhotse = True

            # Optimization - only override if provided, otherwise preserve config (e.g. for Noam)
            if train_cfg.learning_rate and train_cfg.learning_rate != 1e-5:
                cfg.optim.lr = train_cfg.learning_rate

            if train_cfg.weight_decay and train_cfg.weight_decay != 1e-3:
                cfg.optim.weight_decay = train_cfg.weight_decay

        self.asr_model.setup_training_data(train_data_config=cfg.train_ds)
        self.asr_model.setup_multiple_validation_data(
            val_data_config=cfg.validate_ds)
        self.asr_model.setup_optimization(optim_config=cfg.optim)

        # Disable CUDA graphs for stability
        if hasattr(self.asr_model, "cfg"):
            with open_dict(self.asr_model.cfg):
                self.asr_model.cfg.enable_cuda_graphs = False

        from lightning.pytorch.callbacks import Callback

        class PrintLossCallback(Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                metrics = trainer.callback_metrics
                train_loss = metrics.get('train_loss', 0.0)
                val_loss = metrics.get('val_loss', 0.0)
                val_wer = metrics.get('val_wer', 1.0)
                logger.success(
                    f"Epoch {trainer.current_epoch} End | Train Loss: {float(train_loss):.4f} | Val Loss: {float(val_loss):.4f} | Val WER: {float(val_wer):.4f}")

        logger_list = True
        if hasattr(train_cfg, 'wandb_project') and train_cfg.wandb_project:
            from lightning.pytorch.loggers import WandbLogger
            wandb_logger = WandbLogger(
                project=train_cfg.wandb_project, name=train_cfg.wandb_run_name)
            logger_list = [wandb_logger]

        trainer = pl.Trainer(
            devices=1,
            accelerator="gpu" if torch.cuda.is_available(
            ) and self.model_cfg.cuda_id >= 0 else "cpu",
            max_steps=train_cfg.max_steps,
            max_epochs=train_cfg.max_epochs,
            accumulate_grad_batches=train_cfg.accumulate_grad_batches,
            precision=train_cfg.precision,
            val_check_interval=train_cfg.val_check_interval,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            default_root_dir="checkpoints",
            log_every_n_steps=20,
            inference_mode=False,
            callbacks=[PrintLossCallback()],
            logger=logger_list,
            gradient_clip_val=1.0,  # Gradient clipping for training stability
            gradient_clip_algorithm="norm"
        )

        self.asr_model.set_trainer(trainer)
        # Use lightning.pytorch
        import lightning.pytorch.trainer.trainer as pl_trainer

        pl_trainer._maybe_unwrap_optimized = lambda x: x

        # Ensure model is in training mode
        self.asr_model.train()
        trainer.fit(self.asr_model)

        if train_cfg.output_path:
            save_path = train_cfg.output_path
        else:
            save_path = self.model_cfg.asr_model_path.replace(
                ".nemo", "-finetuned.nemo")

        self.asr_model.save_to(save_path)
        logger.success(f"Fine-tuned model saved to {save_path}")
