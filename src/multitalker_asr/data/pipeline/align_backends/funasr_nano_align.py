"""
FunASRNanoAlignBackend: forced alignment via Fun-ASR-MLT-Nano's CTC head.

Mirrors the timestamp path in tmp/FunASR/funasr/models/fun_asr_nano/inference_vllm.py
(`_compute_timestamps`): audio → frontend → audio_encoder → ctc_decoder →
ctc.log_softmax → torchaudio forced_align(ctc_tokenizer.encode(given_text), blank).
Loads ONLY the encoder + CTC parts (no adaptor / LLM / vLLM), so it's a light
forced aligner. The CTC tokenizer is the multilingual tiktoken (60515 tokens) →
expected to handle Vietnamese far better than the Mandarin-only fa-zh aligner.
"""

import os
from typing import Dict, List, Optional

from loguru import logger

from .base import AlignBackendError, AlignedWord, AlignResult, BaseAlignBackend, alignment_score


class FunASRNanoAlignBackend(BaseAlignBackend):
    name = "funasr_nano_align"
    languages: List[str] = []          # multilingual CTC tokenizer

    def __init__(
        self,
        model: str = "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        hub: str = "ms",
        device: str = "cuda",
        dtype: str = "float32",
    ):
        self._model_id = model
        self._hub = hub
        self._device = device
        self._dtype = dtype
        self._frontend = None
        self._encoder = None
        self._ctc_decoder = None
        self._ctc = None
        self._ctc_tokenizer = None
        self._blank_id = None
        self._loaded = False

    # ---- loading (encoder + CTC only) ------------------------------------

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            import torch
            from omegaconf import OmegaConf
            from funasr.register import tables
            from funasr.models.fun_asr_nano.ctc import CTC
        except ImportError as exc:
            raise AlignBackendError(
                "funasr is required for FunASRNanoAlignBackend. Install via `uv add funasr`."
            ) from exc

        self._device = device or self._device
        model_dir = self._resolve_model_dir()
        config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
        cfg = OmegaConf.to_container(config, resolve=True)

        # Frontend
        frontend_class = tables.frontend_classes.get(cfg["frontend"])
        fe_conf = dict(cfg.get("frontend_conf", {}) or {})
        cmvn = fe_conf.get("cmvn_file")
        if cmvn and not os.path.isabs(cmvn):
            fe_conf["cmvn_file"] = os.path.join(model_dir, cmvn)
        self._frontend = frontend_class(**fe_conf)
        self._frontend.eval()

        # Audio encoder
        enc_conf = dict(cfg.get("audio_encoder_conf", {}) or {})
        if enc_conf.get("hub") == "ms":
            from funasr import AutoModel as FunAutoModel
            enc_model = FunAutoModel(model=cfg["audio_encoder"], model_revision="master",
                                     disable_update=True)
            inner = enc_model.model
            enc_out_size = getattr(inner, "encoder_output_size", -1)
            self._encoder = inner.model.encoder if hasattr(inner, "model") else inner.encoder
        else:
            encoder_class = tables.encoder_classes.get(cfg["audio_encoder"])
            self._encoder = encoder_class(input_size=self._frontend.output_size(), **enc_conf)
            enc_out_size = self._encoder.output_size()
        self._encoder.eval()

        # CTC decoder + CTC head + tokenizer (the timestamp path)
        ctc_decoder_name = cfg.get("ctc_decoder")
        if not ctc_decoder_name:
            raise AlignBackendError("Model has no ctc_decoder — cannot force-align.")
        ctc_dec_conf = dict(cfg.get("ctc_decoder_conf", {}) or {})
        if enc_out_size > 0:
            ctc_dec_conf["encoder_dim"] = enc_out_size
        self._ctc_decoder = tables.adaptor_classes.get(ctc_decoder_name)(**ctc_dec_conf)
        self._ctc_decoder.eval()

        ctc_conf = dict(cfg.get("ctc_conf", {}) or {})
        ctc_vocab = cfg.get("ctc_vocab_size", 60515)
        self._blank_id = ctc_conf.get("blank_id", ctc_vocab - 1)
        self._ctc = CTC(odim=ctc_vocab, encoder_output_size=enc_out_size,
                        blank_id=self._blank_id, **ctc_conf)
        self._ctc.eval()

        ds_conf = cfg.get("dataset_conf", {}) or {}
        tok_name = ds_conf.get("ctc_tokenizer")
        tok_conf = dict(ds_conf.get("ctc_tokenizer_conf", {}) or {})
        vocab_path = tok_conf.get("vocab_path")
        multiling = os.path.join(model_dir, "multilingual.tiktoken")
        if (vocab_path is None or not os.path.isabs(vocab_path)) and os.path.exists(multiling):
            tok_conf["vocab_path"] = multiling
        elif vocab_path and not os.path.isabs(vocab_path):
            tok_conf["vocab_path"] = os.path.join(model_dir, vocab_path)
        self._ctc_tokenizer = tables.tokenizer_classes.get(tok_name)(**tok_conf)

        self._load_weights(model_dir)
        self._loaded = True
        logger.info(f"FunASRNanoAlignBackend ready (encoder+CTC) device={self._device}")

    def _resolve_model_dir(self) -> str:
        if os.path.isdir(self._model_id):
            return self._model_id
        if self._hub in ("ms", "modelscope"):
            from modelscope.hub.snapshot_download import snapshot_download
            return snapshot_download(self._model_id, revision="master")
        from huggingface_hub import snapshot_download
        return snapshot_download(self._model_id)

    def _load_weights(self, model_dir: str) -> None:
        import torch
        model_pt = os.path.join(model_dir, "model.pt")
        if not os.path.exists(model_pt):
            raise AlignBackendError(f"model.pt not found at {model_pt}")
        ckpt = torch.load(model_pt, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)

        def sub(prefix, exclude=None):
            out = {}
            for k, v in sd.items():
                if k.startswith(prefix) and not (exclude and k.startswith(exclude)):
                    out[k[len(prefix):]] = v
            return out

        enc = sub("audio_encoder.")
        if enc:
            self._encoder.load_state_dict(enc, strict=False)
        dec = sub("ctc_decoder.")
        ctc = sub("ctc.", exclude="ctc_decoder.")
        # Fun-ASR-MLT-Nano-2512's published model.pt ships only audio_encoder +
        # audio_adaptor + llm — NO CTC head. Without it, forced_align runs through
        # a randomly-initialized CTC and yields meaningless timestamps. Fail loudly
        # rather than silently produce garbage.
        if not dec or not ctc:
            raise AlignBackendError(
                f"Checkpoint at {model_dir} has no CTC weights "
                f"(ctc_decoder={len(dec)}, ctc={len(ctc)} keys) — Fun-ASR-MLT-Nano-2512 "
                "ships encoder+adaptor+LLM only, so CTC forced-alignment is unavailable. "
                "Use mms_fa / qwen3 for alignment, or a Fun-ASR checkpoint that includes the CTC head."
            )
        self._ctc_decoder.load_state_dict(dec, strict=False)
        self._ctc.load_state_dict(ctc, strict=False)

        self._encoder = self._encoder.to(self._device, dtype=torch.float32)
        self._ctc_decoder = self._ctc_decoder.to(self._device, dtype=torch.float32)
        self._ctc = self._ctc.to(self._device, dtype=torch.float32)

    # ---- alignment -------------------------------------------------------

    def align(self, audio_path: str, text: str, language: str) -> AlignResult:
        if not self._loaded:
            raise AlignBackendError("FunASRNanoAlignBackend not loaded. Call load() first.")
        if not text.strip():
            return AlignResult(words=[], score=0.0, backend=self.name)
        try:
            import torch
            from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

            data = load_audio_text_image_video(audio_path, fs=self._frontend.fs)
            speech, speech_lengths = extract_fbank(
                data, data_type="sound", frontend=self._frontend, is_final=True
            )
            speech = speech.to(self._device, dtype=torch.float32)
            speech_lengths = speech_lengths.to(self._device)
            with torch.no_grad():
                enc_out, enc_lens = self._encoder(speech, speech_lengths)
                token_items = self._compute_timestamps(enc_out, enc_lens, text)
        except Exception as exc:
            logger.error(f"FunASR-Nano alignment error for {audio_path}: {exc}")
            return AlignResult(words=[], score=0.0, backend=self.name)

        words = self._group_to_words(token_items)
        return AlignResult(words=words, score=alignment_score(words), backend=self.name,
                           raw={"tokens": token_items})

    def _compute_timestamps(self, encoder_out, encoder_out_lens, text) -> List[Dict]:
        import torch
        from funasr.models.fun_asr_nano.tools.utils import forced_align

        decoder_out, _ = self._ctc_decoder(encoder_out, encoder_out_lens)
        ctc_logits = self._ctc.log_softmax(decoder_out)
        x = ctc_logits[0, : encoder_out_lens[0].item(), :]
        target_ids = torch.tensor(self._ctc_tokenizer.encode(text), dtype=torch.int64)
        if len(target_ids) == 0:
            return []
        items = forced_align(x, target_ids, self._blank_id)
        for ts in items:
            ts["token"] = self._ctc_tokenizer.decode([ts["token"]])
            ts["start_time"] = ts["start_time"] * 6 * 10 / 1000.0   # frame(10ms)*subsample(6)
            ts["end_time"] = ts["end_time"] * 6 * 10 / 1000.0
        return items

    @staticmethod
    def _group_to_words(token_items: List[Dict]) -> List[AlignedWord]:
        """Group subword tokens into words on leading-space boundaries (byte-BPE)."""
        words: List[AlignedWord] = []
        cur, start, end = "", None, None
        for it in token_items:
            tok = it.get("token", "")
            if (tok[:1].isspace() or tok[:1] in {"▁", "Ġ"}) and cur.strip():
                words.append(AlignedWord(cur.strip(), float(start), float(end)))
                cur, start = "", None
            if start is None:
                start = it["start_time"]
            cur += tok
            end = it["end_time"]
        if cur.strip():
            words.append(AlignedWord(cur.strip(), float(start), float(end)))
        return words

    def unload(self) -> None:
        self._frontend = self._encoder = self._ctc_decoder = self._ctc = None
        self._ctc_tokenizer = None
        self._loaded = False
