import json
import os

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import open_dict


class TokenizerExtender:
    def __init__(self, asr_model, merged_vocab_path: str, token_mapping_path: str):
        self._asr_model = asr_model
        self._merged_vocab_path = merged_vocab_path

        with open(token_mapping_path, "r") as f:
            self._mapping = json.load(f)

        self._old_vocab_size = self._mapping["old_vocab_size"]
        self._new_vocab_size = self._mapping["new_vocab_size"]

        logger.info(f"Extending from {self._old_vocab_size} to {self._new_vocab_size} tokens")

    @property
    def old_vocab_size(self) -> int:
        return self._old_vocab_size

    @property
    def new_vocab_size(self) -> int:
        return self._new_vocab_size

    def resize_decoder_embedding(self) -> None:
        old_embed = self._asr_model.decoder.prediction.embed
        old_vocab_with_blank = old_embed.weight.shape[0]
        embed_dim = old_embed.embedding_dim

        logger.info(f"Resizing decoder embedding: {old_vocab_with_blank} -> {self._new_vocab_size + 1}")

        new_embed = nn.Embedding(
            num_embeddings=self._new_vocab_size + 1,
            embedding_dim=embed_dim,
            padding_idx=self._new_vocab_size,
        )

        with torch.no_grad():
            new_embed.weight.data[: self._old_vocab_size] = old_embed.weight.data[
                : self._old_vocab_size
            ].clone()

            nn.init.xavier_uniform_(
                new_embed.weight.data[self._old_vocab_size : self._new_vocab_size]
            )

            new_embed.weight.data[self._new_vocab_size] = old_embed.weight.data[
                old_vocab_with_blank - 1
            ].clone()

        self._asr_model.decoder.prediction.embed = new_embed
        logger.success(f"Decoder embedding resized to {new_embed.weight.shape}")

    def resize_joint_output(self) -> None:
        old_linear = self._asr_model.joint.joint_net[2]
        in_features = old_linear.in_features
        old_out_features = old_linear.out_features

        logger.info(f"Resizing joint output: {old_out_features} -> {self._new_vocab_size + 1}")

        new_linear = nn.Linear(in_features, self._new_vocab_size + 1)

        with torch.no_grad():
            new_linear.weight.data[: self._old_vocab_size] = old_linear.weight.data[
                : self._old_vocab_size
            ].clone()

            nn.init.normal_(
                new_linear.weight.data[self._old_vocab_size : self._new_vocab_size],
                mean=0.0,
                std=0.01,
            )

            new_linear.weight.data[self._new_vocab_size] = old_linear.weight.data[
                old_out_features - 1
            ].clone()

            if old_linear.bias is not None:
                new_linear.bias.data[: self._old_vocab_size] = old_linear.bias.data[
                    : self._old_vocab_size
                ].clone()
                nn.init.constant_(
                    new_linear.bias.data[self._old_vocab_size : self._new_vocab_size],
                    -20.0,
                )
                new_linear.bias.data[self._new_vocab_size] = old_linear.bias.data[
                    old_out_features - 1
                ].clone()

        self._asr_model.joint.joint_net[2] = new_linear
        logger.success(f"Joint output resized to {new_linear.weight.shape}")

    def update_tokenizer_config(self) -> None:
        with open_dict(self._asr_model.cfg):
            self._asr_model.cfg.decoder.vocab_size = self._new_vocab_size + 1
            if "vocabulary" in self._asr_model.cfg.decoder:
                self._asr_model.cfg.decoder.pop("vocabulary")

            self._asr_model.cfg.joint.num_classes = self._new_vocab_size + 1
            if "vocabulary" in self._asr_model.cfg.joint:
                self._asr_model.cfg.joint.pop("vocabulary")

            if hasattr(self._asr_model.cfg, "tokenizer"):
                self._asr_model.cfg.tokenizer.vocab_size = self._new_vocab_size
                if "vocab_path" in self._asr_model.cfg.tokenizer:
                    self._asr_model.cfg.tokenizer.pop("vocab_path")
                if "spe_tokenizer_vocab" in self._asr_model.cfg.tokenizer:
                    self._asr_model.cfg.tokenizer.pop("spe_tokenizer_vocab")

        logger.success("Model configuration updated")

    def rebuild_tokenizer(self, merged_tokenizer_model_path: str) -> None:
        from nemo.collections.common.tokenizers import SentencePieceTokenizer

        new_tokenizer = SentencePieceTokenizer(model_path=merged_tokenizer_model_path)

        if new_tokenizer.vocab_size != self._new_vocab_size:
            raise ValueError(
                f"Tokenizer vocab size mismatch: {new_tokenizer.vocab_size} != {self._new_vocab_size}"
            )

        self._asr_model.tokenizer = new_tokenizer

        if hasattr(self._asr_model.cfg, "tokenizer"):
            new_path = os.path.abspath(merged_tokenizer_model_path)
            self._asr_model.cfg.tokenizer.model_path = new_path
            try:
                self._asr_model.register_artifact("tokenizer.model_path", new_path)
            except Exception as e:
                logger.warning(f"Could not register artifact: {e}")

        logger.success(f"Tokenizer replaced with {self._new_vocab_size} tokens")

    def save_extended_model(self, output_path: str) -> None:
        self._asr_model.save_to(output_path)
        logger.success(f"Extended model saved to {output_path}")

    def extend(self, merged_tokenizer_model_path: str, output_path: str) -> None:
        self.resize_decoder_embedding()
        self.resize_joint_output()
        self.update_tokenizer_config()
        self.rebuild_tokenizer(merged_tokenizer_model_path)
        self.save_extended_model(output_path)
