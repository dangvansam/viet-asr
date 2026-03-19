"""Utilities for extending ASR model tokenizer."""

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import open_dict
import json


class TokenizerExtender:
    """Handles tokenizer extension and model layer resizing."""

    def __init__(self, asr_model, merged_vocab_path, token_mapping_path):
        """
        Initialize tokenizer extender.

        Args:
            asr_model: NeMo ASR model instance
            merged_vocab_path: Path to merged vocabulary file
            token_mapping_path: Path to token mapping JSON
        """
        self.asr_model = asr_model
        self.merged_vocab_path = merged_vocab_path

        # Load token mapping
        with open(token_mapping_path, 'r') as f:
            self.mapping = json.load(f)

        self.old_vocab_size = self.mapping['old_vocab_size']
        self.new_vocab_size = self.mapping['new_vocab_size']

        logger.info(
            f"Extending from {self.old_vocab_size} to {self.new_vocab_size} tokens")

    def resize_decoder_embedding(self):
        """
        Resize decoder embedding layer while preserving English embeddings.

        Current: (1025, 640) = [1024 English tokens + 1 blank]
        New: (new_vocab_size + 1, 640) = [English + Vietnamese tokens + 1 blank]
        """
        old_embed = self.asr_model.decoder.prediction.embed
        old_vocab_with_blank = old_embed.weight.shape[0]  # 1025
        embed_dim = old_embed.embedding_dim  # 640

        logger.info(
            f"Resizing decoder embedding: {old_vocab_with_blank} → {self.new_vocab_size + 1}")

        # Create new embedding layer
        new_embed = nn.Embedding(
            num_embeddings=self.new_vocab_size + 1,  # +1 for blank token
            embedding_dim=embed_dim,
            padding_idx=self.new_vocab_size
        )

        # Transfer English token embeddings (preserve learned representations)
        with torch.no_grad():
            # Copy English tokens (0 to old_vocab_size-1)
            new_embed.weight.data[:self.old_vocab_size] = \
                old_embed.weight.data[:self.old_vocab_size].clone()

            # Initialize Vietnamese tokens with xavier to prevent completely mimicking the blank token (which is 0.0)
            nn.init.xavier_uniform_(
                new_embed.weight.data[self.old_vocab_size:self.new_vocab_size])

            # Keep blank token at the end
            new_embed.weight.data[self.new_vocab_size] = \
                old_embed.weight.data[old_vocab_with_blank - 1].clone()

        # Replace embedding layer
        self.asr_model.decoder.prediction.embed = new_embed

        logger.success(
            f"Decoder embedding resized to {new_embed.weight.shape}")

    def resize_joint_output(self):
        """
        Resize joint network output layer.

        Current: Linear(640 → 1025)
        New: Linear(640 → new_vocab_size + 1)
        """
        old_linear = self.asr_model.joint.joint_net[2]
        in_features = old_linear.in_features  # 640
        old_out_features = old_linear.out_features  # 1025

        logger.info(
            f"Resizing joint output: {old_out_features} → {self.new_vocab_size + 1}")

        # Create new output layer
        new_linear = nn.Linear(in_features, self.new_vocab_size + 1)

        with torch.no_grad():
            # Transfer English token weights
            new_linear.weight.data[:self.old_vocab_size] = \
                old_linear.weight.data[:self.old_vocab_size].clone()

            # Initialize Vietnamese token weights with very small random variance to provide symmetry breaking without huge logits
            nn.init.normal_(
                new_linear.weight.data[self.old_vocab_size:self.new_vocab_size], mean=0.0, std=0.01)

            # Keep blank token
            new_linear.weight.data[self.new_vocab_size] = \
                old_linear.weight.data[old_out_features - 1].clone()

            # Handle bias if present
            if old_linear.bias is not None:
                new_linear.bias.data[:self.old_vocab_size] = \
                    old_linear.bias.data[:self.old_vocab_size].clone()
                # Initialize new token biases to massively negative (-20.0) to mathematically prevent them from winning tie-breakers against natural pretrained negative logits
                nn.init.constant_(
                    new_linear.bias.data[self.old_vocab_size:self.new_vocab_size], -20.0)
                new_linear.bias.data[self.new_vocab_size] = \
                    old_linear.bias.data[old_out_features - 1].clone()

        # Replace layer
        self.asr_model.joint.joint_net[2] = new_linear

        logger.success(f"Joint output resized to {new_linear.weight.shape}")

    def update_tokenizer_config(self):
        """Update model configuration with new vocabulary size."""
        with open_dict(self.asr_model.cfg):
            # Update decoder vocab size
            self.asr_model.cfg.decoder.vocab_size = self.new_vocab_size + 1
            if 'vocabulary' in self.asr_model.cfg.decoder:
                self.asr_model.cfg.decoder.pop('vocabulary')

            # Update joint vocab size
            self.asr_model.cfg.joint.num_classes = self.new_vocab_size + 1
            if 'vocabulary' in self.asr_model.cfg.joint:
                self.asr_model.cfg.joint.pop('vocabulary')

            # Update tokenizer metadata
            if hasattr(self.asr_model.cfg, 'tokenizer'):
                self.asr_model.cfg.tokenizer.vocab_size = self.new_vocab_size
                if 'vocab_path' in self.asr_model.cfg.tokenizer:
                    self.asr_model.cfg.tokenizer.pop('vocab_path')
                if 'spe_tokenizer_vocab' in self.asr_model.cfg.tokenizer:
                    self.asr_model.cfg.tokenizer.pop('spe_tokenizer_vocab')

        logger.success("Model configuration updated")

    def rebuild_tokenizer(self, merged_tokenizer_model_path):
        """
        Replace model's tokenizer with merged version.

        Args:
            merged_tokenizer_model_path: Path to retrained SentencePiece model
                                        with merged vocabulary
        """
        from nemo.collections.common.tokenizers import SentencePieceTokenizer

        # Create new tokenizer instance
        new_tokenizer = SentencePieceTokenizer(
            model_path=merged_tokenizer_model_path)

        # Validate vocab size matches
        if new_tokenizer.vocab_size != self.new_vocab_size:
            raise ValueError(
                f"Tokenizer vocab size mismatch: {new_tokenizer.vocab_size} != {self.new_vocab_size}"
            )

        # Replace model's tokenizer
        self.asr_model.tokenizer = new_tokenizer

        # Critical fix: update model config so save_to() packages the new tokenizer
        if hasattr(self.asr_model.cfg, 'tokenizer'):
            import os
            new_path = os.path.abspath(merged_tokenizer_model_path)
            self.asr_model.cfg.tokenizer.model_path = new_path
            try:
                self.asr_model.register_artifact(
                    "tokenizer.model_path", new_path)
            except Exception as e:
                logger.warning(f"Could not register artifact: {e}")

        logger.success(f"Tokenizer replaced with {self.new_vocab_size} tokens")

    def save_extended_model(self, output_path):
        """Save extended model as .nemo checkpoint."""
        self.asr_model.save_to(output_path)
        logger.success(f"Extended model saved to {output_path}")
