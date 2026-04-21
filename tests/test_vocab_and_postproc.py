import os
import pytest

from multitalker_asr.models.vocab_extension import VocabularyExtender
from multitalker_asr.inference.post_processor import FunASRPostProcessor


class TestVocabularyExtender:
    def test_create_extended_tokenizer(self, tmp_path):
        # Create a mock tokenizer dir
        base_dir = tmp_path / "base_tok"
        base_dir.mkdir()
        vocab_file = base_dir / "tokenizer.vocab"
        vocab_file.write_text("hello\t0\nworld\t0\n")
        model_file = base_dir / "tokenizer.model"
        model_file.write_text("dummy_model_data")

        output_dir = str(tmp_path / "extended_tok")

        result = VocabularyExtender.create_extended_tokenizer(
            base_tokenizer_dir=str(base_dir),
            output_dir=output_dir,
            special_tokens=["<EOU>", "<PAUSE>"],
        )

        assert result == output_dir
        assert os.path.exists(os.path.join(output_dir, "tokenizer.vocab"))
        assert os.path.exists(os.path.join(output_dir, "tokenizer.model"))
        assert os.path.exists(os.path.join(output_dir, "user_defined_symbols.txt"))

        # Verify tokens were added
        with open(os.path.join(output_dir, "tokenizer.vocab")) as f:
            content = f.read()
        assert "<EOU>" in content
        assert "<PAUSE>" in content

    def test_duplicate_token_skipped(self, tmp_path):
        base_dir = tmp_path / "base_tok"
        base_dir.mkdir()
        vocab_file = base_dir / "tokenizer.vocab"
        vocab_file.write_text("<EOU>\t0\nhello\t0\n")

        output_dir = str(tmp_path / "extended_tok")

        VocabularyExtender.create_extended_tokenizer(
            base_tokenizer_dir=str(base_dir),
            output_dir=output_dir,
            special_tokens=["<EOU>"],
        )

        # <EOU> should appear only once
        with open(os.path.join(output_dir, "tokenizer.vocab")) as f:
            lines = f.readlines()
        eou_count = sum(1 for l in lines if l.startswith("<EOU>"))
        assert eou_count == 1

    def test_user_defined_symbols_file(self, tmp_path):
        base_dir = tmp_path / "base_tok"
        base_dir.mkdir()
        (base_dir / "tokenizer.vocab").write_text("a\t0\n")

        output_dir = str(tmp_path / "extended_tok")
        VocabularyExtender.create_extended_tokenizer(
            str(base_dir), output_dir, ["<EOU>", "<BOS>"]
        )

        uds_path = os.path.join(output_dir, "user_defined_symbols.txt")
        with open(uds_path) as f:
            tokens = [l.strip() for l in f.readlines()]
        assert "<EOU>" in tokens
        assert "<BOS>" in tokens


class TestFunASRPostProcessor:
    def test_instantiation_no_load(self):
        pp = FunASRPostProcessor(device="cpu")
        assert not pp.is_loaded

    def test_refine_text_no_audio_returns_raw(self):
        pp = FunASRPostProcessor.__new__(FunASRPostProcessor)
        pp._model = None
        pp._model_name = "test"
        pp._device = "cpu"

        result = pp.refine_text(raw_text="hello world")
        assert result == "hello world"

    def test_refine_text_no_input_returns_empty(self):
        pp = FunASRPostProcessor.__new__(FunASRPostProcessor)
        pp._model = None
        pp._model_name = "test"
        pp._device = "cpu"

        result = pp.refine_text()
        assert result == ""

    def test_refine_batch(self):
        pp = FunASRPostProcessor.__new__(FunASRPostProcessor)
        pp._model = None
        pp._model_name = "test"
        pp._device = "cpu"

        items = [
            {"text": "hello", "audio_path": None},
            {"text": "world", "audio_path": None},
        ]
        result = pp.refine_batch(items)
        assert result[0]["refined_text"] == "hello"
        assert result[1]["refined_text"] == "world"

    def test_refine_nonexistent_audio(self):
        pp = FunASRPostProcessor.__new__(FunASRPostProcessor)
        pp._model = "dummy"  # pretend loaded
        pp._model_name = "test"
        pp._device = "cpu"

        result = pp.refine_text(
            audio_path="/tmp/nonexistent_audio_12345.wav",
            raw_text="fallback"
        )
        assert result == "fallback"
