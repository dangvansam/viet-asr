import pytest
import torch

from multitalker_asr.configs.multitask import MultiTaskConfig
from multitalker_asr.models.prompt_embedding import TaskTokenRegistry
from multitalker_asr.inference.multitask import SpeakerResult, MultitaskInferenceEngine
from multitalker_asr.eval.multitask_evaluator import MultitaskEvaluator


class TestSpeakerResult:
    def test_to_dict(self):
        r = SpeakerResult(
            speaker_id="spk_0",
            text="xin chao",
            emotion="happy",
            gender="male",
        )
        d = r.to_dict()
        assert d["speaker_id"] == "spk_0"
        assert d["text"] == "xin chao"
        assert d["emotion"] == "happy"
        assert d["gender"] == "male"
        assert d["text_refined"] is None

    def test_str_basic(self):
        r = SpeakerResult(speaker_id="spk_0", text="hello")
        s = str(r)
        assert "[spk_0]" in s
        assert "hello" in s

    def test_str_with_labels(self):
        r = SpeakerResult(
            speaker_id="spk_0",
            text="hello",
            emotion="happy",
            gender="female",
            eou_detected=True,
        )
        s = str(r)
        assert "emotion=happy" in s
        assert "gender=female" in s
        assert "[EOU]" in s

    def test_str_prefers_refined(self):
        r = SpeakerResult(text="raw", text_refined="Refined.")
        s = str(r)
        assert "Refined." in s

    def test_default_values(self):
        r = SpeakerResult()
        assert r.speaker_id == ""
        assert r.text == ""
        assert r.emotion is None
        assert r.eou_detected is False


class TestDecodeLabels:
    def test_decode_all_tasks(self):
        cfg = MultiTaskConfig()
        registry = TaskTokenRegistry(cfg)

        engine = MultitaskInferenceEngine.__new__(MultitaskInferenceEngine)
        engine._registry = registry

        result = SpeakerResult(speaker_id="test")
        preds = {
            "emotion": torch.tensor([0]),  # happy
            "gender": torch.tensor([1]),   # female
            "age": torch.tensor([2]),      # middle_age
            "voice_state": torch.tensor([1]),  # drunk
            "language": torch.tensor([0]),  # vi
        }
        result = engine._decode_labels(result, preds)

        assert result.emotion == "happy"
        assert result.gender == "female"
        assert result.age == "middle_age"
        assert result.voice_state == "drunk"
        assert result.language == "vi"


class TestDetectEOU:
    def test_eou_present(self):
        assert MultitaskInferenceEngine.detect_eou([1, 2, 3, 99], eou_token_id=99)

    def test_eou_absent(self):
        assert not MultitaskInferenceEngine.detect_eou([1, 2, 3], eou_token_id=99)

    def test_empty_tokens(self):
        assert not MultitaskInferenceEngine.detect_eou([], eou_token_id=99)


class TestEditDistance:
    def test_identical(self):
        assert MultitaskEvaluator._edit_distance(["a", "b"], ["a", "b"]) == 0

    def test_insertion(self):
        assert MultitaskEvaluator._edit_distance(["a"], ["a", "b"]) == 1

    def test_deletion(self):
        assert MultitaskEvaluator._edit_distance(["a", "b"], ["a"]) == 1

    def test_substitution(self):
        assert MultitaskEvaluator._edit_distance(["a"], ["b"]) == 1

    def test_empty(self):
        assert MultitaskEvaluator._edit_distance([], ["a", "b"]) == 2
        assert MultitaskEvaluator._edit_distance(["a", "b"], []) == 2


class TestMultitaskEvaluatorMetrics:
    def test_compute_task_accuracy(self):
        evaluator = MultitaskEvaluator.__new__(MultitaskEvaluator)

        predictions = [
            SpeakerResult(emotion="happy"),
            SpeakerResult(emotion="sad"),
            SpeakerResult(emotion="happy"),
        ]
        references = [
            {"emotion": "happy"},
            {"emotion": "happy"},
            {"emotion": "happy"},
        ]
        acc = evaluator._compute_task_accuracy("emotion", predictions, references)
        assert acc == pytest.approx(2 / 3)

    def test_no_ground_truth_returns_none(self):
        evaluator = MultitaskEvaluator.__new__(MultitaskEvaluator)

        predictions = [SpeakerResult(emotion="happy")]
        references = [{"text": "hello"}]  # No emotion field
        acc = evaluator._compute_task_accuracy("emotion", predictions, references)
        assert acc is None

    def test_asr_metrics(self):
        evaluator = MultitaskEvaluator.__new__(MultitaskEvaluator)

        predictions = [
            SpeakerResult(text="hello world"),
            SpeakerResult(text="good morning"),
        ]
        references = [
            {"text": "hello world"},
            {"text": "good morning"},
        ]
        metrics = evaluator._compute_asr_metrics(predictions, references)
        assert metrics["wer"] == pytest.approx(0.0)
        assert metrics["cer"] == pytest.approx(0.0)

    def test_asr_metrics_with_errors(self):
        evaluator = MultitaskEvaluator.__new__(MultitaskEvaluator)

        predictions = [SpeakerResult(text="hello")]
        references = [{"text": "hello world"}]
        metrics = evaluator._compute_asr_metrics(predictions, references)
        assert metrics["wer"] > 0
