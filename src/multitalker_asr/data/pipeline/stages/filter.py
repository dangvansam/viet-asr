from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

from ...manifest_schema import ATTRIBUTE_KEYS
from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import PipelineConfig


@dataclass
class FilterRule:
    name: str

    def check(self, record: Dict) -> Tuple[bool, Optional[str]]:
        return True, None


@dataclass
class TextLengthRule(FilterRule):
    name: str = "text_length"
    min_chars: int = 1
    max_chars: int = 10_000

    def check(self, record: Dict) -> Tuple[bool, Optional[str]]:
        text = record.get("text", "")
        if not text or len(text) < self.min_chars:
            return False, f"text shorter than {self.min_chars} chars"
        if len(text) > self.max_chars:
            return False, f"text exceeds {self.max_chars} chars"
        return True, None


@dataclass
class DurationRatioRule(FilterRule):
    name: str = "duration_ratio"
    min_chars_per_second: float = 0.5
    max_chars_per_second: float = 30.0

    def check(self, record: Dict) -> Tuple[bool, Optional[str]]:
        duration = float(record.get("duration", 0.0))
        text_len = len(record.get("text", ""))
        if duration <= 0 or text_len == 0:
            return True, None
        ratio = text_len / duration
        if ratio < self.min_chars_per_second:
            return False, f"chars/sec {ratio:.2f} below {self.min_chars_per_second}"
        if ratio > self.max_chars_per_second:
            return False, f"chars/sec {ratio:.2f} above {self.max_chars_per_second}"
        return True, None


@dataclass
class LanguageConsistencyRule(FilterRule):
    name: str = "lang_consistency"
    expected_language: Optional[str] = None

    def check(self, record: Dict) -> Tuple[bool, Optional[str]]:
        if self.expected_language is None:
            return True, None
        actual = record.get("language")
        if actual is None:
            return True, None
        if not actual.startswith(self.expected_language[:2]):
            return False, f"lang={actual} expected={self.expected_language}"
        return True, None


@dataclass
class AttributeConfidenceRule(FilterRule):
    name: str = "attribute_confidence"
    min_confidence: Dict[str, float] = field(default_factory=dict)
    drop_record: bool = False

    def check(self, record: Dict) -> Tuple[bool, Optional[str]]:
        confidences = record.get("attribute_confidence", {})
        violations = []
        for axis, threshold in self.min_confidence.items():
            conf = float(confidences.get(axis, 0.0))
            if conf < threshold:
                violations.append(f"{axis}={conf:.2f}<{threshold}")
        if violations and self.drop_record:
            return False, "low_confidence:" + ",".join(violations)
        if violations:
            self._mask(record, list(self.min_confidence.keys()), confidences)
        return True, None

    def _mask(
        self,
        record: Dict,
        axes: List[str],
        confidences: Dict[str, float],
    ) -> None:
        for axis in axes:
            if axis not in record:
                continue
            threshold = self.min_confidence.get(axis, 0.0)
            if float(confidences.get(axis, 0.0)) < threshold:
                record[axis] = None


class FilterStage(BaseStage):
    name = "filter"

    def __init__(self, rules: Optional[List[FilterRule]] = None):
        self._rules: List[FilterRule] = rules or self._default_rules()
        self._stats: Dict[str, int] = {}

    def _default_rules(self) -> List[FilterRule]:
        return [
            TextLengthRule(),
            DurationRatioRule(),
            AttributeConfidenceRule(
                min_confidence={a: 0.5 for a in ATTRIBUTE_KEYS},
                drop_record=False,
            ),
        ]

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        kept: List[Dict] = []
        self._stats = {"kept": 0, "dropped": 0}
        rule_drops: Dict[str, int] = {}

        for record in records:
            keep = True
            for rule in self._rules:
                ok, reason = rule.check(record)
                if not ok:
                    keep = False
                    rule_drops[rule.name] = rule_drops.get(rule.name, 0) + 1
                    logger.debug(f"Drop {record.get('id')} via {rule.name}: {reason}")
                    break
            if keep:
                kept.append(record)
                checkpoint.mark_processed(record["id"], self.name)
                self._stats["kept"] += 1
            else:
                self._stats["dropped"] += 1

        for rule_name, count in rule_drops.items():
            self._stats[f"drop_{rule_name}"] = count

        logger.info(
            f"FilterStage kept {self._stats['kept']}/{len(records)} records "
            f"(dropped {self._stats['dropped']})"
        )
        return kept
