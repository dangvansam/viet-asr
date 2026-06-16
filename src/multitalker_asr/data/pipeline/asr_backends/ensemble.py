import json
import re
from abc import ABC, abstractmethod
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from loguru import logger

from ....utils.llm_client import LLMClient
from .base import ASRResult, WordTiming

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _norm_word(word: str) -> str:
    return _PUNCT.sub("", word).strip().lower()


class BaseASREnsembler(ABC):
    name: str = ""

    @abstractmethod
    def combine(
        self, hypotheses: List[ASRResult], reference: Optional[ASRResult] = None
    ) -> ASRResult:
        ...

    def _drop_empty(self, hypotheses: List[ASRResult]) -> List[ASRResult]:
        return [h for h in hypotheses if h.text]


class SingleASREnsembler(BaseASREnsembler):
    name = "single"

    def combine(
        self, hypotheses: List[ASRResult], reference: Optional[ASRResult] = None
    ) -> ASRResult:
        if not hypotheses:
            return ASRResult(text="", confidence=0.0, backend=self.name)
        return hypotheses[0]


class VoteASREnsembler(BaseASREnsembler):
    name = "vote"

    def __init__(self, similarity_threshold: float = 0.85, min_agree: int = 2):
        self._similarity_threshold = similarity_threshold
        self._min_agree = min_agree

    def combine(
        self, hypotheses: List[ASRResult], reference: Optional[ASRResult] = None
    ) -> ASRResult:
        active = self._drop_empty(hypotheses)
        if not active:
            return ASRResult(text="", confidence=0.0, backend=self.name)
        if len(active) == 1:
            return active[0]

        clusters = self._cluster(active)
        clusters.sort(key=lambda group: (len(group), self._avg_conf(group)), reverse=True)
        best = clusters[0]
        winner = self._select_within_cluster(best)
        agree = len(best)
        return ASRResult(
            text=winner.text,
            confidence=min(1.0, self._avg_conf(best)),
            language=winner.language,
            word_timings=winner.word_timings,
            backend=self.name,
            raw={
                "agree": agree,
                "min_agree": self._min_agree,
                "cluster_backends": [h.backend for h in best],
                "all_backends": [h.backend for h in active],
            },
        )

    def _cluster(self, hypotheses: List[ASRResult]) -> List[List[ASRResult]]:
        clusters: List[List[ASRResult]] = []
        for hyp in hypotheses:
            placed = False
            for cluster in clusters:
                if self._similarity(hyp.text, cluster[0].text) >= self._similarity_threshold:
                    cluster.append(hyp)
                    placed = True
                    break
            if not placed:
                clusters.append([hyp])
        return clusters

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _avg_conf(self, group: List[ASRResult]) -> float:
        if not group:
            return 0.0
        return sum(h.confidence for h in group) / len(group)

    def _select_within_cluster(self, group: List[ASRResult]) -> ASRResult:
        return max(group, key=lambda h: (h.confidence, len(h.text)))


class LLMJudgeASREnsembler(BaseASREnsembler):
    name = "llm_judge"

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: Optional[str] = None,
    ):
        self._client = llm_client
        self._system_prompt = system_prompt or (
            "You are an expert ASR transcription judge. Given multiple ASR hypotheses, "
            "select the most accurate one or produce a corrected version. "
            "Return JSON: {\"text\": str, \"selected_backend\": str, \"confidence\": float}."
        )

    def combine(
        self, hypotheses: List[ASRResult], reference: Optional[ASRResult] = None
    ) -> ASRResult:
        active = self._drop_empty(hypotheses)
        if not active:
            return ASRResult(text="", confidence=0.0, backend=self.name)
        if len(active) == 1:
            return active[0]

        user_prompt = self._format_prompt(active)
        try:
            response = self._client.complete(self._system_prompt, user_prompt)
            text, selected, confidence = self._parse_response(response.text, active)
        except Exception as exc:
            logger.warning(f"LLMJudgeASREnsembler fell back to vote: {exc}")
            return VoteASREnsembler().combine(active)

        return ASRResult(
            text=text,
            confidence=confidence,
            language=active[0].language,
            backend=self.name,
            raw={
                "selected_backend": selected,
                "all_backends": [h.backend for h in active],
            },
        )

    def _format_prompt(self, hypotheses: List[ASRResult]) -> str:
        lines = ["Hypotheses:"]
        for idx, hyp in enumerate(hypotheses):
            lines.append(f"  [{idx}] backend={hyp.backend} conf={hyp.confidence:.3f}")
            lines.append(f"      text: {hyp.text}")
        lines.append("\nReturn JSON only.")
        return "\n".join(lines)

    def _parse_response(self, text: str, hypotheses: List[ASRResult]):
        try:
            data = json.loads(text)
            selected = data.get("selected_backend", hypotheses[0].backend)
            return (
                str(data.get("text", hypotheses[0].text)),
                str(selected),
                float(data.get("confidence", 1.0)),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            return text.strip(), "llm_judge_freeform", 1.0


class WordRoverEnsembler(BaseASREnsembler):
    """Time-aligned, word-level ROVER anchored on the first (primary) hypothesis.

    Walks the anchor's word timeline; for each anchor word, gathers the
    time-overlapping word from every voter that has word timings (other backends
    + the subtitle reference), and votes. The anchor word is kept unless it is
    out-voted by a weighted majority — preserving the primary's punctuation/casing
    while correcting wrong words where the others (esp. the subtitle) agree.

    Voters without word timings (e.g. an LLM ASR) don't join word voting; when the
    anchor lacks timings entirely it falls back to subtitle-guided text selection.
    """

    name = "rover"

    def __init__(self, subtitle_weight: float = 1.5, min_agree: int = 2):
        self._subtitle_weight = subtitle_weight
        self._min_agree = min_agree

    def combine(
        self, hypotheses: List[ASRResult], reference: Optional[ASRResult] = None
    ) -> ASRResult:
        active = self._drop_empty(hypotheses)
        if not active:
            return ASRResult(text="", confidence=0.0, backend=self.name)
        anchor = active[0]
        voters = [h for h in active if h.word_timings]
        if reference is not None and reference.word_timings and reference.text:
            voters = voters + [reference]

        if not anchor.word_timings or len(voters) < 2:
            return self._text_fallback(active, reference)

        tokens: List[str] = []
        timings: List[WordTiming] = []
        changed = agreed = 0
        for aw in anchor.word_timings:
            votes: Dict[str, float] = {}
            originals: Dict[str, str] = {}
            for v in voters:
                w = self._overlap_word(v, aw.start, aw.end)
                if w is None:
                    continue
                norm = _norm_word(w.word)
                if not norm:
                    continue
                votes[norm] = votes.get(norm, 0.0) + self._weight(v)
                originals.setdefault(norm, w.word)

            anchor_norm = _norm_word(aw.word)
            if not votes:
                tokens.append(aw.word)
                timings.append(aw)
                continue
            # Tie → prefer the anchor word (keeps its punctuation/casing).
            best = max(votes, key=lambda k: (votes[k], k == anchor_norm))
            if votes[best] >= self._min_agree:
                agreed += 1
            if best == anchor_norm:
                token = aw.word
            else:
                token = originals[best]
                changed += 1
            tokens.append(token)
            timings.append(WordTiming(word=token, start=aw.start, end=aw.end))

        text = " ".join(t for t in tokens if t)
        n = max(1, len(tokens))
        return ASRResult(
            text=text,
            confidence=round(agreed / n, 4),
            language=anchor.language,
            word_timings=timings,
            backend=self.name,
            raw={
                "anchor": anchor.backend,
                "voters": [v.backend for v in voters],
                "words_changed": changed,
                "words_total": n,
                "subtitle_used": reference is not None and bool(reference.word_timings),
            },
        )

    def _text_fallback(
        self, active: List[ASRResult], reference: Optional[ASRResult]
    ) -> ASRResult:
        """No usable word timings: pick the hypothesis closest to the subtitle,
        else keep the anchor (first) hypothesis."""
        if reference is not None and reference.text:
            best = max(active, key=lambda h: SequenceMatcher(
                None, _norm_word(h.text), _norm_word(reference.text)).ratio())
            return best
        return active[0]

    def _overlap_word(self, hyp: ASRResult, start: float, end: float):
        best = None
        best_ov = 0.0
        for w in hyp.word_timings or []:
            ov = min(end, w.end) - max(start, w.start)
            if ov > best_ov:
                best_ov = ov
                best = w
        return best if best_ov > 0 else None

    def _weight(self, hyp: ASRResult) -> float:
        if hyp.vote_weight is not None:
            return hyp.vote_weight
        return self._subtitle_weight if hyp.backend == "subtitle" else 1.0


class ASREnsemblerFactory:
    @staticmethod
    def build(
        strategy: str,
        llm_client: Optional[LLMClient] = None,
        min_agree: int = 2,
        similarity_threshold: float = 0.85,
        subtitle_weight: float = 1.5,
    ) -> BaseASREnsembler:
        strategy = strategy.lower()
        if strategy == "single":
            return SingleASREnsembler()
        if strategy == "vote":
            return VoteASREnsembler(
                similarity_threshold=similarity_threshold, min_agree=min_agree
            )
        if strategy == "rover":
            return WordRoverEnsembler(
                subtitle_weight=subtitle_weight, min_agree=min_agree
            )
        if strategy == "llm_judge":
            if llm_client is None:
                raise ValueError("llm_judge strategy requires llm_client")
            return LLMJudgeASREnsembler(llm_client)
        raise ValueError(
            f"Unknown ensemble strategy '{strategy}'. "
            f"Valid: single, vote, rover, llm_judge"
        )
