import json
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Optional

from loguru import logger

from ....utils.llm_client import LLMClient
from .base import BaseITNBackend, ITNBackendError, ITNResult


class BaseITNStrategy(ABC):
    name: str = ""

    @abstractmethod
    def apply(
        self,
        backends: List[BaseITNBackend],
        text: str,
        language: str,
    ) -> ITNResult:
        ...

    def _empty_result(self, language: str) -> ITNResult:
        return ITNResult(text_itn="", backend=self.name, language=language)


class FirstSuccessStrategy(BaseITNStrategy):
    name = "first_success"

    def apply(
        self,
        backends: List[BaseITNBackend],
        text: str,
        language: str,
    ) -> ITNResult:
        if not text:
            return self._empty_result(language)
        last_exc: Optional[Exception] = None
        for backend in backends:
            if not backend.supports_language(language):
                continue
            try:
                return backend.normalize(text, language=language)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"ITN backend '{backend.name}' failed for lang={language}: {exc}"
                )
                continue
        if last_exc is not None:
            raise ITNBackendError(
                f"All ITN backends failed for lang={language}: {last_exc}"
            ) from last_exc
        return self._empty_result(language)


class VoteITNStrategy(BaseITNStrategy):
    name = "vote"

    def apply(
        self,
        backends: List[BaseITNBackend],
        text: str,
        language: str,
    ) -> ITNResult:
        if not text:
            return self._empty_result(language)

        results: List[ITNResult] = []
        for backend in backends:
            if not backend.supports_language(language):
                continue
            try:
                results.append(backend.normalize(text, language=language))
            except Exception as exc:
                logger.warning(f"ITN backend '{backend.name}' failed: {exc}")

        if not results:
            return self._empty_result(language)
        if len(results) == 1:
            return results[0]

        counter: Counter = Counter(r.text_itn for r in results)
        top_text, top_count = counter.most_common(1)[0]
        winners = [r for r in results if r.text_itn == top_text]
        avg_conf = sum(r.confidence for r in winners) / len(winners)
        return ITNResult(
            text_itn=top_text,
            text_spoken=text,
            confidence=avg_conf * (top_count / len(results)),
            backend=self.name,
            language=language,
            raw={
                "votes": dict(counter),
                "winner_backends": [r.backend for r in winners],
                "all_backends": [r.backend for r in results],
            },
        )


class LLMJudgeITNStrategy(BaseITNStrategy):
    name = "llm_judge"

    def __init__(self, llm_client: LLMClient, system_prompt: Optional[str] = None):
        self._client = llm_client
        self._system_prompt = system_prompt or (
            "You are an expert text normalization judge. Given multiple ITN "
            "candidates, return JSON: {\"text\": str, \"selected_backend\": str}."
        )

    def apply(
        self,
        backends: List[BaseITNBackend],
        text: str,
        language: str,
    ) -> ITNResult:
        if not text:
            return self._empty_result(language)

        candidates: List[ITNResult] = []
        for backend in backends:
            if not backend.supports_language(language):
                continue
            try:
                candidates.append(backend.normalize(text, language=language))
            except Exception as exc:
                logger.warning(f"ITN backend '{backend.name}' failed: {exc}")

        if not candidates:
            return self._empty_result(language)
        if len(candidates) == 1:
            return candidates[0]

        user = self._format_prompt(text, candidates, language)
        try:
            response = self._client.complete(self._system_prompt, user)
            text_itn, selected = self._parse_response(response.text, candidates)
        except Exception as exc:
            logger.warning(f"LLMJudgeITNStrategy fell back to vote: {exc}")
            return VoteITNStrategy().apply(backends, text, language)

        return ITNResult(
            text_itn=text_itn,
            text_spoken=text,
            confidence=1.0,
            backend=self.name,
            language=language,
            raw={
                "selected_backend": selected,
                "all_backends": [c.backend for c in candidates],
            },
        )

    def _format_prompt(
        self,
        original: str,
        candidates: List[ITNResult],
        language: str,
    ) -> str:
        lines = [f"Language: {language}", f"Original spoken: {original}", "", "Candidates:"]
        for idx, cand in enumerate(candidates):
            lines.append(f"  [{idx}] backend={cand.backend}: {cand.text_itn}")
        lines.append("\nReturn JSON only.")
        return "\n".join(lines)

    def _parse_response(self, text: str, candidates: List[ITNResult]):
        try:
            data = json.loads(text)
            return (
                str(data.get("text", candidates[0].text_itn)),
                str(data.get("selected_backend", candidates[0].backend)),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            return text.strip(), "llm_judge_freeform"


class ITNStrategyFactory:
    @staticmethod
    def build(strategy: str, llm_client: Optional[LLMClient] = None) -> BaseITNStrategy:
        strategy = strategy.lower()
        if strategy == "first_success":
            return FirstSuccessStrategy()
        if strategy == "vote":
            return VoteITNStrategy()
        if strategy == "llm_judge":
            if llm_client is None:
                raise ValueError("llm_judge strategy requires llm_client")
            return LLMJudgeITNStrategy(llm_client)
        raise ValueError(
            f"Unknown ITN strategy '{strategy}'. Valid: first_success, vote, llm_judge"
        )
