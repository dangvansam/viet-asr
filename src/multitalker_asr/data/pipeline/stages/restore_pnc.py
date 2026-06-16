from typing import Dict, List, Optional

from loguru import logger

from ....configs.llm import LLMConfig
from ....utils.llm_client import LLMClient, LLMClientError
from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import PipelineConfig


class RestorePnCStage(BaseStage):
    name = "restore_pnc"

    DEFAULT_SYSTEM = (
        "You are an expert Vietnamese/multilingual text editor. Restore punctuation "
        "and capitalization in the transcript. Do NOT translate, paraphrase, or add "
        "content. Return ONLY the corrected text."
    )

    def __init__(
        self,
        llm_config: LLMConfig,
        system_prompt: Optional[str] = None,
        skip_if_has_punct: bool = True,
        user_template: Optional[str] = None,
    ):
        self._llm_config = llm_config
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self._skip_if_has_punct = skip_if_has_punct
        self._user_template = (
            user_template
            or "Language: {language}\nTranscript:\n{text}\nRestored:"
        )
        self._client: Optional[LLMClient] = None

    def _ensure_loaded(self) -> None:
        if self._client is None:
            self._client = LLMClient(self._llm_config)

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        to_process, done = self._skip_processed(records, checkpoint)
        if not to_process:
            return done

        self._ensure_loaded()
        results: List[Dict] = list(done)
        for record in to_process:
            try:
                enriched = self._process_record(record)
                checkpoint.mark_processed(record["id"], self.name)
                results.append(enriched)
            except Exception as exc:
                logger.error(f"RestorePnCStage failed on {record['id']}: {exc}")
                results.append(record)
        return results

    def _process_record(self, record: Dict) -> Dict:
        text = record.get("text", "")
        if not text:
            return record
        if self._skip_if_has_punct and self._has_punctuation(text):
            return record

        language = record.get("language", "vi")
        user = self._user_template.format(language=language, text=text)
        try:
            response = self._client.complete(self._system_prompt, user)
            restored = response.text.strip()
        except LLMClientError as exc:
            logger.warning(f"PnC restore failed, keeping original: {exc}")
            return record

        enriched = dict(record)
        enriched["text"] = restored
        enriched["pnc_restored"] = True
        return enriched

    def _has_punctuation(self, text: str) -> bool:
        return any(ch in text for ch in ".!?,;:")
