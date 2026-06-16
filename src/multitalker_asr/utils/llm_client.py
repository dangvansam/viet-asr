import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loguru import logger

from ..configs.llm import LLMConfig


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: Dict[str, int]
    raw: Optional[Dict] = None


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    """Provider-agnostic LLM client backed by LiteLLM (OpenAI-compatible)."""

    def __init__(self, config: LLMConfig):
        self._config = config
        self._completion_fn = None

    @property
    def config(self) -> LLMConfig:
        return self._config

    def _resolve_api_key(self) -> Optional[str]:
        if self._config.api_key:
            return self._config.api_key
        if self._config.api_key_env:
            return os.environ.get(self._config.api_key_env)
        return None

    def _load_completion(self):
        if self._completion_fn is not None:
            return self._completion_fn
        try:
            from litellm import completion
        except ImportError as exc:
            raise LLMClientError(
                "litellm is required for LLMClient. Install via `uv add litellm`."
            ) from exc
        self._completion_fn = completion
        return completion

    def _build_kwargs(self, messages: List[Dict[str, str]], **overrides) -> Dict:
        params: Dict = {
            "model": self._config.litellm_model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "timeout": self._config.timeout,
        }
        api_key = self._resolve_api_key()
        if api_key:
            params["api_key"] = api_key
        if self._config.base_url:
            params["api_base"] = self._config.base_url
        if self._config.extra_headers:
            params["extra_headers"] = dict(self._config.extra_headers)
        params.update(self._config.extra_params)
        params.update(overrides)
        return params

    def complete(
        self,
        system: Optional[str],
        user: str,
        **overrides,
    ) -> LLMResponse:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return self._complete_messages(messages, **overrides)

    def chat(self, messages: List[Dict[str, str]], **overrides) -> LLMResponse:
        return self._complete_messages(messages, **overrides)

    def batch_complete(
        self,
        prompts: List[Tuple[Optional[str], str]],
        **overrides,
    ) -> List[LLMResponse]:
        results: List[LLMResponse] = []
        for system, user in prompts:
            results.append(self.complete(system, user, **overrides))
        return results

    def _complete_messages(
        self, messages: List[Dict[str, str]], **overrides
    ) -> LLMResponse:
        completion = self._load_completion()
        params = self._build_kwargs(messages, **overrides)

        last_exc: Optional[Exception] = None
        delay = 1.0
        for attempt in range(self._config.max_retries + 1):
            try:
                response = completion(**params)
                return self._parse_response(response)
            except Exception as exc:
                last_exc = exc
                if attempt >= self._config.max_retries:
                    break
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/"
                    f"{self._config.max_retries + 1}): {exc}. Retrying in {delay:.1f}s"
                )
                time.sleep(delay)
                delay *= self._config.retry_backoff

        raise LLMClientError(
            f"LLM call failed after {self._config.max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    def _parse_response(self, response) -> LLMResponse:
        try:
            choice = response["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(f"Malformed LLM response: {exc}") from exc

        usage_raw = response.get("usage", {}) if isinstance(response, dict) else {}
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0)),
            "completion_tokens": int(usage_raw.get("completion_tokens", 0)),
            "total_tokens": int(usage_raw.get("total_tokens", 0)),
        }
        model = (
            response.get("model", self._config.model)
            if isinstance(response, dict)
            else self._config.model
        )
        return LLMResponse(
            text=content if content is not None else "",
            model=str(model),
            usage=usage,
            raw=response if isinstance(response, dict) else None,
        )


class LLMClientFactory:
    @staticmethod
    def from_env(prefix: str = "LLM_") -> LLMClient:
        cfg = LLMConfig(
            provider=os.environ.get(f"{prefix}PROVIDER", "openai"),
            model=os.environ.get(f"{prefix}MODEL", "gpt-4o-mini"),
            base_url=os.environ.get(f"{prefix}BASE_URL") or None,
            api_key=os.environ.get(f"{prefix}API_KEY") or None,
            api_key_env=os.environ.get(f"{prefix}API_KEY_ENV", "OPENAI_API_KEY"),
            temperature=float(os.environ.get(f"{prefix}TEMPERATURE", "0.0")),
            max_tokens=int(os.environ.get(f"{prefix}MAX_TOKENS", "1024")),
            timeout=float(os.environ.get(f"{prefix}TIMEOUT", "60.0")),
            max_retries=int(os.environ.get(f"{prefix}MAX_RETRIES", "3")),
        )
        return LLMClient(cfg)

    @staticmethod
    def from_config(config: LLMConfig) -> LLMClient:
        return LLMClient(config)
