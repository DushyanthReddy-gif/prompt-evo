"""OpenAI-compatible chat client.

Works with any endpoint speaking the OpenAI chat-completions protocol
via ``base_url``. Includes rate-limit-aware retry handling for providers
such as Google Gemini.
"""
from __future__ import annotations

import os
import time
import re

from prompt_evo.config import LLMSection
from prompt_evo.llm.base import LLMError, LLMResponse, OptimizerModel, WorkerModel


class _ChatClient:
    def __init__(self, cfg: LLMSection) -> None:
        self.cfg = cfg
        self._client = None

        # Gemini free-tier currently allows only a small number of
        # requests per minute. Keep calls spaced out to reduce 429s.
        self._last_call_time = 0.0

    def _get(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise LLMError(
                    "the 'openai' package is required: pip install openai"
                ) from e

            key = os.environ.get(self.cfg.api_key_env)

            if not key:
                raise LLMError(
                    f"environment variable {self.cfg.api_key_env} is not set"
                )

            self._client = OpenAI(
                api_key=key,
                base_url=self.cfg.base_url,
                timeout=self.cfg.timeout_s,
                max_retries=0,
            )

        return self._client

    def _wait_between_calls(self) -> None:
        """Keep requests below the Gemini free-tier RPM limit."""

        # ~13 seconds between requests gives approximately
        # 4-5 requests per minute.
        min_interval = 13.0

        now = time.time()
        elapsed = now - self._last_call_time

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_call_time = time.time()

    def _retry_delay(self, error: Exception, attempt: int) -> float:
        """Determine how long to wait before retrying an API error."""

        # First try to get an explicit retry delay from the exception.
        # Google Gemini may include text such as:
        # "Please retry in 48.26561393s."
        message = str(error)

        match = re.search(
            r"retry in\s+([0-9]+(?:\.[0-9]+)?)s",
            message,
            re.IGNORECASE,
        )

        if match:
            return float(match.group(1)) + 1.0

        # Also handle a Retry-After HTTP header if the SDK exposes it.
        response = getattr(error, "response", None)

        if response is not None:
            headers = getattr(response, "headers", None)

            if headers:
                retry_after = headers.get("retry-after")

                if retry_after:
                    try:
                        return float(retry_after) + 1.0
                    except ValueError:
                        pass

        # Generic exponential backoff.
        return min(2 ** attempt, 30)

    def chat(self, system: str, user: str) -> LLMResponse:
        client = self._get()
        last = None

        for attempt in range(self.cfg.max_retries + 1):

            # Pace every request.
            self._wait_between_calls()

            t0 = time.time()

            try:
                r = client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    messages=[
                        {
                            "role": "system",
                            "content": system,
                        },
                        {
                            "role": "user",
                            "content": user,
                        },
                    ],
                )

                usage = r.usage

                return LLMResponse(
                    text=(r.choices[0].message.content or ""),
                    prompt_tokens=getattr(
                        usage,
                        "prompt_tokens",
                        0,
                    )
                    or 0,
                    completion_tokens=getattr(
                        usage,
                        "completion_tokens",
                        0,
                    )
                    or 0,
                    call_id=getattr(r, "id", None),
                    latency=time.time() - t0,
                )

            except Exception as e:
                last = e

                status = getattr(e, "status_code", None)

                # Non-retryable 4xx errors.
                if (
                    status is not None
                    and status < 500
                    and status != 429
                ):
                    break

                # Rate limit.
                if status == 429:
                    delay = self._retry_delay(e, attempt)

                    # Print a useful message instead of silently waiting.
                    print(
                        f"Rate limit reached. Waiting {delay:.1f}s "
                        f"before retry {attempt + 1}/"
                        f"{self.cfg.max_retries}..."
                    )

                    time.sleep(delay)

                # Server error / timeout.
                else:
                    delay = min(2 ** attempt, 30)
                    time.sleep(delay)

        raise LLMError(
            f"LLM call failed after retries: {last}"
        )


class OpenAIWorker(WorkerModel):
    def __init__(self, cfg: LLMSection) -> None:
        self.name = cfg.model
        self.temperature = cfg.temperature
        self._chat = _ChatClient(cfg)

    def generate(self, prompt: str, input: str) -> LLMResponse:
        return self._chat.chat(prompt, input)


class OpenAIOptimizer(OptimizerModel):
    def __init__(self, cfg: LLMSection) -> None:
        self.name = cfg.model
        self._chat = _ChatClient(cfg)

    def complete(self, system: str, user: str) -> LLMResponse:
        return self._chat.chat(system, user)