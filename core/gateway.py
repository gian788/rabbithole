import os
import sys
import time
import abc
from dataclasses import dataclass, field
from typing import Optional

import openai
import anthropic

MODEL_PRICING_LEDGER: dict[str, dict[str, dict[str, float]]] = {
    "openai": {
        "text-embedding-3-small": {"input": 0.00002 / 1000, "output": 0.0},
        "gpt-4o-mini":            {"input": 0.00015 / 1000, "output": 0.00060 / 1000},
    },
    "anthropic": {
        "claude-haiku-4-5-20251001": {"input": 0.00080 / 1000, "output": 0.00400 / 1000},
    },
}


@dataclass
class ModelResponse:
    text_content:      Optional[str]         = None
    embedding_vector:  Optional[list[float]] = None
    input_tokens:      int                   = 0
    output_tokens:     int                   = 0
    latency_ms:        int                   = 0
    cost:              float                 = 0.0
    model:             str                   = ""
    provider:          str                   = ""


class BaseAIProvider(abc.ABC):
    @abc.abstractmethod
    def generate_completion(self, prompt: str, system_prompt: str, model: str) -> ModelResponse:
        pass

    @abc.abstractmethod
    def generate_embedding(self, text: str, model: str) -> ModelResponse:
        pass


class OpenAIProvider(BaseAIProvider):
    def __init__(self) -> None:
        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate_completion(self, prompt: str, system_prompt: str, model: str) -> ModelResponse:
        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
        )
        return ModelResponse(
            text_content=resp.choices[0].message.content,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            model=model,
            provider="openai",
        )

    def stream_completion(self, prompt: str, system_prompt: str, model: str):
        """Yields (token: str, usage) tuples. Usage is non-None only on the final chunk."""
        stream = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            yield token, chunk.usage  # usage is None until the last chunk

    def generate_embedding(self, text: str, model: str = "text-embedding-3-small") -> ModelResponse:
        resp = self._client.embeddings.create(input=text, model=model)
        return ModelResponse(
            embedding_vector=resp.data[0].embedding,
            input_tokens=resp.usage.prompt_tokens,
            model=model,
            provider="openai",
        )


class AnthropicProvider(BaseAIProvider):
    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate_completion(self, prompt: str, system_prompt: str, model: str) -> ModelResponse:
        resp = self._client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return ModelResponse(
            text_content=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=model,
            provider="anthropic",
        )

    def generate_embedding(self, text: str, model: str = "") -> ModelResponse:
        raise NotImplementedError("Anthropic does not provide an embedding API")


class ModelGateway:
    """Wraps AI providers with timing, cost calculation, and telemetry logging."""

    def __init__(self, db_conn=None) -> None:
        self._providers: dict[str, BaseAIProvider] = {
            "openai":    OpenAIProvider(),
            "anthropic": AnthropicProvider(),
        }
        self.db_conn = db_conn  # optional — telemetry skipped when None

    def _calculate_cost(self, provider: str, model: str,
                        input_tokens: int, output_tokens: int) -> float:
        pricing = MODEL_PRICING_LEDGER.get(provider, {}).get(model, {})
        return (
            pricing.get("input", 0.0) * input_tokens
            + pricing.get("output", 0.0) * output_tokens
        )

    def _log_telemetry(self, transaction_type: str,
                       response: ModelResponse, associated_id: str) -> None:
        if not self.db_conn:
            return
        try:
            with self.db_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_telemetry
                        (transaction_type, provider, model, input_tokens, output_tokens,
                         latency_ms, cost, associated_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        transaction_type,
                        response.provider,
                        response.model,
                        response.input_tokens,
                        response.output_tokens,
                        response.latency_ms,
                        response.cost,
                        associated_id,
                    ),
                )
            self.db_conn.commit()
        except Exception as exc:
            print(f"[telemetry] log failed: {exc}", file=sys.stderr)

    def get_completion(
        self,
        prompt: str,
        system_prompt: str,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        associated_id: str = "",
    ) -> ModelResponse:
        t0 = time.time()
        response = self._providers[provider].generate_completion(prompt, system_prompt, model)
        response.latency_ms = int((time.time() - t0) * 1000)
        response.cost = self._calculate_cost(
            provider, model, response.input_tokens, response.output_tokens
        )
        self._log_telemetry("completion", response, associated_id)
        return response

    def stream_completion(
        self,
        prompt: str,
        system_prompt: str,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        associated_id: str = "",
    ):
        """Generator that yields token strings. Logs telemetry after stream ends."""
        t0 = time.time()
        input_tokens = output_tokens = 0
        collected: list[str] = []
        try:
            for token, usage in self._providers[provider].stream_completion(
                prompt, system_prompt, model
            ):
                if token:
                    collected.append(token)
                    yield token
                if usage:
                    input_tokens  = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
        finally:
            latency_ms = int((time.time() - t0) * 1000)
            cost = self._calculate_cost(provider, model, input_tokens, output_tokens)
            resp = ModelResponse(
                text_content="".join(collected),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost=cost,
                model=model,
                provider=provider,
            )
            self._log_telemetry("completion", resp, associated_id)

    def get_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small",
        provider: str = "openai",
        associated_id: str = "",
    ) -> ModelResponse:
        t0 = time.time()
        response = self._providers[provider].generate_embedding(text, model)
        response.latency_ms = int((time.time() - t0) * 1000)
        response.cost = self._calculate_cost(provider, model, response.input_tokens, 0)
        self._log_telemetry("embedding", response, associated_id)
        return response
