"""LLM backends behind one tool-calling interface.

The agent loop is written against the provider-neutral types in this module,
not against any SDK. That is what lets the same loop run on Gemini in
production and on a scripted stub in the test suite - the loop cannot tell the
difference, so testing it proves something about the real thing.
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .config import Settings

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant", "tool"]


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    #: Opaque provider state that has to survive the round trip.
    #:
    #: Gemini 3.x attaches a `thought_signature` to every function call and
    #: rejects the next request if it is not sent back, so a neutral wrapper
    #: that keeps only name and arguments silently breaks multi-turn tool use.
    #: The loop never reads this; only the provider adapter does.
    provider_state: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Message:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_name: str | None = None      # set on role="tool"
    tool_call_id: str | None = None


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]        # JSON Schema


@dataclass(slots=True)
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(ABC):
    name: str

    @abstractmethod
    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse: ...


class ScriptedLLM(LLMClient):
    """Returns a prepared sequence of responses.

    Used by the tests. Scripting the model is deliberate: it makes the
    scenarios that matter - a malformed query followed by a correction, a
    model that never stops calling tools - reproducible, which a "clever" mock
    guessing at intent could never be.
    """

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self.name = "scripted"
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if not self._responses:
            return LLMResponse(text="No further scripted responses.")
        return self._responses.pop(0)


class GeminiLLM(LLMClient):
    """Gemini with function calling, on the free tier.

    Free-tier limits are per-minute as well as per-day, so a long agent run
    can hit 429 mid-conversation. Retries use exponential backoff with jitter
    rather than failing the whole question.
    """

    def __init__(self, settings: Settings) -> None:
        # The key first: someone without one needs to hear about the key, not
        # about a package they would only need once they had one.
        api_key = settings.require_api_key()
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("The Gemini SDK is not installed: pip install google-genai") from exc

        self._settings = settings
        self._client = genai.Client(
            api_key=api_key,
            # The SDK retries internally by default. Left on, a 503 would be
            # retried by the SDK *and* by `_call_with_retry` below, so five
            # configured attempts become twenty-five with compounding backoff
            # and a request that appears to hang. One retry layer, and it is
            # this module's, because only it knows the agent's step budget.
            http_options=types.HttpOptions(
                timeout=settings.request_timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self._model = settings.gemini_model
        self.name = self._model

    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self._settings.temperature,
            max_output_tokens=self._settings.max_output_tokens,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters_json_schema=tool.parameters,
                        )
                        for tool in tools
                    ]
                )
            ]
            if tools
            else None,
        )

        response = self._call_with_retry(_to_gemini(messages), config)
        return _from_gemini(response)

    def _call_with_retry(self, contents, config):
        settings = self._settings
        for attempt in range(settings.max_retries):
            try:
                return self._client.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
            except Exception as exc:
                if not is_retryable(exc) or attempt == settings.max_retries - 1:
                    raise
                delay = backoff_delay(settings.retry_base_delay, attempt)
                logger.warning("Gemini call failed (%s); retrying in %.1fs", exc, delay)
                time.sleep(delay)
        raise RuntimeError("Unreachable retry state")


class OpenAICompatibleLLM(LLMClient):
    """Any OpenAI-format chat-completions endpoint.

    The same wire format is served by OpenAI, Groq, Together, Fireworks,
    OpenRouter and a local Ollama or vLLM, so one adapter plus a `base_url`
    covers all of them. That is the practical payoff of keeping the agent loop
    provider-neutral: switching vendors is a config change, not a rewrite.

    The differences from Gemini that actually matter here:

    * tools are declared as `{"type": "function", "function": {...}}` rather
      than bare function declarations;
    * a tool result goes back under a dedicated `tool` role keyed by
      `tool_call_id`, where Gemini matches on function name - so the ids this
      module generates have to survive the round trip;
    * arguments arrive as a JSON *string*, not a parsed object.
    """

    def __init__(self, settings: Settings) -> None:
        api_key = settings.require_openai_key()  # before the import; see GeminiLLM
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The OpenAI SDK is not installed: pip install openai") from exc

        self._settings = settings
        self._client = OpenAI(
            api_key=api_key,
            base_url=settings.openai_base_url or None,
            timeout=settings.request_timeout_ms / 1000,
            # One retry layer, for the same reason as the Gemini client: the
            # backoff below is the one that knows the agent's step budget.
            max_retries=0,
        )
        self._model = settings.openai_model
        self.name = self._model
        self.last_model = self._model

    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        payload = [{"role": "system", "content": system}] + _to_openai(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": payload,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_output_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
        return _from_openai(self._call_with_retry(kwargs))

    def _call_with_retry(self, kwargs: dict[str, Any]):
        """Try each model in turn; back off only when all of them are busy.

        Rate limits on hosts like Groq are per model, so a second model is a
        second quota: when the first is out of tokens for the minute - or the
        day - the next answers at once instead of the visitor waiting.
        """
        settings = self._settings
        models = [self._model] + [m for m in settings.openai_fallback_models if m != self._model]
        last: Exception | None = None
        for attempt in range(settings.max_retries):
            for model in models:
                try:
                    response = self._client.chat.completions.create(**{**kwargs, "model": model})
                    self.last_model = model
                    return response
                except Exception as exc:
                    if not is_retryable(exc):
                        raise
                    last = exc
                    logger.warning("%s unavailable (%s)", model, str(exc)[:160])
            if attempt == settings.max_retries - 1:
                break
            # A per-minute token limit says exactly how long to wait: waiting
            # less spends a retry, waiting more keeps a visitor staring.
            server = _retry_after(last)
            delay = min(server + random.uniform(0.5, 1.5), MAX_BACKOFF_SECONDS) if server                 else backoff_delay(settings.retry_base_delay, attempt)
            logger.warning("All models busy; retrying in %.1fs", delay)
            time.sleep(delay)
        assert last is not None
        raise last


# No single wait longer than this. Doubling from 2s passes a minute by the
# sixth retry; beyond that a visitor has given up and the quota window has
# long since moved.
MAX_BACKOFF_SECONDS = 30.0


def backoff_delay(base: float, attempt: int) -> float:
    delay = min(base * (2**attempt), MAX_BACKOFF_SECONDS)
    return delay + random.uniform(0, delay * 0.25)  # jitter avoids lockstep retries


def _retry_after(exc: Exception) -> float:
    """Seconds the server asked us to wait, from a Retry-After header; 0 if none."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        return max(0.0, float(headers.get("retry-after", 0)))
    except (TypeError, ValueError):
        return 0.0


def _to_openai(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Map neutral messages onto the OpenAI chat-completions shape."""
    payload: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "user":
            payload.append({"role": "user", "content": message.content or ""})
        elif message.role == "assistant":
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or None,
            }
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        # Arguments go over the wire as a JSON string here,
                        # unlike Gemini which takes a structured object.
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
            payload.append(entry)
        elif message.role == "tool":
            payload.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id or "",
                    "content": message.content or "",
                }
            )
    return payload


def _from_openai(response) -> LLMResponse:
    choice = response.choices[0] if response.choices else None
    message = getattr(choice, "message", None)

    tool_calls: list[ToolCall] = []
    for raw in (getattr(message, "tool_calls", None) or []):
        try:
            arguments = json.loads(raw.function.arguments or "{}")
        except json.JSONDecodeError:
            # A model can emit malformed JSON. Surfacing it as an empty call
            # lets the tool layer reply with a usable error instead of the
            # whole run dying on a parse exception.
            logger.warning("Could not parse tool arguments: %r", raw.function.arguments)
            arguments = {}
        tool_calls.append(
            ToolCall(name=raw.function.name, arguments=arguments, id=raw.id)
        )

    usage = {}
    if getattr(response, "usage", None):
        usage = {
            "prompt_tokens": response.usage.prompt_tokens or 0,
            "output_tokens": response.usage.completion_tokens or 0,
        }

    return LLMResponse(
        text=(getattr(message, "content", None) or None),
        tool_calls=tool_calls,
        usage=usage,
    )


def _to_gemini(messages: Sequence[Message]):
    """Map neutral messages onto Gemini's Content/Part structure.

    Gemini has no "tool" role: a tool result is a function-response part sent
    back under the user role, matched to its call by function name.
    """
    from google.genai import types

    contents = []
    for message in messages:
        if message.role == "user":
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=message.content or "")])
            )
        elif message.role == "assistant":
            parts = []
            if message.content:
                parts.append(types.Part.from_text(text=message.content))
            for call in message.tool_calls:
                part = types.Part.from_function_call(name=call.name, args=call.arguments)
                signature = call.provider_state.get("thought_signature")
                if signature is not None:
                    # Required by Gemini 3.x; dropping it is a 400 on the turn
                    # after any tool call.
                    part.thought_signature = signature
                parts.append(part)
            if parts:
                contents.append(types.Content(role="model", parts=parts))
        elif message.role == "tool":
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=message.tool_name or "unknown",
                            response={"result": message.content or ""},
                        )
                    ],
                )
            )
    return contents


def _from_gemini(response) -> LLMResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for candidate in response.candidates or []:
        for part in (candidate.content.parts if candidate.content else []) or []:
            if getattr(part, "function_call", None):
                call = part.function_call
                signature = getattr(part, "thought_signature", None)
                tool_calls.append(
                    ToolCall(
                        name=call.name,
                        arguments=dict(call.args or {}),
                        id=call.id or uuid.uuid4().hex[:8],
                        provider_state=(
                            {"thought_signature": signature} if signature else {}
                        ),
                    )
                )
            elif getattr(part, "text", None):
                text_parts.append(part.text)

    usage = {}
    if getattr(response, "usage_metadata", None):
        usage = {
            "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
            "output_tokens": response.usage_metadata.candidates_token_count or 0,
        }

    return LLMResponse(
        text="\n".join(text_parts).strip() or None,
        tool_calls=tool_calls,
        usage=usage,
    )


_RETRYABLE_MARKERS = (
    "429", "rate limit", "resource_exhausted", "503", "500",
    "unavailable", "deadline", "internal error", "overloaded",
)


def is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _RETRYABLE_MARKERS)


def get_llm(settings: Settings) -> LLMClient:
    if settings.llm_backend == "gemini":
        return GeminiLLM(settings)
    if settings.llm_backend == "openai":
        return OpenAICompatibleLLM(settings)
    if settings.llm_backend == "scripted":
        return ScriptedLLM([])
    raise ValueError(f"Unknown LLM backend: {settings.llm_backend!r}")


def tool_result_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)
