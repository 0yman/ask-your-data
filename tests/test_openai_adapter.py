"""Tests for the OpenAI-compatible wire format mapping.

The agent loop is provider-neutral, so the only thing that can break when
swapping vendors is the translation at the edge. These tests drive that
translation directly with fake SDK response objects, so they verify the
mapping without a network call, an API key, or a bill.

The cases that matter are exactly the places OpenAI and Gemini differ:
tool-call ids, JSON-string arguments, and the dedicated `tool` role.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.llm import Message, ToolCall, _from_openai, _to_openai


def fake_response(content=None, tool_calls=None, prompt_tokens=10, output_tokens=5):
    """Mimic the shape of an openai SDK ChatCompletion object."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=output_tokens
        ),
    )


def fake_tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class TestToOpenAI:
    def test_user_message(self):
        assert _to_openai([Message(role="user", content="How many berths?")]) == [
            {"role": "user", "content": "How many berths?"}
        ]

    def test_assistant_tool_call_serialises_arguments_as_a_json_string(self):
        """OpenAI takes arguments as a string; Gemini takes a structured object."""
        call = ToolCall("run_sql", {"sql": "SELECT 1"}, id="abc123")
        payload = _to_openai([Message(role="assistant", tool_calls=[call])])

        entry = payload[0]["tool_calls"][0]
        assert entry["id"] == "abc123"
        assert entry["type"] == "function"
        assert entry["function"]["name"] == "run_sql"
        assert isinstance(entry["function"]["arguments"], str)
        assert json.loads(entry["function"]["arguments"]) == {"sql": "SELECT 1"}

    def test_assistant_without_tool_calls_has_no_tool_calls_key(self):
        payload = _to_openai([Message(role="assistant", content="Done.")])
        assert "tool_calls" not in payload[0]
        assert payload[0]["content"] == "Done."

    def test_tool_result_uses_the_tool_role_and_call_id(self):
        """Gemini matches a result to its call by function name; OpenAI matches
        by id, so the id generated for the call has to survive the round trip."""
        payload = _to_openai(
            [
                Message(
                    role="tool",
                    content="8 rows",
                    tool_name="run_sql",
                    tool_call_id="abc123",
                )
            ]
        )
        assert payload == [
            {"role": "tool", "tool_call_id": "abc123", "content": "8 rows"}
        ]

    def test_a_full_turn_round_trips_in_order(self):
        call = ToolCall("list_tables", {}, id="t1")
        payload = _to_openai(
            [
                Message(role="user", content="q"),
                Message(role="assistant", tool_calls=[call]),
                Message(role="tool", content="dim_berth", tool_name="list_tables", tool_call_id="t1"),
            ]
        )
        assert [entry["role"] for entry in payload] == ["user", "assistant", "tool"]
        assert payload[1]["tool_calls"][0]["id"] == payload[2]["tool_call_id"]


class TestFromOpenAI:
    def test_plain_text_response(self):
        result = _from_openai(fake_response(content="There are 3 berths."))
        assert result.text == "There are 3 berths."
        assert result.tool_calls == []
        assert result.wants_tools is False

    def test_tool_call_arguments_are_parsed_back_to_a_dict(self):
        response = fake_response(
            tool_calls=[fake_tool_call("c1", "run_sql", '{"sql": "SELECT 1"}')]
        )
        result = _from_openai(response)
        assert result.wants_tools
        assert result.tool_calls[0].name == "run_sql"
        assert result.tool_calls[0].arguments == {"sql": "SELECT 1"}
        assert result.tool_calls[0].id == "c1"

    def test_malformed_arguments_do_not_crash_the_run(self):
        """A model can emit invalid JSON. An empty call lets the tool layer
        reply with a usable error the agent can correct, which is strictly
        better than the whole question dying on a parse exception."""
        response = fake_response(
            tool_calls=[fake_tool_call("c1", "run_sql", "{not valid json")]
        )
        result = _from_openai(response)
        assert result.tool_calls[0].arguments == {}
        assert result.tool_calls[0].name == "run_sql"

    def test_parallel_tool_calls_are_all_returned(self):
        response = fake_response(
            tool_calls=[
                fake_tool_call("c1", "list_tables", "{}"),
                fake_tool_call("c2", "describe_table", '{"table": "dim_berth"}'),
            ]
        )
        result = _from_openai(response)
        assert [c.name for c in result.tool_calls] == ["list_tables", "describe_table"]

    def test_usage_is_normalised_to_the_shared_key_names(self):
        result = _from_openai(fake_response(content="x", prompt_tokens=120, output_tokens=8))
        assert result.usage == {"prompt_tokens": 120, "output_tokens": 8}

    def test_empty_choices_does_not_raise(self):
        result = _from_openai(SimpleNamespace(choices=[], usage=None))
        assert result.text is None
        assert result.tool_calls == []


class TestBackendSelection:
    def test_openai_backend_requires_a_key(self, settings):
        from agent.llm import get_llm

        without_key = settings.model_copy(
            update={"llm_backend": "openai", "openai_api_key": None}
        )
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            get_llm(without_key)

    def test_unknown_backend_is_rejected(self, settings):
        from agent.llm import get_llm

        with pytest.raises(ValueError, match="Unknown LLM backend"):
            get_llm(settings.model_copy(update={"llm_backend": "llama"}))


class TestModelFallback:
    """Groq limits each model separately: a busy model hands over to the next."""

    def make(self, settings, behaviour, fallbacks=("backup",)):
        from agent.llm import OpenAICompatibleLLM

        tried = []

        def create(**kwargs):
            tried.append(kwargs["model"])
            outcome = behaviour(kwargs["model"], len(tried))
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        llm = OpenAICompatibleLLM.__new__(OpenAICompatibleLLM)
        llm._settings = settings.model_copy(update={
            "openai_fallback_models": list(fallbacks), "retry_base_delay": 0.0,
        })
        llm._model = "main"
        llm.last_model = "main"
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        llm._client = client
        llm._routes = [(client, "main")] + [(client, m) for m in fallbacks]
        return llm, tried

    def test_a_rate_limited_model_hands_over_at_once(self, settings):
        def behaviour(model, n):
            if model == "main":
                return RuntimeError("429 rate limit reached: tokens per day")
            return fake_response(content="from backup")

        llm, tried = self.make(settings, behaviour)
        response = llm.complete("sys", [Message(role="user", content="q")], [])
        assert response.text == "from backup"
        assert tried == ["main", "backup"]
        assert llm.last_model == "backup"

    def test_the_main_model_is_tried_first_every_time(self, settings):
        llm, tried = self.make(settings, lambda model, n: fake_response(content=model))
        llm.complete("sys", [Message(role="user", content="q")], [])
        llm.complete("sys", [Message(role="user", content="q")], [])
        assert tried == ["main", "main"]

    def test_real_errors_are_not_retried_on_another_model(self, settings):
        llm, tried = self.make(settings, lambda model, n: RuntimeError("400 invalid request"))
        with pytest.raises(RuntimeError, match="400"):
            llm.complete("sys", [Message(role="user", content="q")], [])
        assert tried == ["main"]

    def test_when_every_model_is_busy_it_backs_off_and_gives_up(self, settings):
        llm, tried = self.make(settings, lambda model, n: RuntimeError("503 overloaded"))
        with pytest.raises(RuntimeError, match="503"):
            llm.complete("sys", [Message(role="user", content="q")], [])
        assert tried == ["main", "backup"] * settings.max_retries


def test_backoff_never_waits_longer_than_the_cap():
    from agent.llm import MAX_BACKOFF_SECONDS, backoff_delay

    assert backoff_delay(2.0, 0) < 2.6
    assert all(backoff_delay(2.0, attempt) <= MAX_BACKOFF_SECONDS * 1.25 for attempt in range(20))


def test_a_backup_host_answers_when_the_main_host_is_busy(settings, monkeypatch):
    """Cerebras first, Groq behind it: two hosts, two quotas, one conversation."""
    import openai

    from agent.llm import OpenAICompatibleLLM

    made = {}

    class FakeOpenAI:
        def __init__(self, api_key, base_url, timeout, max_retries):
            made[base_url] = self
            self.calls = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            self.base_url = base_url

        def create(self, **kwargs):
            self.calls.append(kwargs["model"])
            if "main-host" in self.base_url:
                raise RuntimeError("429 rate limit reached: requests per minute")
            return fake_response(content=f"from {kwargs['model']}")

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    llm = OpenAICompatibleLLM(settings.model_copy(update={
        "llm_backend": "openai", "openai_api_key": "k1",
        "openai_base_url": "https://main-host/v1", "openai_model": "qwen-main",
        "openai_backup_api_key": "k2", "openai_backup_base_url": "https://backup-host/v1",
        "openai_backup_model": "qwen-backup",
    }))
    response = llm.complete("sys", [Message(role="user", content="q")], [])
    assert response.text == "from qwen-backup"
    assert made["https://main-host/v1"].calls == ["qwen-main"]
    assert made["https://backup-host/v1"].calls == ["qwen-backup"]
    assert llm.last_model == "qwen-backup"


def test_without_a_backup_key_there_is_no_backup_route(settings, monkeypatch):
    import openai

    from agent.llm import OpenAICompatibleLLM

    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: SimpleNamespace())
    llm = OpenAICompatibleLLM(settings.model_copy(update={
        "llm_backend": "openai", "openai_api_key": "k1", "openai_model": "m",
        "openai_backup_model": "other",  # a model but no key: not a route
    }))
    assert [model for _, model in llm._routes] == ["m"]


def test_a_model_out_for_the_day_fails_at_once_instead_of_retrying(settings):
    from agent.llm import ModelLimitReached

    helper = TestModelFallback()
    daily = "Error code: 429 - Rate limit reached on tokens per day (TPD): Limit 200000, Used 198585"
    llm, tried = helper.make(settings, lambda model, n: RuntimeError(daily), fallbacks=())
    with pytest.raises(ModelLimitReached):
        llm.complete("sys", [Message(role="user", content="q")], [])
    assert tried == ["main"]  # one call, no waiting


def test_a_route_out_for_the_day_is_skipped_for_the_next(settings):
    daily = "429 rate limit: tokens per day (TPD) reached"

    def behaviour(model, n):
        return RuntimeError(daily) if model == "main" else fake_response(content="backup answers")

    llm, tried = TestModelFallback().make(settings, behaviour)
    assert llm.complete("sys", [Message(role="user", content="q")], []).text == "backup answers"
