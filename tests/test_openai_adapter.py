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
