"""Choosing the model: the picker, per-model limits, and the progress stream."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from agent import api
from agent.llm import LLMResponse, ScriptedLLM, ToolCall
from agent.sessions import GEMINI

from .test_public import Visitor


def answering(*answers):
    responses = []
    for answer in answers:
        responses += [
            LLMResponse(tool_calls=[ToolCall("run_sql", {"sql": "SELECT 1 AS one"})]),
            LLMResponse(tool_calls=[ToolCall("final_answer", {"answer": answer})]),
        ]
    return ScriptedLLM(responses)


@pytest.fixture
def two_models(monkeypatch, settings, tmp_path):
    """A public server with both catalog models' keys set."""
    holder = {"settings": settings.model_copy(update={
        "public_mode": True,
        "sessions_dir": tmp_path / "sessions",
        "groq_api_key": "gsk-test", "mistral_api_key": "mistral-test",
        "public_questions_per_hour": 50,
    })}
    monkeypatch.setattr(api, "get_settings", lambda **kwargs: holder["settings"])
    monkeypatch.setattr(api, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(api, "_check_key", lambda s: "ok")
    with TestClient(api.app) as client:
        yield client


class TestPicker:
    def test_both_models_are_offered_and_the_first_is_the_default(self, two_models):
        body = Visitor(two_models).get("/status").json()
        assert [m["id"] for m in body["models"]] == ["ministral-14b", "qwen-groq"]
        assert body["model"] == "ministral-14b"
        assert body["engine_label"] == "Ministral 14B · Mistral"

    def test_switching_changes_what_answers_for_this_visitor_only(self, two_models):
        a, b = Visitor(two_models), Visitor(two_models, address="198.51.100.9")
        a.get("/status")
        b.get("/status")
        assert a.post("/model", json={"model": "qwen-groq"}).json()["engine_label"] == "Qwen 3.8 27B · Groq"
        assert a.session.settings.openai_base_url == "https://api.groq.com/openai/v1"
        assert a.session.settings.openai_model == "qwen/qwen3.8-27b"
        assert a.session.settings.openai_api_key == "gsk-test"
        assert b.get("/status").json()["model"] == "ministral-14b"

    def test_an_unknown_model_is_refused(self, two_models):
        assert Visitor(two_models).post("/model", json={"model": "gpt-99"}).status_code == 404

    def test_gemini_is_offered_only_with_the_visitors_own_key(self, two_models):
        a = Visitor(two_models)
        assert GEMINI not in [m["id"] for m in a.get("/status").json()["models"]]
        assert a.post("/model", json={"model": GEMINI}).status_code == 404
        a.post("/settings/key", json={"key": "AIza-visitor-key-1234"})
        body = a.get("/status").json()
        assert body["model"] == GEMINI
        assert GEMINI in [m["id"] for m in body["models"]]
        # ... and they can switch away and back.
        a.post("/model", json={"model": "qwen-groq"})
        assert a.post("/model", json={"model": GEMINI}).json()["model"] == GEMINI

    def test_without_catalog_keys_there_is_nothing_to_pick(self, monkeypatch, settings):
        monkeypatch.setattr(api, "get_settings", lambda **kwargs: settings)
        with TestClient(api.app) as client:
            body = client.get("/status").json()
            assert body["models"] == []
            assert body["model"] is None


class TestPerModelLimits:
    def test_each_model_has_its_own_daily_allowance(self, two_models, monkeypatch):
        from agent import config

        catalog = tuple(o.model_copy(update={"per_day": 1}) for o in config.MODEL_CATALOG)
        monkeypatch.setattr(config, "MODEL_CATALOG", catalog)
        a = Visitor(two_models)
        a.get("/status")
        a.session.agent._llm = answering("m1", "m2")
        assert a.post("/ask", json={"question": "q"}).status_code == 200
        response = a.post("/ask", json={"question": "q"})
        assert response.status_code == 429
        assert "Ministral 14B" in response.json()["detail"]
        # Qwen's allowance is untouched.
        a.post("/model", json={"model": "qwen-groq"})
        a.session.agent._llm = answering("q1")
        assert a.post("/ask", json={"question": "q"}).status_code == 200

    def test_the_answer_names_the_model_that_gave_it(self, two_models):
        a = Visitor(two_models)
        a.get("/status")
        a.post("/model", json={"model": "qwen-groq"})
        a.session.agent._llm = answering("fine")
        assert a.post("/ask", json={"question": "q"}).json()["model"] == "Qwen 3.8 27B · Groq"


def stream(client, question, headers=None):
    response = client.post("/ask/stream", json={"question": question}, headers=headers or {})
    assert response.status_code == 200
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


class TestStream:
    def test_steps_arrive_before_the_answer(self, settings, monkeypatch):
        monkeypatch.setattr(api, "get_settings", lambda **kwargs: settings)
        with TestClient(api.app) as client:
            api._store.local.agent._llm = ScriptedLLM([
                LLMResponse(tool_calls=[ToolCall("run_sql", {"sql": "SELECT berth_code FROM dim_berth"})]),
                LLMResponse(tool_calls=[ToolCall("final_answer", {"answer": "There are 3 berths."})]),
            ])
            events = stream(client, "How many berths?")
        kinds = [e["type"] for e in events]
        assert kinds == ["thinking", "tool", "thinking", "result"]
        assert events[1]["name"] == "run_sql" and events[1]["ok"] and events[1]["rows"] == 3
        assert events[-1]["data"]["answer"] == "There are 3 berths."
        assert events[-1]["data"]["result"]["row_count"] == 3

    def test_a_refusal_arrives_as_an_error_line(self, settings, monkeypatch):
        monkeypatch.setattr(api, "get_settings", lambda **kwargs: settings)
        with TestClient(api.app) as client:
            client.post("/dataset", json={"dataset": "mine"})
            events = stream(client, "anything")
        assert events[-1]["type"] == "error"
        assert events[-1]["status"] == 409
        assert "Upload" in events[-1]["detail"]

    def test_waiting_out_a_rate_limit_is_announced(self, settings, monkeypatch):
        monkeypatch.setattr(api, "get_settings", lambda **kwargs: settings)

        class Patient(ScriptedLLM):
            def complete(self, system, messages, tools):
                if not self.calls:
                    self._waiting(12.4)   # what a real client does before sleeping
                return super().complete(system, messages, tools)

        with TestClient(api.app) as client:
            api._store.local.agent._llm = Patient([
                LLMResponse(tool_calls=[ToolCall("final_answer", {"answer": "ok"})]),
            ])
            events = stream(client, "q")
        assert {"type": "wait", "seconds": 12} in events

    def test_the_local_model_choice_survives_a_restart(self, settings, monkeypatch):
        with_keys = settings.model_copy(update={"groq_api_key": "g", "mistral_api_key": "m"})
        monkeypatch.setattr(api, "get_settings", lambda **kwargs: with_keys)
        with TestClient(api.app) as client:
            client.post("/model", json={"model": "qwen-groq"})
        with TestClient(api.app) as restarted:
            assert restarted.get("/status").json()["model"] == "qwen-groq"


def test_a_model_out_of_allowance_is_shown_resting(two_models):
    from agent.llm import ModelLimitReached

    class OutForTheDay(ScriptedLLM):
        def complete(self, system, messages, tools):
            raise ModelLimitReached("429 tokens per day (TPD)", retry_after=300)

    a = Visitor(two_models)
    a.get("/status")
    a.post("/model", json={"model": "qwen-groq"})
    a.session.agent._llm = OutForTheDay([])
    response = a.post("/ask", json={"question": "q"})
    assert response.status_code == 429
    assert "Qwen 3.8 27B has used its free allowance" in response.json()["detail"]
    models = {m["id"]: m for m in a.get("/status").json()["models"]}
    assert models["qwen-groq"]["resting_minutes"] == 5
    assert models["ministral-14b"]["resting_minutes"] is None
