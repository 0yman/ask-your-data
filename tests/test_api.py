from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent import api
from agent.agent import PortAnalystAgent
from agent.llm import LLMResponse, ScriptedLLM, ToolCall


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, settings):
    """Point the app at the test warehouse.

    `lifespan` calls `get_settings()` itself, which would otherwise open the
    real `data/port.duckdb` - or fail if a developer has not built it.
    """
    monkeypatch.setattr(api, "get_settings", lambda **kwargs: settings)


@pytest.fixture
def client(warehouse, settings):
    llm = ScriptedLLM(
        [
            LLMResponse(tool_calls=[ToolCall("run_sql", {"sql": "SELECT COUNT(*) FROM dim_berth"})]),
            LLMResponse(tool_calls=[ToolCall("final_answer", {"answer": "There are 3 berths."})]),
        ]
    )
    with TestClient(api.app) as test_client:
        api._state["agent"] = PortAnalystAgent(warehouse, llm, settings)
        api._state["error"] = None
        yield test_client
    api._state["agent"] = None


@pytest.fixture
def keyless_client(warehouse, settings):
    """An agent with no LLM client and no API key available."""
    no_key = settings.model_copy(update={"google_api_key": None, "llm_backend": "gemini"})
    with TestClient(api.app) as test_client:
        api._state["agent"] = PortAnalystAgent(warehouse, None, no_key)
        api._state["error"] = None
        yield test_client
    api._state["agent"] = None


class TestHealth:
    def test_reports_ok_when_everything_is_up(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["warehouse_ready"] and body["llm_ready"]

    def test_missing_key_is_degraded_not_down(self, keyless_client):
        """The warehouse endpoints still work, so the service is not down."""
        body = keyless_client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["warehouse_ready"] is True
        assert body["llm_ready"] is False
        assert "GOOGLE_API_KEY" in body["error"]


class TestSchema:
    def test_lists_tables_and_columns(self, client):
        body = client.get("/schema").json()
        names = {table["name"] for table in body["tables"]}
        assert {"dim_berth", "fact_vessel_call"} <= names
        berth = next(t for t in body["tables"] if t["name"] == "dim_berth")
        assert any(column["name"] == "crane_count" for column in berth["columns"])

    def test_works_without_an_api_key(self, keyless_client):
        assert keyless_client.get("/schema").status_code == 200


class TestSQL:
    def test_runs_a_select(self, client):
        body = client.post("/sql", json={"sql": "SELECT berth_code FROM dim_berth ORDER BY 1"}).json()
        assert body["rows"] == [["B01"], ["B02"], ["B03"]]
        assert body["limit_applied"] is True

    def test_write_is_rejected_with_a_reason(self, client):
        response = client.post("/sql", json={"sql": "DROP TABLE dim_berth"})
        assert response.status_code == 400
        assert "read-only" in response.json()["detail"]

    def test_file_reading_function_is_rejected(self, client):
        response = client.post("/sql", json={"sql": "SELECT * FROM read_csv('/etc/passwd')"})
        assert response.status_code == 400

    def test_broken_sql_is_a_client_error(self, client):
        response = client.post("/sql", json={"sql": "SELECT nope FROM dim_berth"})
        assert response.status_code == 400

    def test_works_without_an_api_key(self, keyless_client):
        assert keyless_client.post("/sql", json={"sql": "SELECT 1"}).status_code == 200

    @pytest.mark.parametrize("payload", [{}, {"sql": ""}])
    def test_invalid_payloads_are_rejected(self, client, payload):
        assert client.post("/sql", json=payload).status_code == 422


class TestAsk:
    def test_returns_the_answer_and_the_full_trace(self, client):
        body = client.post("/ask", json={"question": "How many berths?"}).json()
        assert body["succeeded"] is True
        assert body["answer"] == "There are 3 berths."
        assert body["stop_reason"] == "final_answer"
        assert body["sql_queries"]
        names = [call["name"] for step in body["steps"] for call in step["tool_calls"]]
        assert names == ["run_sql", "final_answer"]

    def test_requires_an_api_key(self, keyless_client):
        response = keyless_client.post("/ask", json={"question": "How many berths?"})
        assert response.status_code == 503
        assert "GOOGLE_API_KEY" in response.json()["detail"]

    @pytest.mark.parametrize("payload", [{}, {"question": ""}])
    def test_invalid_payloads_are_rejected(self, client, payload):
        assert client.post("/ask", json=payload).status_code == 422
