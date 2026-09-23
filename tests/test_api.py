"""The web app's API, exercised the way the page uses it."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook

from agent import api
from agent.agent import PortAnalystAgent
from agent.llm import LLMResponse, ScriptedLLM, ToolCall

SALES_CSV = b"region,month,revenue\nNorth,2025-07,1200.5\nSouth,2025-07,980\nNorth,2025-08,1430\n"


def scripted_answer(sql: str, answer: str) -> ScriptedLLM:
    return ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("run_sql", {"sql": sql})]),
        LLMResponse(tool_calls=[ToolCall("final_answer", {"answer": answer})]),
    ])


@pytest.fixture(autouse=True)
def isolated(monkeypatch, settings, tmp_path):
    """Point the app at test settings and a throwaway .env, so no test reads
    the real data folder or writes a real key file."""
    holder = {"settings": settings}
    monkeypatch.setattr(api, "get_settings", lambda **kwargs: holder["settings"])
    monkeypatch.setattr(api, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(api, "_check_key", lambda s: "ok")
    return holder


@pytest.fixture
def client():
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture
def keyless(isolated, settings):
    isolated["settings"] = settings.model_copy(update={"llm_backend": "gemini", "google_api_key": None})
    with TestClient(api.app) as test_client:
        yield test_client


def use_llm(llm):
    """Swap a scripted model into whichever agent is active."""
    agent: PortAnalystAgent = api._store.local.agent
    agent._llm = llm


def upload(client, *files):
    return client.post(
        "/data",
        files=[("files", (name, io.BytesIO(data), "application/octet-stream")) for name, data in files],
    )


def xlsx_bytes(rows) -> bytes:
    workbook = Workbook()
    for row in rows:
        workbook.active.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


class TestPage:
    def test_the_web_app_is_served_at_the_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Ask your data" in response.text


class TestStatus:
    def test_starts_on_the_sample_data(self, client):
        body = client.get("/status").json()
        assert body["dataset"] == "sample"
        assert body["has_key"] is True
        assert ".xlsx" in body["accepted_types"]

    def test_lists_the_sample_tables_with_columns(self, client):
        tables = {t["name"]: t for t in client.get("/tables").json()}
        assert "dim_berth" in tables
        assert any(c["name"] == "crane_count" for c in tables["dim_berth"]["columns"])

    def test_without_a_key_the_page_is_told_so(self, keyless):
        assert keyless.get("/status").json()["has_key"] is False

    def test_health_reports_the_missing_key(self, keyless):
        body = keyless.get("/health").json()
        assert body["warehouse_ready"] is True
        assert body["llm_ready"] is False


class TestKey:
    def test_a_key_saved_from_the_page_is_written_and_used(self, keyless, tmp_path):
        assert keyless.post("/settings/key", json={"key": "AIza-test-key-123456"}).json()["saved"] is True
        assert "GOOGLE_API_KEY=AIza-test-key-123456" in (tmp_path / ".env").read_text()
        assert keyless.get("/status").json()["has_key"] is True

    def test_an_existing_env_file_keeps_its_other_settings(self, keyless, tmp_path):
        (tmp_path / ".env").write_text("AGENT_MAX_STEPS=7\nGOOGLE_API_KEY=old\n", encoding="utf-8")
        keyless.post("/settings/key", json={"key": "AIza-new-key-1234567"})
        text = (tmp_path / ".env").read_text()
        assert "AGENT_MAX_STEPS=7" in text
        assert "GOOGLE_API_KEY=AIza-new-key-1234567" in text
        assert "old" not in text

    def test_a_rejected_key_is_not_saved(self, keyless, tmp_path, monkeypatch):
        monkeypatch.setattr(api, "_check_key", lambda s: "rejected")
        response = keyless.post("/settings/key", json={"key": "not-a-real-key-000"})
        assert response.status_code == 400
        assert "rejected" in response.json()["detail"]
        assert not (tmp_path / ".env").exists()

    def test_a_busy_model_still_saves_the_key(self, keyless, monkeypatch):
        monkeypatch.setattr(api, "_check_key", lambda s: "unknown")
        body = keyless.post("/settings/key", json={"key": "AIza-test-key-123456"}).json()
        assert body["saved"] is True
        assert "busy" in body["note"]

    def test_a_key_cannot_smuggle_in_a_second_setting(self, keyless, tmp_path):
        response = keyless.post("/settings/key", json={"key": "abcdefghijk\nAGENT_MAX_STEPS=999"})
        assert response.status_code in (400, 422, 500)
        assert "AGENT_MAX_STEPS=999" not in ((tmp_path / ".env").read_text() if (tmp_path / ".env").exists() else "")

    def test_keys_can_only_be_set_from_this_computer(self, keyless, monkeypatch):
        monkeypatch.setattr(api, "_is_local", lambda request: False)
        assert keyless.post("/settings/key", json={"key": "AIza-test-key-123456"}).status_code == 403


class TestUploads:
    def test_a_csv_becomes_a_queryable_table_and_is_selected(self, client):
        result = upload(client, ("Sales Q3.csv", SALES_CSV)).json()[0]
        assert result == {"filename": "Sales Q3.csv", "status": "added", "table": "sales_q3",
                          "rows": 3, "columns": 3, "reason": None}
        assert client.get("/status").json()["dataset"] == "mine"
        body = client.post("/sql", json={"sql": "SELECT SUM(revenue) FROM sales_q3"}).json()
        assert body["rows"] == [[3610.5]]

    def test_an_excel_file_becomes_a_table(self, client):
        data = xlsx_bytes([["employee", "salary"], ["Mona", 9000], ["Omar", 12000]])
        result = upload(client, ("staff.xlsx", data)).json()[0]
        assert result["status"] == "added"
        assert client.post("/sql", json={"sql": "SELECT SUM(salary) FROM staff"}).json()["rows"] == [[21000]]

    def test_re_uploading_replaces_the_table(self, client):
        upload(client, ("sales.csv", SALES_CSV))
        upload(client, ("sales.csv", b"region,revenue\nEast,5\n"))
        assert client.post("/sql", json={"sql": "SELECT COUNT(*) FROM sales"}).json()["rows"] == [[1]]

    @pytest.mark.parametrize(
        "name,data,reason",
        [
            ("notes.pdf", b"%PDF", "unsupported"),
            ("empty.csv", b"a,b\n", "no rows"),
            ("broken.xlsx", b"not a workbook", "Excel"),
        ],
    )
    def test_bad_files_are_skipped_with_a_reason(self, client, name, data, reason):
        result = upload(client, (name, data)).json()[0]
        assert result["status"] == "skipped"
        assert reason in result["reason"]

    def test_a_failed_upload_does_not_switch_away_from_the_sample(self, client):
        upload(client, ("notes.pdf", b"%PDF"))
        assert client.get("/status").json()["dataset"] == "sample"

    def test_oversized_files_are_refused(self, client, settings):
        settings.max_upload_mb = 1
        big = b"a,b\n" + b"1,2\n" * (1024 * 1024 // 4 + 10)
        assert "MB limit" in upload(client, ("big.csv", big)).json()[0]["reason"]

    def test_raw_uploads_are_not_kept_once_imported(self, client, settings):
        upload(client, ("sales.csv", SALES_CSV))
        assert not any(settings.uploads_dir.glob("*"))

    def test_removing_a_table(self, client):
        upload(client, ("sales.csv", SALES_CSV))
        assert client.delete("/data/sales").status_code == 200
        assert client.get("/tables").json() == []
        assert client.delete("/data/sales").status_code == 404


class TestDatasets:
    def test_switching_changes_which_tables_are_listed(self, client):
        upload(client, ("sales.csv", SALES_CSV))
        assert [t["name"] for t in client.get("/tables").json()] == ["sales"]
        client.post("/dataset", json={"dataset": "sample"})
        assert "dim_berth" in [t["name"] for t in client.get("/tables").json()]

    def test_my_files_with_nothing_uploaded_is_an_empty_state(self, client):
        client.post("/dataset", json={"dataset": "mine"})
        assert client.get("/tables").json() == []
        response = client.post("/ask", json={"question": "anything"})
        assert response.status_code == 409
        assert "Upload" in response.json()["detail"]

    def test_the_choice_survives_a_restart(self, client):
        upload(client, ("sales.csv", SALES_CSV))
        with TestClient(api.app) as restarted:
            assert restarted.get("/status").json()["dataset"] == "mine"

    def test_the_port_rules_are_only_given_for_the_port_data(self, client):
        assert "Berth productivity" in api._store.local.agent.system_prompt
        upload(client, ("sales.csv", SALES_CSV))
        prompt = api._store.local.agent.system_prompt
        assert "Berth productivity" not in prompt
        assert "sales" in prompt


class TestAsk:
    def test_returns_the_answer_the_steps_and_the_rows(self, client):
        use_llm(scripted_answer("SELECT berth_code FROM dim_berth ORDER BY berth_key", "There are 3 berths."))
        body = client.post("/ask", json={"question": "How many berths?"}).json()
        assert body["answer"] == "There are 3 berths."
        assert body["result"]["columns"] == ["berth_code"]
        assert body["result"]["rows"] == [["B01"], ["B02"], ["B03"]]
        assert [c["name"] for s in body["steps"] for c in s["tool_calls"]] == ["run_sql", "final_answer"]

    def test_questions_about_uploaded_data(self, client):
        upload(client, ("sales.csv", SALES_CSV))
        use_llm(scripted_answer(
            "SELECT region, SUM(revenue) AS total FROM sales GROUP BY region ORDER BY region",
            "North sold 2630.5 and South 980.",
        ))
        body = client.post("/ask", json={"question": "Revenue by region?"}).json()
        assert body["result"]["rows"] == [["North", 2630.5], ["South", 980.0]]

    def test_without_a_key_it_says_how_to_get_one(self, keyless):
        response = keyless.post("/ask", json={"question": "How many berths?"})
        assert response.status_code == 428
        assert "key" in response.json()["detail"]

    def test_an_overloaded_model_is_explained_not_500d(self, client):
        class Busy:
            name = "busy"

            def complete(self, *args):
                raise RuntimeError("503 UNAVAILABLE high demand")

        use_llm(Busy())
        response = client.post("/ask", json={"question": "anything"})
        assert response.status_code == 503
        assert "overloaded" in response.json()["detail"]

    @pytest.mark.parametrize("payload", [{}, {"question": ""}])
    def test_invalid_payloads_are_rejected(self, client, payload):
        assert client.post("/ask", json=payload).status_code == 422


class TestSQL:
    def test_runs_a_select_without_a_key(self, keyless):
        body = keyless.post("/sql", json={"sql": "SELECT berth_code FROM dim_berth ORDER BY 1"}).json()
        assert body["rows"] == [["B01"], ["B02"], ["B03"]]

    def test_writes_are_refused(self, client):
        response = client.post("/sql", json={"sql": "DROP TABLE dim_berth"})
        assert response.status_code == 400
        assert "read-only" in response.json()["detail"]

    def test_uploaded_tables_are_read_only_to_sql_too(self, client):
        upload(client, ("sales.csv", SALES_CSV))
        assert client.post("/sql", json={"sql": "DELETE FROM sales"}).status_code == 400
        assert client.post("/sql", json={"sql": "SELECT COUNT(*) FROM sales"}).json()["rows"] == [[3]]

    def test_file_reading_is_refused(self, client):
        assert client.post("/sql", json={"sql": "SELECT * FROM read_csv('/etc/passwd')"}).status_code == 400


def test_the_port_prompt_is_unchanged(settings):
    """The published evaluation ran on this exact text. Any edit to it has to
    be a deliberate decision that re-runs the evaluation, not an accident."""
    from pathlib import Path

    from agent.agent import build_agent

    expected = (Path(__file__).parent / "port_system_prompt.txt").read_text(encoding="utf-8")
    real = settings.model_copy(update={"db_path": Path(__file__).resolve().parents[1] / "data" / "port.duckdb"})
    if not real.db_path.exists():
        pytest.skip("sample database not built")
    agent = build_agent(real, llm=ScriptedLLM([]))
    assert agent.system_prompt == expected
    agent.warehouse.close()
