"""The public demo: one private workspace per visitor, and a rationed key."""

from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from agent import api
from agent.datasets import DatasetError, import_file
from agent.llm import LLMResponse, ScriptedLLM, ToolCall
from agent.sessions import Quota, QuotaExceeded

SALES_CSV = b"region,revenue\nNorth,1200.5\nSouth,980\n"


class Visitor:
    """A browser tab: remembers the workspace token the server hands it,
    and sends it back, the way the page does."""

    def __init__(self, client: TestClient, address: str = "203.0.113.7") -> None:
        self.client = client
        self.address = address
        self.token: str | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"X-Forwarded-For": self.address}
        if self.token:
            headers["X-Session"] = self.token
        return headers

    def _remember(self, response):
        self.token = response.headers.get("X-Session", self.token)
        return response

    def get(self, path):
        return self._remember(self.client.get(path, headers=self._headers()))

    def post(self, path, **kwargs):
        return self._remember(self.client.post(path, headers=self._headers(), **kwargs))

    def delete(self, path):
        return self._remember(self.client.delete(path, headers=self._headers()))

    def upload(self, name, data):
        return self.post("/data", files=[("files", (name, io.BytesIO(data), "text/csv"))])

    @property
    def session(self):
        return api._store._sessions[self.token]

    def script(self, *answers: str):
        """Give this visitor's agent a model that answers without a network."""
        responses = []
        for answer in answers:
            responses += [
                LLMResponse(tool_calls=[ToolCall("run_sql", {"sql": "SELECT 1 AS one"})]),
                LLMResponse(tool_calls=[ToolCall("final_answer", {"answer": answer})]),
            ]
        self.session.agent._llm = ScriptedLLM(responses)


@pytest.fixture
def public(monkeypatch, settings, tmp_path):
    holder = {"settings": settings.model_copy(update={
        "public_mode": True,
        "sessions_dir": tmp_path / "sessions",
        "public_questions_per_hour": 3,
        "public_questions_per_day": 5,
    })}
    monkeypatch.setattr(api, "get_settings", lambda **kwargs: holder["settings"])
    monkeypatch.setattr(api, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(api, "_check_key", lambda s: "ok")
    return holder


@pytest.fixture
def client(public):
    with TestClient(api.app) as test_client:
        yield test_client


class TestWorkspaces:
    def test_each_visitor_gets_a_token(self, client):
        a, b = Visitor(client), Visitor(client)
        a.get("/status")
        b.get("/status")
        assert a.token and b.token and a.token != b.token
        assert a.get("/status").headers["X-Session"] == a.token  # and keeps it

    def test_one_visitors_files_are_invisible_to_another(self, client):
        a, b = Visitor(client), Visitor(client)
        assert a.upload("sales.csv", SALES_CSV).json()[0]["status"] == "added"
        assert [t["name"] for t in a.get("/tables").json()] == ["sales"]

        assert b.get("/status").json()["dataset"] == "sample"
        b.post("/dataset", json={"dataset": "mine"})
        assert b.get("/tables").json() == []
        assert b.post("/sql", json={"sql": "SELECT * FROM sales"}).status_code == 409

    def test_the_same_file_name_does_not_collide(self, client):
        a, b = Visitor(client), Visitor(client)
        a.upload("sales.csv", SALES_CSV)
        b.upload("sales.csv", b"region,revenue\nEast,5\n")
        assert a.post("/sql", json={"sql": "SELECT COUNT(*) FROM sales"}).json()["rows"] == [[2]]
        assert b.post("/sql", json={"sql": "SELECT COUNT(*) FROM sales"}).json()["rows"] == [[1]]

    def test_the_dataset_choice_is_per_visitor(self, client, public):
        a, b = Visitor(client), Visitor(client)
        a.upload("sales.csv", SALES_CSV)
        assert a.get("/status").json()["dataset"] == "mine"
        assert b.get("/status").json()["dataset"] == "sample"
        assert not public["settings"].state_path.exists()  # nothing global is written

    def test_idle_workspaces_are_deleted_with_their_files(self, client, public, monkeypatch):
        a = Visitor(client)
        a.upload("sales.csv", SALES_CSV)
        folder = a.session.settings.user_db_path.parent
        assert folder.exists()
        a.session.last_seen -= public["settings"].session_idle_minutes * 60 + 1
        Visitor(client).get("/status")  # any request sweeps
        assert not folder.exists()
        assert a.token not in api._store._sessions

    def test_an_expired_page_is_told_rather_than_answered(self, client):
        a = Visitor(client)
        a.upload("sales.csv", SALES_CSV)
        api._store._sessions.pop(a.token)  # as if it had expired
        response = a.post("/ask", json={"question": "Total revenue?"})
        assert response.status_code == 409
        assert "idle" in response.json()["detail"]
        # The page has its new token now and carries on normally.
        a.script("Fine.")
        assert a.post("/ask", json={"question": "Total revenue?"}).status_code == 200

    def test_a_forged_token_is_not_trusted(self, client):
        a = Visitor(client)
        a.token = "../../etc/passwd"
        a.get("/status")
        assert a.token != "../../etc/passwd"
        assert a.token in api._store._sessions

    def test_the_oldest_workspace_makes_room_when_full(self, client, public):
        public["settings"].max_sessions = 2
        first, second, third = Visitor(client), Visitor(client), Visitor(client)
        first.get("/status")
        second.get("/status")
        third.get("/status")
        assert set(api._store._sessions) == {second.token, third.token}


class TestUploadLimits:
    def test_the_public_size_limit_applies(self, client, public):
        public["settings"].public_max_upload_mb = 1
        big = b"a,b\n" + b"1,2\n" * (1024 * 1024 // 4 + 10)
        result = Visitor(client).upload("big.csv", big).json()[0]
        assert "1 MB limit" in result["reason"]

    def test_a_visitor_holds_a_limited_number_of_tables(self, client, public):
        public["settings"].public_max_tables = 2
        a = Visitor(client)
        a.upload("one.csv", SALES_CSV)
        a.upload("two.csv", SALES_CSV)
        assert "2 tables" in a.upload("three.csv", SALES_CSV).json()[0]["reason"]
        assert a.upload("two.csv", SALES_CSV).json()[0]["status"] == "added"  # replacing is fine


class TestKeys:
    def test_a_visitors_key_is_theirs_alone_and_never_written(self, public, tmp_path):
        public["settings"] = public["settings"].model_copy(update={"llm_backend": "gemini", "google_api_key": None})
        with TestClient(api.app) as keyless:
            a, b = Visitor(keyless), Visitor(keyless)
            assert a.get("/status").json()["has_key"] is False
            assert a.post("/settings/key", json={"key": "AIza-visitor-key-1234"}).json()["saved"] is True
            assert a.get("/status").json()["has_key"] is True
            assert b.get("/status").json()["has_key"] is False
            assert not (tmp_path / ".env").exists()

    def test_keys_can_be_added_from_anywhere(self, client, monkeypatch):
        monkeypatch.setattr(api, "_is_local", lambda request: False)
        assert Visitor(client).post("/settings/key", json={"key": "AIza-visitor-key-1234"}).status_code == 200


class TestQuota:
    def test_questions_on_the_server_key_are_rationed_per_visitor(self, client):
        a = Visitor(client)
        a.get("/status")
        a.script("1", "2", "3", "4")
        assert a.get("/status").json()["questions_left"] == 3
        for _ in range(3):
            assert a.post("/ask", json={"question": "q"}).status_code == 200
        response = a.post("/ask", json={"question": "q"})
        assert response.status_code == 429
        assert "own free Gemini key" in response.json()["detail"]
        # Someone else, elsewhere, is unaffected.
        b = Visitor(client, address="198.51.100.2")
        b.get("/status")
        b.script("fine")
        assert b.post("/ask", json={"question": "q"}).status_code == 200

    def test_a_new_tab_does_not_reset_the_count(self, client):
        for _ in range(3):
            tab = Visitor(client)
            tab.get("/status")
            tab.script("ok")
            assert tab.post("/ask", json={"question": "q"}).status_code == 200
        tab = Visitor(client)
        tab.get("/status")
        assert tab.post("/ask", json={"question": "q"}).status_code == 429

    def test_visitors_with_their_own_key_are_not_counted(self, client):
        a = Visitor(client)
        a.post("/settings/key", json={"key": "AIza-visitor-key-1234"})
        a.script(*"12345")
        for _ in range(5):
            assert a.post("/ask", json={"question": "q"}).status_code == 200
        assert a.get("/status").json()["questions_left"] is None

    def test_questions_that_cannot_run_are_not_counted(self, client):
        a = Visitor(client)
        a.post("/dataset", json={"dataset": "mine"})
        assert a.post("/ask", json={"question": "q"}).status_code == 409
        assert a.get("/status").json()["questions_left"] == 3


class TestQuotaUnit:
    def test_the_hourly_window_slides(self):
        now = [1_000_000.0]
        quota = Quota(per_hour=2, per_day=100, clock=lambda: now[0])
        quota.take("a")
        quota.take("a")
        with pytest.raises(QuotaExceeded, match="about"):
            quota.take("a")
        now[0] += 3601
        quota.take("a")
        assert quota.left("a") == 1

    def test_the_daily_total_covers_everyone_and_resets(self):
        now = [86400 * 20_000 + 10.0]  # just after midnight UTC
        quota = Quota(per_hour=100, per_day=3, clock=lambda: now[0])
        for address in "abc":
            quota.take(address)
        with pytest.raises(QuotaExceeded, match="Today"):
            quota.take("d")
        assert quota.left("d") == 0
        now[0] += 86400
        quota.take("d")


def test_the_local_app_is_unchanged_by_all_this(settings, monkeypatch, tmp_path):
    """No token, no header, one workspace: the app on your own computer."""
    monkeypatch.setattr(api, "get_settings", lambda **kwargs: settings)
    with TestClient(api.app) as local:
        response = local.get("/status")
        assert "X-Session" not in response.headers
        assert response.json()["public"] is False
        assert response.json()["questions_left"] is None


def test_a_zip_bomb_workbook_is_refused_before_it_is_opened(tmp_path):
    path = tmp_path / "bomb.xlsx"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", b"\0" * 50_000_000)
    with pytest.raises(DatasetError, match="unpacks"):
        import_file(path, tmp_path / "out.duckdb")
