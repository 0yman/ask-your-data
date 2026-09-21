"""Shared fixtures.

The whole suite runs on a tiny warehouse built into a temp directory and a
scripted LLM, so it needs no API key, no network, and never touches the real
`data/port.duckdb`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from agent.config import Settings, get_settings  # noqa: E402
from agent.warehouse import Warehouse  # noqa: E402


@pytest.fixture(scope="session")
def tiny_db(tmp_path_factory) -> Path:
    """A miniature star schema with known values.

    Small enough that every expected number in the tests can be worked out by
    hand, which is the only way an assertion about an aggregate means anything.
    """
    path = tmp_path_factory.mktemp("warehouse") / "test.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute(
        """
        CREATE TABLE dim_berth (
            berth_key INTEGER, berth_code VARCHAR, terminal VARCHAR,
            harbour VARCHAR, max_draft_m DOUBLE, crane_count INTEGER
        );
        INSERT INTO dim_berth VALUES
            (1, 'B01', 'West Terminal', 'West', 12.0, 2),
            (2, 'B02', 'West Terminal', 'West', 14.0, 4),
            (3, 'B03', 'East Quay',     'East',  9.0, 1);

        CREATE TABLE dim_cargo_type (
            cargo_type_key INTEGER, cargo_type VARCHAR,
            is_hazardous BOOLEAN, requires_reefer BOOLEAN
        );
        INSERT INTO dim_cargo_type VALUES
            (1, 'General Cargo', FALSE, FALSE),
            (2, 'Hazardous Chemicals', TRUE, FALSE);

        CREATE TABLE fact_vessel_call (
            call_id INTEGER, vessel_key INTEGER, berth_key INTEGER,
            arrival_date_key INTEGER, waiting_hours DOUBLE,
            berth_hours DOUBLE, total_moves INTEGER, moves_per_hour DOUBLE
        );
        INSERT INTO fact_vessel_call VALUES
            (1, 1, 1, 20240101,  2.0, 10.0, 100, 10.0),
            (2, 1, 2, 20240102,  4.0, 10.0, 300, 30.0),
            (3, 2, 2, 20240103,  6.0, 10.0, 200, 20.0),
            (4, 2, 3, 20240104, 20.0, 10.0,  50,  5.0);

        CREATE TABLE fact_container_movement (
            movement_id INTEGER, call_id INTEGER, berth_key INTEGER,
            cargo_type_key INTEGER, container_count INTEGER, teu INTEGER,
            dwell_days DOUBLE, customs_hold BOOLEAN, demurrage_usd DOUBLE
        );
        INSERT INTO fact_container_movement VALUES
            (1, 1, 1, 1, 10,  15,  3.0, FALSE,    0.0),
            (2, 2, 2, 1, 20,  30,  4.0, FALSE,    0.0),
            (3, 3, 2, 2, 30,  45, 10.0, TRUE,  1000.0),
            (4, 4, 3, 2, 40,  60,  8.0, TRUE,   500.0);
        """
    )
    connection.execute("CHECKPOINT")
    connection.close()
    return path


@pytest.fixture
def settings(tiny_db: Path) -> Settings:
    return get_settings(
        llm_backend="scripted",
        db_path=tiny_db,
        max_steps=6,
        max_sql_retries=2,
        max_rows=100,
    )


@pytest.fixture
def warehouse(settings: Settings) -> Warehouse:
    wh = Warehouse(
        settings.db_path,
        max_rows=settings.max_rows,
        timeout_seconds=settings.query_timeout_seconds,
    )
    yield wh
    wh.close()
