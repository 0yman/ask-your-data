"""Guardrail tests.

This is the security boundary of the whole project: everything the model
writes passes through `validate` before it reaches the database. The blocked
cases below are written as the attacks they are, because that is what they
would be if the guardrails had a gap.
"""

from __future__ import annotations

import pytest

from agent.guardrails import UnsafeSQLError, validate


class TestAllowed:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1",
            "SELECT * FROM dim_berth",
            "SELECT b.berth_code, AVG(c.moves_per_hour) FROM fact_vessel_call c "
            "JOIN dim_berth b USING(berth_key) GROUP BY 1",
            "WITH ranked AS (SELECT berth_key, AVG(moves_per_hour) m FROM fact_vessel_call "
            "GROUP BY 1) SELECT * FROM ranked ORDER BY m DESC",
            "SELECT 1 UNION ALL SELECT 2",
            "SELECT berth_key, COUNT(*) FROM fact_vessel_call GROUP BY 1 HAVING COUNT(*) > 1",
            "SELECT *, ROW_NUMBER() OVER (ORDER BY moves_per_hour DESC) FROM fact_vessel_call",
        ],
    )
    def test_read_only_queries_pass(self, sql):
        assert validate(sql).sql


class TestBlocked:
    @pytest.mark.parametrize(
        "sql,reason",
        [
            ("DROP TABLE dim_berth", "DROP"),
            ("DELETE FROM dim_berth", "DELETE"),
            ("UPDATE dim_berth SET terminal = 'x'", "UPDATE"),
            ("INSERT INTO dim_berth VALUES (9, 'X', 'Y', 'Z', 1.0, 1)", "INSERT"),
            ("CREATE TABLE evil AS SELECT 1", "CREATE"),
            ("ALTER TABLE dim_berth ADD COLUMN x INTEGER", "ALTER"),
            ("TRUNCATE dim_berth", "TRUNCATE"),
        ],
    )
    def test_writes_are_rejected(self, sql, reason):
        with pytest.raises(UnsafeSQLError, match=reason):
            validate(sql)

    def test_statement_stacking_is_rejected(self):
        """The classic injection: a valid query with a second one appended."""
        with pytest.raises(UnsafeSQLError, match="one statement"):
            validate("SELECT * FROM dim_berth; DROP TABLE dim_berth")

    def test_dml_hidden_inside_a_cte_is_rejected(self):
        with pytest.raises(UnsafeSQLError):
            validate("WITH x AS (SELECT 1) INSERT INTO dim_berth SELECT * FROM x")

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM read_csv('/etc/passwd')",
            "SELECT * FROM read_parquet('s3://bucket/secrets.parquet')",
            "SELECT * FROM read_json('C:/Users/secrets.json')",
            "SELECT * FROM glob('/**')",
        ],
    )
    def test_file_reading_functions_are_rejected(self, sql):
        """DuckDB's file functions turn a read-only SQL endpoint into
        arbitrary local file access - the real exfiltration path."""
        with pytest.raises(UnsafeSQLError):
            validate(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "ATTACH 'http://evil.example/db' AS remote",
            "INSTALL httpfs",
            "LOAD httpfs",
            "COPY dim_berth TO '/tmp/leak.csv'",
            "PRAGMA database_list",
        ],
    )
    def test_extension_and_export_statements_are_rejected(self, sql):
        with pytest.raises(UnsafeSQLError):
            validate(sql)

    @pytest.mark.parametrize("sql", ["", "   ", "\n"])
    def test_empty_input_is_rejected(self, sql):
        with pytest.raises(UnsafeSQLError, match="Empty"):
            validate(sql)

    def test_unparseable_sql_says_so(self):
        with pytest.raises(UnsafeSQLError, match="parse"):
            validate("SELECT * FROM (((")


class TestLimitEnforcement:
    def test_missing_limit_is_added(self):
        result = validate("SELECT * FROM dim_berth", max_rows=50)
        assert result.limit_applied
        assert "LIMIT 50" in result.sql.upper()

    def test_existing_limit_is_left_alone(self):
        result = validate("SELECT * FROM dim_berth LIMIT 3", max_rows=50)
        assert not result.limit_applied
        assert "LIMIT 3" in result.sql

    def test_set_operations_get_a_limit_too(self):
        result = validate("SELECT 1 UNION SELECT 2", max_rows=10)
        assert result.limit_applied
        assert "LIMIT 10" in result.sql.upper()

    def test_original_sql_is_preserved_for_logging(self):
        result = validate("SELECT * FROM dim_berth;")
        assert result.original_sql == "SELECT * FROM dim_berth"


class TestErrorMessages:
    def test_rejection_explains_what_to_do_instead(self):
        """The message goes back to the agent, so it has to be actionable."""
        with pytest.raises(UnsafeSQLError) as info:
            validate("DELETE FROM dim_berth")
        assert "SELECT" in str(info.value)

    def test_file_function_rejection_names_the_function(self):
        with pytest.raises(UnsafeSQLError, match="read_csv"):
            validate("SELECT * FROM read_csv('x.csv')")
