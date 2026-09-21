"""Tests for the evaluation comparison logic.

If `results_match` is wrong, every accuracy number the project reports is
wrong - in whichever direction the bug happens to point. These check it
against cases worked out by hand.
"""

from __future__ import annotations

import pytest

from agent.warehouse import QueryResult
from run_eval import (
    answer_mentions,
    gold_figures,
    looks_like_decline,
    normalise_rows,
    results_match,
)


def result(rows: list[tuple], columns: list[str] | None = None) -> QueryResult:
    return QueryResult(
        sql="SELECT 1",
        columns=columns or [f"c{i}" for i in range(len(rows[0]) if rows else 0)],
        rows=rows,
        row_count=len(rows),
        truncated=False,
        elapsed_ms=1.0,
    )


class TestNormalisation:
    def test_row_order_is_ignored(self):
        assert normalise_rows([("a", 1), ("b", 2)]) == normalise_rows([("b", 2), ("a", 1)])

    def test_floats_are_rounded_to_two_places(self):
        assert normalise_rows([(1.234567,)]) == normalise_rows([(1.23,)])

    def test_int_and_float_of_equal_value_match(self):
        """`COUNT(*)` returns an int, `SUM(1)` may return a float."""
        assert normalise_rows([(8,)]) == normalise_rows([(8.0,)])

    def test_booleans_and_their_integer_casts_are_treated_as_equal(self):
        """`customs_hold` is a BOOLEAN; an agent that casts it to INTEGER has
        still answered the question, so the comparison folds the two."""
        assert normalise_rows([(True,)]) == normalise_rows([(1,)])
        assert normalise_rows([(False,)]) == normalise_rows([(0,)])
        assert normalise_rows([(True,)]) != normalise_rows([(False,)])

    def test_whitespace_around_strings_is_ignored(self):
        assert normalise_rows([(" Alexandria ",)]) == normalise_rows([("Alexandria",)])


class TestResultsMatch:
    def test_identical_results_match(self):
        gold = result([("B06", 44.43), ("B04", 36.93)])
        assert results_match(gold, result([("B04", 36.93), ("B06", 44.43)]))

    def test_different_row_counts_do_not_match(self):
        assert not results_match(result([("a",)]), result([("a",), ("b",)]))

    def test_different_values_do_not_match(self):
        assert not results_match(result([("B06", 44.43)]), result([("B06", 44.44)]))

    def test_rounding_difference_within_two_places_matches(self):
        assert results_match(result([("B06", 44.4312)]), result([("B06", 44.43)]))

    def test_column_names_are_irrelevant(self):
        """Many correct queries; the alias the model picked is not the point."""
        gold = result([(8,)], columns=["berths"])
        assert results_match(gold, result([(8,)], columns=["count_star()"]))

    def test_empty_results_match_each_other(self):
        assert results_match(result([]), result([]))


class TestAnswerFigures:
    def test_exact_figure_is_found(self):
        assert answer_mentions("There are 8 berths.", [8.0]) == 1.0

    def test_thousands_separators_do_not_defeat_it(self):
        assert answer_mentions("Throughput was 204,832 TEU.", [204832.0]) == 1.0

    def test_partial_coverage_is_scored_as_a_fraction(self):
        assert answer_mentions("Dwell was 10.27 days.", [10.27, 5.06]) == pytest.approx(0.5)

    def test_wrong_figure_scores_zero(self):
        assert answer_mentions("There are 12 berths.", [8.0]) == 0.0

    def test_small_rounding_is_tolerated(self):
        assert answer_mentions("About 44.4 moves per hour.", [44.43]) == 1.0

    def test_answer_with_no_numbers_scores_zero(self):
        assert answer_mentions("It varies by berth.", [8.0]) == 0.0

    def test_questions_with_no_gold_figures_are_not_penalised(self):
        assert answer_mentions("Anything", []) == 1.0

    def test_gold_figures_skips_booleans_and_nulls(self):
        assert gold_figures([(True, None, 5.0), ("text", 3)]) == [5.0, 3.0]


class TestDeclineDetection:
    @pytest.mark.parametrize(
        "answer",
        [
            "The warehouse does not contain crew data.",
            "I cannot answer this; there is no revenue column.",
            "That information is not recorded in these tables.",
        ],
    )
    def test_refusals_are_recognised(self, answer):
        assert looks_like_decline(answer)

    @pytest.mark.parametrize(
        "answer",
        ["The average crew size was 21.", "Revenue in 2025 was $4.2M."],
    )
    def test_invented_answers_are_not_mistaken_for_refusals(self, answer):
        assert not looks_like_decline(answer)
