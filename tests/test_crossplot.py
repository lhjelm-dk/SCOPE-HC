"""Tests for the cross-plot statistics used by the Results-page explorer.

These pin the two things most easily got wrong: the percentile convention (P10 is the
HIGH case under probability-of-exceedance, which is the app default) and the binning,
where the numbers in the exported table have to match what the plot overlay draws.
"""

import numpy as np
import pandas as pd
import pytest

from scopehc.crossplot import (
    MAX_BINS,
    bin_edges_for,
    bin_statistics,
    correlation_matrix,
    fit_line,
    pair_statistics,
    percentiles_by_convention,
    rank_drivers,
    suggest_bin_width,
)


# --- percentile convention ---------------------------------------------------------


def test_exceedance_convention_puts_the_high_case_in_p10():
    values = np.arange(0.0, 101.0)  # 0..100

    exceedance = percentiles_by_convention(values, use_exceedance=True)
    traditional = percentiles_by_convention(values, use_exceedance=False)

    assert exceedance["P10"] == pytest.approx(90.0)
    assert exceedance["P90"] == pytest.approx(10.0)
    assert traditional["P10"] == pytest.approx(10.0)
    assert traditional["P90"] == pytest.approx(90.0)
    # The median does not move with the convention.
    assert exceedance["P50"] == traditional["P50"] == pytest.approx(50.0)


def test_percentiles_of_empty_input_are_nan_not_an_exception():
    stats = percentiles_by_convention(np.array([]), use_exceedance=True)
    assert all(np.isnan(v) for v in stats.values())


# --- binning -----------------------------------------------------------------------


def test_bin_edges_anchor_on_a_round_multiple_of_the_width():
    # Depths 2203..2287 with a 10 m bin must start at 2200, not at 2203.
    edges = bin_edges_for(np.array([2203.0, 2287.0]), bin_width=10.0)
    assert edges[0] == pytest.approx(2200.0)
    assert edges[-1] == pytest.approx(2290.0)
    assert np.allclose(np.diff(edges), 10.0)


def test_bin_means_match_a_hand_computed_case():
    # Two 10 m bins: [2200,2210) holds y = 1,2,3 and [2210,2220] holds y = 10,20.
    x = np.array([2200.0, 2205.0, 2209.9, 2210.0, 2219.0])
    y = np.array([1.0, 2.0, 3.0, 10.0, 20.0])

    table = bin_statistics(x, y, bin_width=10.0)

    assert list(table["count"]) == [3, 2]
    assert table["mean"].tolist() == pytest.approx([2.0, 15.0])
    assert table["bin_min"].tolist() == pytest.approx([2200.0, 2210.0])
    # std is the sample standard deviation (ddof=1), consistent with summarize_array.
    assert table["std"].iloc[0] == pytest.approx(1.0)


def test_mean_per_unit_x_is_the_mean_divided_by_the_bin_width():
    x = np.array([2200.0, 2205.0, 2209.0])
    y = np.array([10.0, 20.0, 30.0])

    table = bin_statistics(x, y, bin_width=10.0)

    assert table["mean"].iloc[0] == pytest.approx(20.0)
    assert table["mean_per_unit_x"].iloc[0] == pytest.approx(2.0)


def test_the_largest_value_lands_in_the_last_bin_not_in_one_of_its_own():
    # 100 is exactly on an edge. It must close the last bin rather than open a new one.
    x = np.linspace(0.0, 100.0, 11)
    table = bin_statistics(x, x, bin_width=10.0)

    assert len(table) == 10
    assert int(table["count"].sum()) == x.size
    assert table["bin_max"].iloc[-1] == pytest.approx(100.0)


def test_empty_bins_are_kept_as_rows_with_count_zero():
    # Nothing between 20 and 40, so the middle bins must still appear.
    x = np.array([5.0, 15.0, 45.0])
    y = np.array([1.0, 2.0, 3.0])

    table = bin_statistics(x, y, bin_width=10.0)

    assert len(table) == 5  # 0-10, 10-20, 20-30, 30-40, 40-50
    assert list(table["count"]) == [1, 1, 0, 0, 1]
    assert table["mean"].isna().tolist() == [False, False, True, True, False]


def test_bin_percentiles_follow_the_requested_convention():
    rng = np.random.default_rng(7)
    x = np.full(500, 2205.0)  # one bin
    y = rng.normal(100.0, 10.0, 500)

    exceedance = bin_statistics(x, y, bin_width=10.0, use_exceedance=True)
    traditional = bin_statistics(x, y, bin_width=10.0, use_exceedance=False)

    assert exceedance["P10"].iloc[0] > exceedance["P90"].iloc[0]
    assert traditional["P10"].iloc[0] < traditional["P90"].iloc[0]
    assert exceedance["P10"].iloc[0] == pytest.approx(traditional["P90"].iloc[0])


def test_too_many_bins_is_refused_rather_than_rendered():
    x = np.array([0.0, 10_000.0])
    with pytest.raises(ValueError, match="use a wider bin"):
        bin_statistics(x, x, bin_width=0.1)
    assert MAX_BINS > 0


def test_non_positive_bin_width_is_refused():
    with pytest.raises(ValueError, match="positive"):
        bin_statistics(np.array([1.0, 2.0]), np.array([1.0, 2.0]), bin_width=0.0)


def test_binning_drops_pairs_outside_an_explicit_origin():
    x = np.array([5.0, 15.0, 25.0])
    y = np.array([1.0, 2.0, 3.0])

    table = bin_statistics(x, y, bin_width=10.0, origin=10.0)

    # The 5.0 pair is below the origin and must not be folded into the first bin.
    assert int(table["count"].sum()) == 2
    assert table["bin_min"].iloc[0] == pytest.approx(10.0)


def test_nan_pairs_are_dropped_before_binning():
    x = np.array([2201.0, np.nan, 2203.0, 2204.0])
    y = np.array([1.0, 2.0, np.nan, 5.0])

    table = bin_statistics(x, y, bin_width=10.0)

    assert int(table["count"].sum()) == 2  # only the first and last pair survive
    assert table["mean"].iloc[0] == pytest.approx(3.0)


# --- pair statistics ---------------------------------------------------------------


def test_a_known_linear_relation_recovers_r_and_slope():
    x = np.linspace(1.0, 100.0, 200)
    y = 3.0 * x + 7.0

    stats = pair_statistics(x, y)

    assert stats["n"] == 200
    assert stats["pearson_r"] == pytest.approx(1.0)
    assert stats["spearman_rho"] == pytest.approx(1.0)
    assert stats["r_squared"] == pytest.approx(1.0)
    assert stats["slope"] == pytest.approx(3.0)
    assert stats["intercept"] == pytest.approx(7.0)


def test_an_inverse_relation_gives_a_negative_correlation():
    x = np.linspace(1.0, 50.0, 100)
    stats = pair_statistics(x, -2.0 * x)

    assert stats["pearson_r"] == pytest.approx(-1.0)
    assert stats["slope"] == pytest.approx(-2.0)


def test_a_monotonic_but_curved_relation_scores_higher_on_spearman():
    x = np.linspace(1.0, 10.0, 200)
    y = x**4

    stats = pair_statistics(x, y)

    assert stats["spearman_rho"] == pytest.approx(1.0)
    assert stats["pearson_r"] < stats["spearman_rho"]
    # A power law is straight in log-log, so the log fit recovers the exponent.
    assert stats["log_slope"] == pytest.approx(4.0)
    assert stats["log_r_squared"] == pytest.approx(1.0)


def test_constant_input_yields_nan_rather_than_raising():
    x = np.full(50, 3.0)
    y = np.linspace(0.0, 1.0, 50)

    stats = pair_statistics(x, y)

    assert np.isnan(stats["pearson_r"])
    assert np.isnan(stats["spearman_rho"])
    assert np.isnan(stats["slope"])
    assert stats["n"] == 50


def test_too_few_points_yields_nan_statistics():
    stats = pair_statistics([1.0, 2.0], [1.0, 2.0])
    assert stats["n"] == 2
    assert np.isnan(stats["pearson_r"])


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="same length"):
        pair_statistics([1.0, 2.0, 3.0], [1.0, 2.0])


# --- fit line ----------------------------------------------------------------------


def test_fit_line_lies_on_the_underlying_relation():
    x = np.linspace(1.0, 10.0, 50)
    y = 2.0 * x + 1.0

    xs, ys = fit_line(x, y)

    assert ys == pytest.approx(2.0 * xs + 1.0)


def test_fit_line_in_log_space_is_returned_in_data_coordinates():
    x = np.linspace(1.0, 100.0, 100)
    y = 5.0 * x**2

    xs, ys = fit_line(x, y, log_x=True, log_y=True)

    assert np.all(xs > 0) and np.all(ys > 0)
    assert ys == pytest.approx(5.0 * xs**2, rel=1e-6)


def test_fit_line_returns_none_when_there_is_nothing_to_fit():
    assert fit_line([1.0, 2.0], [1.0, 2.0]) is None
    assert fit_line(np.full(10, 2.0), np.linspace(0, 1, 10)) is None
    # Log axes with non-positive data leave too few points.
    assert fit_line(np.array([-1.0, -2.0, -3.0]), np.array([1.0, 2.0, 3.0]), log_x=True) is None


# --- matrix and drivers ------------------------------------------------------------


def _frame():
    """A frame where depth is deliberately the stronger driver of STOIIP.

    Depth contributes a spread of 0.05 x 50 = 2.5 to STOIIP against porosity's
    50 x 0.02 = 1.0, so the expected driver ranking is depth first.
    """
    rng = np.random.default_rng(42)
    depth = rng.normal(2200.0, 50.0, 400)
    por = rng.normal(0.2, 0.02, 400)
    return pd.DataFrame(
        {
            "Depth": depth,
            "Porosity": por,
            "STOIIP": 0.05 * (2400.0 - depth) + 50.0 * por + rng.normal(0, 0.2, 400),
            "Constant": np.full(400, 1.0),
        }
    )


def test_correlation_matrix_skips_columns_that_do_not_vary():
    df = _frame()

    matrix = correlation_matrix(df, ["Depth", "Porosity", "STOIIP", "Constant"])

    assert "Constant" not in matrix.columns
    assert list(matrix.columns) == ["Depth", "Porosity", "STOIIP"]
    assert np.allclose(np.diag(matrix.to_numpy()), 1.0)
    assert matrix.loc["Depth", "STOIIP"] < 0  # deeper contact, less oil


def test_correlation_matrix_needs_two_usable_columns():
    df = _frame()
    assert correlation_matrix(df, ["Depth"]).empty
    assert correlation_matrix(df, ["Constant", "Depth"]).empty


def test_rank_drivers_sorts_by_absolute_strength():
    df = _frame()

    ranked = rank_drivers(df, "STOIIP", ["Depth", "Porosity", "Constant"])

    assert list(ranked["variable"]) == ["Depth", "Porosity"]  # Constant is unusable
    assert ranked["abs_correlation"].is_monotonic_decreasing
    assert ranked["correlation"].iloc[0] < 0


def test_rank_drivers_on_a_missing_target_returns_an_empty_frame():
    ranked = rank_drivers(_frame(), "NotThere", ["Depth"])
    assert ranked.empty
    assert list(ranked.columns) == ["variable", "correlation", "abs_correlation"]


# --- bin width suggestion ----------------------------------------------------------


def test_suggested_bin_width_is_ten_metres_for_a_depth_range():
    assert suggest_bin_width(np.array([2150.0, 2350.0]), depth_like=True) == pytest.approx(10.0)


def test_suggested_bin_width_is_a_round_number_for_other_ranges():
    width = suggest_bin_width(np.linspace(0.0, 1.0, 100), target_bins=20)
    assert width in (0.05, 0.1)


def test_suggested_bin_width_survives_degenerate_input():
    assert suggest_bin_width(np.array([])) > 0
    assert suggest_bin_width(np.full(10, 5.0)) > 0
