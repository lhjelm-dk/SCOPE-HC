"""Tests for summarize_array, which implements the percentile convention for the whole app.

Every P-number the app shows - summary tables, histogram overlays, exports, reports -
comes from here, and under the default probability-of-exceedance convention P10 is the
HIGH case and P90 the low one, the opposite of the textbook ordering. Nothing pinned
that until now, which is how the GRV depth-area plot came to draw bands from raw
np.percentile and label the 10th percentile "P10".

Also covers p10_p90_by_convention, the per-axis helper those bands now use.
"""

import numpy as np
import pytest
import streamlit as st

from scopehc.utils import p10_p90_by_convention, summarize_array

# 0..100, so the Nth percentile is exactly N and a mislabelled percentile is obvious.
RAMP = np.arange(0.0, 101.0)

LADDER = ["P1", "P5", "P10", "P25", "P50", "P75", "P90", "P95", "P99"]


@pytest.fixture
def exceedance(monkeypatch):
    """Session state with the app default: P10 is the high case."""
    monkeypatch.setattr(st, "session_state", {"percentile_exceedance": True})


@pytest.fixture
def non_exceedance(monkeypatch):
    """Session state with the textbook convention: P10 is the low case."""
    monkeypatch.setattr(st, "session_state", {"percentile_exceedance": False})


# --- the convention ----------------------------------------------------------------


def test_exceedance_swaps_p10_and_p90_and_leaves_p50_alone(exceedance):
    stats = summarize_array(RAMP)

    assert stats["P10"] == pytest.approx(90.0)
    assert stats["P90"] == pytest.approx(10.0)
    assert stats["P50"] == pytest.approx(50.0)


def test_non_exceedance_gives_the_textbook_ordering(non_exceedance):
    stats = summarize_array(RAMP)

    assert stats["P10"] == pytest.approx(10.0)
    assert stats["P90"] == pytest.approx(90.0)
    assert stats["P50"] == pytest.approx(50.0)


def test_the_whole_ladder_descends_under_exceedance(exceedance):
    """P1 is the largest and P99 the smallest: a P-number is a chance of exceeding."""
    values = [summarize_array(RAMP)[k] for k in LADDER]

    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(99.0)   # P1  -> 99th percentile
    assert values[-1] == pytest.approx(1.0)   # P99 -> 1st percentile


def test_the_whole_ladder_ascends_under_non_exceedance(non_exceedance):
    values = [summarize_array(RAMP)[k] for k in LADDER]

    assert values == sorted(values)
    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(99.0)


def test_the_two_conventions_are_mirror_images_of_each_other(monkeypatch):
    """Same numbers, opposite labels - nothing is computed differently."""
    monkeypatch.setattr(st, "session_state", {"percentile_exceedance": True})
    exc = summarize_array(RAMP)
    monkeypatch.setattr(st, "session_state", {"percentile_exceedance": False})
    trad = summarize_array(RAMP)

    for low, high in [("P1", "P99"), ("P5", "P95"), ("P10", "P90"), ("P25", "P75")]:
        assert exc[low] == pytest.approx(trad[high])
        assert exc[high] == pytest.approx(trad[low])
    assert exc["P50"] == pytest.approx(trad["P50"])
    # The convention touches percentiles only.
    for key in ["mean", "min", "max", "std_dev", "variance", "skewness"]:
        assert exc[key] == pytest.approx(trad[key])


def test_exceedance_is_the_fallback_when_session_state_is_unreachable(monkeypatch):
    """Outside a Streamlit run the sidebar toggle cannot be read; the default stands."""

    class NoContext:
        def get(self, *args, **kwargs):
            raise RuntimeError("missing ScriptRunContext")

    monkeypatch.setattr(st, "session_state", NoContext())

    stats = summarize_array(RAMP)
    assert stats["P10"] == pytest.approx(90.0)
    assert stats["P90"] == pytest.approx(10.0)


# --- degenerate input --------------------------------------------------------------


@pytest.mark.parametrize("x", [np.array([]), np.array([np.nan, np.nan])])
def test_an_array_with_no_usable_values_returns_an_empty_dict(exceedance, x):
    """Callers test the dict for emptiness; raising here would break every page."""
    assert summarize_array(x) == {}


def test_a_single_sample_is_its_own_every_percentile(exceedance):
    stats = summarize_array(np.array([7.5]))

    assert all(stats[k] == pytest.approx(7.5) for k in LADDER)
    assert stats["mean"] == stats["mode"] == pytest.approx(7.5)
    # No spread is defined from one sample, and ddof=1 must not divide by zero.
    assert stats["std_dev"] == 0.0
    assert stats["variance"] == 0.0
    assert stats["skewness"] == 0.0


def test_a_constant_array_has_no_spread_and_no_skew(exceedance):
    """The zero-std and zero-range guards: skew(constant) is otherwise a nan."""
    stats = summarize_array(np.full(100, 3.25))

    assert all(stats[k] == pytest.approx(3.25) for k in LADDER)
    assert stats["std_dev"] == 0.0
    assert stats["variance"] == pytest.approx(0.0)
    assert stats["skewness"] == 0.0
    assert np.isfinite(stats["mode"])


def test_nans_are_dropped_rather_than_poisoning_the_statistics(exceedance):
    with_nans = np.concatenate([RAMP, np.full(10, np.nan)])

    assert summarize_array(with_nans) == pytest.approx(summarize_array(RAMP))


# --- the per-axis helper used by the GRV depth-area bands ---------------------------


def test_the_axis_helper_puts_the_high_curve_in_p10_under_exceedance():
    # Three realizations of a two-point curve; percentiles run across realizations.
    realizations = np.array([[0.0, 10.0], [50.0, 60.0], [100.0, 110.0]])

    p10, p90 = p10_p90_by_convention(realizations, use_exceedance=True, axis=0)

    assert np.all(p10 >= p90)
    assert p10 == pytest.approx(np.percentile(realizations, 90, axis=0))
    assert p90 == pytest.approx(np.percentile(realizations, 10, axis=0))


def test_the_axis_helper_follows_the_textbook_ordering_when_asked():
    realizations = np.array([[0.0, 10.0], [50.0, 60.0], [100.0, 110.0]])

    p10, p90 = p10_p90_by_convention(realizations, use_exceedance=False, axis=0)

    assert np.all(p10 <= p90)
    assert p10 == pytest.approx(np.percentile(realizations, 10, axis=0))


def test_the_axis_helper_spans_the_same_band_either_way():
    """Only the names swap: the band is bounded by the same two curves."""
    realizations = np.array([[0.0, 10.0], [50.0, 60.0], [100.0, 110.0]])

    a = p10_p90_by_convention(realizations, use_exceedance=True, axis=0)
    b = p10_p90_by_convention(realizations, use_exceedance=False, axis=0)

    assert a[0] == pytest.approx(b[1])
    assert a[1] == pytest.approx(b[0])


def test_the_axis_helper_agrees_with_summarize_array_on_a_flat_array(exceedance):
    stats = summarize_array(RAMP)
    p10, p90 = p10_p90_by_convention(RAMP, use_exceedance=True)

    assert float(p10) == pytest.approx(stats["P10"])
    assert float(p90) == pytest.approx(stats["P90"])
