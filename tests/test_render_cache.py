"""Tests for the Results-page render cache.

The cache exists for speed, so the thing worth pinning is that it never changes
what the page says. These tests render the real Results page through Streamlit's
own AppTest runner and compare the elements between a cold render and a cached
one - charts by their serialised Plotly spec, tables by value.

A stale volume shown as current would be far worse than a slow page, so the
cases below are the ones that could produce one:

* a rerun with nothing changed must be byte-identical, and must actually hit the
  cache (otherwise the comparison proves nothing);
* a cached render must match what the same page builds with the cache bypassed;
* changing any sampled input must redraw every figure and table that input
  feeds, named one by one;
* flipping the percentile convention must move every P10/P90 on the page, in the
  same session;
* the THR figure must not collect another three percentile lines per rerun,
  which is what mutating a cached figure would look like.

Asserting on *named* figures matters more than it looks. An earlier version of
this file compared whole-page signatures, and that passed even with the data
removed from the figure cache key: the tables are keyed separately and moved, so
the page changed while six of the seven histograms stayed stale - a histogram
contradicting the table printed under it. Each assertion below names the element
it is about.
"""

from __future__ import annotations

import numpy as np
import pytest

from scopehc.ui import explorer as explorer_mod
from scopehc.ui import render_cache
from scopehc.ui import results as results_mod

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="needs the Streamlit test runner"
).AppTest

# Small enough to keep the suite quick; identity does not depend on trial count.
N = 1_500

SCRIPT = """
import scopehc.ui.results as results
results.render()
"""

# The seven histogram/CDF figures, by the title that appears in their spec.
STOIIP = "Oil in Place (STOIIP)"
GIIP = "Gas in Place (GIIP)"
OIL = "Recoverable Oil"
FREE_GAS = "Recoverable Free Gas"
ASSOC_GAS = "Recoverable Associated Gas"
COND = "Recoverable Condensate"
THR = "Total Hydrocarbon Resource (THR) Distribution"
PERCENTILE_CHARTS = [STOIIP, GIIP, OIL, FREE_GAS, ASSOC_GAS, COND, THR]

# Which figures each sampled input feeds, from the arithmetic in
# compute.compute_results. In-place volumes carry no recovery factor; condensate
# comes from FREE gas, so it follows RF_gas; associated gas follows the oil side.
INPUT_TARGETS = [
    (("sNtG",), 0.80, PERCENTILE_CHARTS),
    (("sp",), 1.10, PERCENTILE_CHARTS),
    (("sGRV_oil_m3",), 1.30, [STOIIP, OIL, ASSOC_GAS, THR]),
    (("sGRV_gas_m3",), 1.30, [GIIP, FREE_GAS, COND, THR]),
    (("Shc_oil",), 0.90, [STOIIP, OIL, ASSOC_GAS, THR]),
    (("Shc_gas",), 0.90, [GIIP, FREE_GAS, COND, THR]),
    (("sInvBo",), 1.05, [STOIIP, OIL, ASSOC_GAS, THR]),
    (("sBg",), 1.15, [GIIP, FREE_GAS, COND, THR]),
    (("sRF_oil",), 0.50, [OIL, ASSOC_GAS, THR]),
    (("sRF_gas",), 0.90, [FREE_GAS, COND, THR]),
    (("sRF_assoc_gas",), 0.70, [ASSOC_GAS, THR]),
    (("sGOR",), 1.40, [ASSOC_GAS, THR]),
    (("sCY",), 1.30, [COND, THR]),
    (("sRF_cond",), 0.60, [COND, THR]),
]


def make_state(n: int = N, seed: int = 7) -> dict:
    """A plausible Oil + Gas sample set, enough for every section to render."""
    rng = np.random.default_rng(seed)
    grv = rng.lognormal(np.log(400e6), 0.45, n)
    f_oil = np.clip(rng.normal(0.55, 0.08, n), 0.05, 0.95)
    return {
        "sGRV_m3_final": grv,
        "sGRV_oil_m3": grv * f_oil,
        "sGRV_gas_m3": grv * (1.0 - f_oil),
        "sNtG": np.clip(rng.normal(0.72, 0.09, n), 0.05, 1.0),
        "sp": np.clip(rng.normal(0.22, 0.03, n), 0.01, 0.4),
        "sBg": np.clip(rng.normal(0.0042, 0.0005, n), 1e-5, None),
        "sInvBo": np.clip(rng.normal(0.78, 0.05, n), 0.1, 1.5),
        "sRF_oil": np.clip(rng.normal(0.35, 0.06, n), 0.0, 1.0),
        "sRF_gas": np.clip(rng.normal(0.70, 0.07, n), 0.0, 1.0),
        "sRF_assoc_gas": np.clip(rng.normal(0.80, 0.05, n), 0.0, 1.0),
        "sGOR": np.clip(rng.normal(650.0, 120.0, n), 0.0, None),
        "sCY": np.clip(rng.normal(35.0, 8.0, n), 0.0, None),
        "sRF_cond": np.clip(rng.normal(0.55, 0.08, n), 0.0, 1.0),
        "Shc_oil": np.clip(rng.normal(0.78, 0.05, n), 0.0, 1.0),
        "Shc_gas": np.clip(rng.normal(0.85, 0.04, n), 0.0, 1.0),
        "sOil_Column_Height": np.clip(rng.normal(85.0, 22.0, n), 0.0, None),
        "sGas_Column_Height": np.clip(rng.normal(60.0, 18.0, n), 0.0, None),
    }


def results_app(**overrides):
    """An AppTest on the Results page, seeded with a complete simulation."""
    at = AppTest.from_string(SCRIPT, default_timeout=900)
    for key, value in make_state().items():
        at.session_state[key] = value
    settings = {
        "fluid_type": "Oil + Gas",
        "grv_option": "Direct GRV",
        "unit_system": "oilfield",
        "percentile_exceedance": True,
        "gas_scf_per_boe": 6000.0,
        "num_sims": N,
        "sat_mode": "Global",
    }
    settings.update(overrides)
    for key, value in settings.items():
        at.session_state[key] = value
    return at


def run(at):
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


# ---------------------------------------------------------------------------------
# Reading the rendered page
# ---------------------------------------------------------------------------------


def chart_specs(at) -> list:
    return [el.proto.spec for el in at.get("plotly_chart")]


def percentile_charts(at) -> dict:
    """The seven histogram/CDF figures, keyed by title.

    They are picked out by the secondary axis title, which names the convention.
    The cross-plot's own figure is deliberately not cached, so matching on all
    twelve charts would let a stale cached figure hide behind that one always
    changing.
    """
    specs = [
        s
        for s in chart_specs(at)
        if "Probability of Exceedance" in s or "Cumulative Probability" in s
    ]
    assert len(specs) == 7, f"expected 7 histogram/CDF figures, got {len(specs)}"
    found = {}
    for title in PERCENTILE_CHARTS:
        matches = [s for s in specs if title in s]
        assert len(matches) == 1, f"expected one {title!r} figure, got {len(matches)}"
        found[title] = matches[0]
    return found


def chart_matching(at, needle: str) -> str:
    matches = [s for s in chart_specs(at) if needle in s]
    assert len(matches) == 1, f"expected one chart matching {needle!r}, got {len(matches)}"
    return matches[0]


def tables(at) -> list:
    """Every rendered table, by value.

    By value and not by protobuf because two of them go through a pandas Styler,
    which mints a fresh uuid for its CSS selectors on every render - so their
    bytes differ run to run even when every number in them is identical. That
    predates this cache and is nothing to do with it.
    """
    return [el.value.to_csv() for el in at.get("dataframe")]


def percentile_tables(at) -> list:
    """The summary tables, which carry the P1..P99 columns."""
    found = [t for t in tables(at) if "P90," in t or t.rstrip().endswith("P90")]
    assert found, "no summary table rendered"
    return found


def metrics(at) -> list:
    """The cross-plot's Pearson/Spearman/R2/slope tiles."""
    found = [el.proto.SerializeToString() for el in at.get("metric")]
    assert found, "the cross-plot statistics did not render"
    return found


def signature(at) -> tuple:
    """Everything the page renders. Counts asserted so this cannot degrade."""
    charts = tuple(chart_specs(at))
    frames = tuple(tables(at))
    text = tuple(el.value for el in at.markdown) + tuple(el.value for el in at.caption)
    assert len(charts) == 12, f"expected 12 charts, got {len(charts)}"
    assert len(frames) == 12, f"expected 12 tables, got {len(frames)}"
    return charts, frames, tuple(metrics(at)), text


def cache_stats(at) -> dict:
    return dict(at.session_state["_render_cache_stats"])


# ---------------------------------------------------------------------------------
# The cache does not change the page
# ---------------------------------------------------------------------------------


def test_rerun_is_identical_and_hits_the_cache():
    at = run(results_app())
    first = signature(at)
    cold = cache_stats(at)
    assert cold["misses"] > 0, "nothing was built on the first render"

    run(at)  # a rerun of the same session: what a sidebar click causes
    assert signature(at) == first

    warm = cache_stats(at)
    assert warm["hits"] > cold["hits"], "the rerun did not use the cache"
    assert warm["misses"] == cold["misses"], "the rerun rebuilt something"


def test_cached_page_matches_the_page_built_without_the_cache(monkeypatch):
    """Bypassing the cache must give the same page, element for element."""
    passthrough = lambda key, build: build()
    monkeypatch.setattr(results_mod, "memo", passthrough)
    monkeypatch.setattr(explorer_mod, "memo", passthrough)
    uncached = signature(run(results_app()))

    monkeypatch.undo()
    cached_at = run(results_app())
    assert cache_stats(cached_at)["misses"] > 0
    first = signature(cached_at)
    second = signature(run(cached_at))

    assert first == uncached, "a cold cached render differs from an uncached one"
    assert second == uncached, "a cache hit differs from an uncached render"


def test_every_figure_is_distinct():
    """Two different charts must not come back sharing one cached figure."""
    at = run(results_app())
    specs = chart_specs(at)
    assert len(set(specs)) == len(specs), "two charts rendered the same figure"


# ---------------------------------------------------------------------------------
# Invalidation: every input redraws what it feeds
# ---------------------------------------------------------------------------------


def test_each_sampled_input_redraws_the_figures_it_feeds():
    """Named figure by named figure, in one session of successive reruns.

    One session on purpose: a stale cache hit can only happen on a rerun, and
    reusing the session is also what makes this affordable to run.
    """
    at = run(results_app())
    before = percentile_charts(at)

    for keys, factor, targets in INPUT_TARGETS:
        for key in keys:
            at.session_state[key] = np.asarray(at.session_state[key], dtype=float) * factor
        run(at)
        after = percentile_charts(at)

        for title in targets:
            assert after[title] != before[title], (
                f"the {title!r} figure did not redraw after {keys} changed"
            )
        before = after


def test_each_sampled_input_redraws_the_summary_tables():
    at = run(results_app())
    before = percentile_tables(at)

    for keys, factor, _targets in INPUT_TARGETS:
        for key in keys:
            at.session_state[key] = np.asarray(at.session_state[key], dtype=float) * factor
        run(at)
        after = percentile_tables(at)
        assert after != before, f"no summary table changed after {keys} changed"
        before = after


def test_one_swapped_trial_is_enough_to_redraw():
    """The key is the array bytes, not a summary of them.

    This is why compute_input_hash is not used as the cache key: it records only
    the mean, std, min, max and length of each array. Swapping two trials leaves
    all of those identical while the sample is different.
    """
    at = run(results_app())
    before = percentile_charts(at)

    ntg = np.asarray(at.session_state["sNtG"], dtype=float).copy()
    ntg[0], ntg[1] = ntg[1], ntg[0]
    assert ntg[0] != ntg[1]
    at.session_state["sNtG"] = ntg
    run(at)

    after = percentile_charts(at)
    assert after[STOIIP] != before[STOIIP]
    assert after[THR] != before[THR]


def test_gas_to_boe_factor_redraws_the_boe_figures():
    at = run(results_app())
    before = percentile_charts(at)
    violin_before = chart_matching(at, "Hydrocarbon Volume Distribution Analysis")

    at.session_state["gas_scf_per_boe"] = 5500.0
    run(at)

    # THR and the BOE violin both divide gas by this factor; the oil histogram
    # does not involve it at all.
    assert percentile_charts(at)[THR] != before[THR]
    assert chart_matching(at, "Hydrocarbon Volume Distribution Analysis") != violin_before
    assert percentile_charts(at)[OIL] == before[OIL]


def test_unit_system_redraws_everything():
    at = run(results_app())
    before = percentile_charts(at)
    at.session_state["unit_system"] = "si"
    run(at)
    after = percentile_charts(at)
    for title in PERCENTILE_CHARTS:
        assert after[title] != before[title], f"{title!r} kept its oilfield figure"


@pytest.mark.parametrize("key", ["sOil_Column_Height", "sGas_Column_Height"])
def test_changing_a_column_height_redraws_the_overlay(key):
    at = run(results_app())
    before = chart_matching(at, "Column Height")

    at.session_state[key] = np.asarray(at.session_state[key], dtype=float) * 1.2
    run(at)

    assert chart_matching(at, "Column Height") != before


def test_changing_a_volume_redraws_the_boe_violin():
    at = run(results_app())
    needle = "Hydrocarbon Volume Distribution Analysis"
    before = chart_matching(at, needle)

    at.session_state["sRF_gas"] = np.asarray(at.session_state["sRF_gas"], dtype=float) * 0.5
    run(at)

    assert chart_matching(at, needle) != before


# ---------------------------------------------------------------------------------
# Invalidation: the percentile convention
# ---------------------------------------------------------------------------------


def test_convention_flip_redraws_every_percentile_figure_in_one_session():
    """The figures, not just the tables, must follow the convention.

    This is the case a cache gets wrong: the arrays are byte-identical in both
    conventions, so a key that leaves the convention out serves the old figure
    while the tables beside it change.
    """
    at = run(results_app(percentile_exceedance=True))
    before = percentile_charts(at)
    assert all("Probability of Exceedance" in s for s in before.values())

    at.session_state["percentile_exceedance"] = False
    run(at)
    after = percentile_charts(at)

    assert not [s for s in after.values() if "Probability of Exceedance" in s], (
        "a figure still shows the exceedance CDF after the convention changed"
    )
    assert all("Cumulative Probability" in s for s in after.values())
    for title in PERCENTILE_CHARTS:
        assert after[title] != before[title], f"{title!r} was not redrawn"

    at.session_state["percentile_exceedance"] = True
    run(at)
    assert percentile_charts(at) == before, "the first convention's figures did not return"


def test_convention_flip_swaps_p10_and_p90_in_the_tables_in_one_session():
    """P10 is the HIGH case under exceedance, the low one otherwise."""

    def p10_p90_pairs(at_):
        pairs = []
        for el in at_.get("dataframe"):
            frame = el.value
            columns = list(getattr(frame, "columns", []))
            if "P10" not in columns or "P90" not in columns:
                continue
            for p10, p90 in zip(frame["P10"], frame["P90"]):
                try:
                    p10, p90 = float(p10), float(p90)
                except (TypeError, ValueError):
                    continue
                # The cross-plot's binned table lists empty intervals with no
                # statistics, so NaN rows are expected and carry no ordering.
                if not (np.isfinite(p10) and np.isfinite(p90)):
                    continue
                if p10 != p90:
                    pairs.append((p10, p90))
        assert pairs, "no table with differing P10/P90 was rendered"
        return pairs

    at = run(results_app(percentile_exceedance=True))
    exceedance = p10_p90_pairs(at)
    assert all(p10 > p90 for p10, p90 in exceedance), exceedance

    at.session_state["percentile_exceedance"] = False
    run(at)
    non_exceedance = p10_p90_pairs(at)
    assert all(p10 < p90 for p10, p90 in non_exceedance), non_exceedance


def test_convention_flip_moves_the_thr_percentile_lines():
    """The P10/P90 annotations on the THR histogram must change value."""

    def thr_annotations(at):
        spec = percentile_charts(at)[THR]
        return [
            fragment.split('"')[0]
            for label in ("P10", "P90")
            for fragment in spec.split(f'"{label}: ')[1:]
        ]

    at = run(results_app(percentile_exceedance=True))
    exceedance = thr_annotations(at)
    assert exceedance, "the THR figure carries no percentile annotations"

    at.session_state["percentile_exceedance"] = False
    run(at)
    assert thr_annotations(at) != exceedance


def test_convention_change_alongside_the_results_cache_being_cleared():
    """_clear_results_on_convention_change drops results_cache, not this cache.

    The render cache needs no clearing - the convention is in every key - but
    the two must not interfere, so exercise them together.
    """
    at = run(results_app(percentile_exceedance=True))
    before = percentile_charts(at)

    at.session_state["results_cache"] = {"stale": np.zeros(3)}
    at.session_state["percentile_exceedance"] = False
    run(at)
    assert percentile_charts(at) != before

    at.session_state["percentile_exceedance"] = True
    run(at)
    assert percentile_charts(at) == before


# ---------------------------------------------------------------------------------
# Invalidation: the cross-plot's cached statistics
# ---------------------------------------------------------------------------------


# The correlation heatmap is the only figure with a [-1, 1] colour range;
# '"type":"heatmap"' is no use as a needle because every figure's Plotly
# template carries heatmap colorscale defaults. The driver ranking is the only
# one titled "Correlation with ...".
HEATMAP = '"zmin":-1'
DRIVERS = "Correlation with"


def trace_values(at, needle: str, field: str):
    """The numeric payload of a chart's first trace.

    Comparing whole chart specs is not enough for these two: the method name
    ("Spearman"/"Pearson") appears in their titles and hover templates, so the
    spec changes when the radio is switched even if the correlations behind it
    came back stale from the cache. Plotly serialises the numbers as a base64
    buffer, which compares exactly.
    """
    import json

    trace = json.loads(chart_matching(at, needle))["data"][0]
    return trace[field]


def test_changing_an_input_redraws_the_crossplot_statistics():
    """pair_statistics, the correlation matrix and the driver ranking are cached.

    The input is replaced with an independent sample rather than rescaled.
    Pearson and Spearman are invariant under a positive affine transform, so
    multiplying an array by a constant leaves every correlation on the page
    genuinely unchanged - a rescale would make this test prove nothing.
    """
    at = run(results_app())
    before_metrics = metrics(at)
    before_drivers = chart_matching(at, DRIVERS)
    before_heatmap = chart_matching(at, HEATMAP)

    before_driver_values = trace_values(at, DRIVERS, "x")
    before_matrix_values = trace_values(at, HEATMAP, "z")

    rng = np.random.default_rng(99)
    at.session_state["sNtG"] = np.clip(rng.normal(0.72, 0.09, N), 0.05, 1.0)
    run(at)

    assert metrics(at) != before_metrics, "the Pearson/Spearman tiles went stale"
    assert chart_matching(at, DRIVERS) != before_drivers, "the driver ranking went stale"
    assert chart_matching(at, HEATMAP) != before_heatmap, "the correlation matrix went stale"
    assert trace_values(at, DRIVERS, "x") != before_driver_values
    assert trace_values(at, HEATMAP, "z") != before_matrix_values


def test_switching_correlation_method_redraws_the_matrix_and_drivers():
    """The method radio lives inside the fragment, so this is a fragment rerun.

    Spearman and Pearson read the same columns, so only the method being in the
    cache key keeps the two apart.
    """
    at = run(results_app())
    spearman_matrix = trace_values(at, HEATMAP, "z")
    spearman_drivers = trace_values(at, DRIVERS, "x")

    at.session_state["xplot_matrix_method"] = "Pearson"
    run(at)

    assert trace_values(at, HEATMAP, "z") != spearman_matrix, (
        "the correlation matrix kept its Spearman values under Pearson"
    )
    assert trace_values(at, DRIVERS, "x") != spearman_drivers, (
        "the driver ranking kept its Spearman values under Pearson"
    )


def test_convention_flip_changes_the_binned_table():
    """bin_statistics computes its percentiles on the active convention."""
    at = run(results_app(percentile_exceedance=True))
    before = tables(at)

    at.session_state["percentile_exceedance"] = False
    run(at)

    assert tables(at) != before


# ---------------------------------------------------------------------------------
# The mutation hazard
# ---------------------------------------------------------------------------------


def test_thr_figure_does_not_accumulate_percentile_lines():
    """A cached figure must not be mutated after it is handed out.

    The THR histogram gets three P10/P50/P90 lines. Those used to be added by
    the caller to the figure make_hist_cdf_figure returned; with the figure
    cached, that would add three more on every rerun.
    """
    at = results_app()
    counts = []
    for _ in range(3):
        run(at)
        spec = percentile_charts(at)[THR]
        counts.append(sum(spec.count(f'"{label}: ') for label in ("P10", "P50", "P90")))

    assert len(set(counts)) == 1, f"annotation count drifted across reruns: {counts}"
    assert counts[0] >= 3, counts


# ---------------------------------------------------------------------------------
# fingerprint / memo in isolation
# ---------------------------------------------------------------------------------


def test_fingerprint_distinguishes_a_single_element():
    a = np.arange(1000, dtype=float)
    b = a.copy()
    b[500] = 500.0000001
    assert render_cache.fingerprint(a) != render_cache.fingerprint(b)


def test_fingerprint_is_stable_for_equal_arrays():
    a = np.arange(100, dtype=float)
    assert render_cache.fingerprint(a) == render_cache.fingerprint(a.copy())


def test_fingerprint_ignores_memory_layout_but_not_content():
    base = np.arange(40, dtype=float).reshape(4, 10)
    view = base[:, ::2]  # non-contiguous
    assert render_cache.fingerprint(view) == render_cache.fingerprint(
        np.ascontiguousarray(view)
    )
    assert render_cache.fingerprint(view) != render_cache.fingerprint(base)


def test_fingerprint_distinguishes_dtype_and_shape():
    a = np.arange(6, dtype=np.float64)
    assert render_cache.fingerprint(a) != render_cache.fingerprint(a.astype(np.float32))
    assert render_cache.fingerprint(a) != render_cache.fingerprint(a.reshape(2, 3))


def test_fingerprint_distinguishes_bool_from_int():
    # Several keys carry a convention flag next to a count.
    assert render_cache.fingerprint(True) != render_cache.fingerprint(1)
    assert render_cache.fingerprint(False) != render_cache.fingerprint(0)


def test_fingerprint_does_not_run_parts_together():
    assert render_cache.fingerprint("ab", "c") != render_cache.fingerprint("a", "bc")
    assert render_cache.fingerprint(["a"], ["b"]) != render_cache.fingerprint(["a", "b"])


def test_fingerprint_distinguishes_none_from_its_text():
    assert render_cache.fingerprint(None) != render_cache.fingerprint("None")


def test_fingerprint_handles_nan_and_sign_of_zero():
    nan_a = np.array([1.0, np.nan, 3.0])
    assert render_cache.fingerprint(nan_a) == render_cache.fingerprint(nan_a.copy())
    assert render_cache.fingerprint(np.array([0.0])) != render_cache.fingerprint(
        np.array([-0.0])
    )


def test_fingerprint_reads_dataframes_by_value():
    import pandas as pd

    a = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    b = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.5]})
    assert render_cache.fingerprint(a) == render_cache.fingerprint(a.copy())
    assert render_cache.fingerprint(a) != render_cache.fingerprint(b)
    assert render_cache.fingerprint(a) != render_cache.fingerprint(
        a.rename(columns={"y": "z"})
    )


def test_memo_builds_once_per_key():
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return calls["n"]

    render_cache.clear()
    key = render_cache.fingerprint("memo-test", 1)
    other = render_cache.fingerprint("memo-test", 2)

    assert render_cache.memo(key, build) == 1
    assert render_cache.memo(key, build) == 1, "a second call rebuilt the value"
    assert render_cache.memo(other, build) == 2, "a different key reused a value"
    assert render_cache.memo(key, build) == 1, "the first key lost its value"
    assert calls["n"] == 2

    stats = render_cache.cache_stats()
    assert stats["hits"] == 2 and stats["misses"] == 2
    assert stats["size"] == 2

    render_cache.clear()
    assert render_cache.cache_stats()["size"] == 0
    assert render_cache.memo(key, build) == 3, "clear() did not drop the entry"
    render_cache.clear()


def test_memo_evicts_the_oldest_entries_first():
    """The store is bounded, so a long session cannot grow without limit."""
    render_cache.clear()
    limit = render_cache._MAX_ENTRIES
    keys = [render_cache.fingerprint("evict", i) for i in range(limit + 10)]
    for i, key in enumerate(keys):
        render_cache.memo(key, lambda i=i: i)

    assert render_cache.cache_stats()["size"] == limit
    assert render_cache.memo(keys[-1], lambda: "rebuilt") == len(keys) - 1
    assert render_cache.memo(keys[0], lambda: "rebuilt") == "rebuilt"
    render_cache.clear()
