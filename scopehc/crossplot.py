"""
Cross-plot statistics: binning, correlation and fits for the Results-page explorer.

Deliberately free of any Streamlit import so the numbers can be unit-tested directly
(see tests/test_crossplot.py). The percentile convention is passed in as an argument
rather than read from session state, for the same reason - the UI layer is the only
place that knows what the user picked in the sidebar.

Convention note: with probability-of-exceedance (the app default) P10 is the HIGH
case and P90 the low one, matching scopehc.utils.summarize_array.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# A per-bin table longer than this is unreadable and slow to render; the caller is
# expected to catch the ValueError and tell the user to widen the bins.
MAX_BINS = 500

BIN_COLUMNS = (
    "bin_min",
    "bin_max",
    "bin_mid",
    "count",
    "mean",
    "std",
    "P10",
    "P50",
    "P90",
    "mean_per_unit_x",
)


def finite_pairs(x, y) -> Tuple[np.ndarray, np.ndarray]:
    """Return x, y as float arrays keeping only entries finite in BOTH."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"x and y must be the same length (got {x.size} and {y.size})")
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def percentiles_by_convention(
    values: np.ndarray, use_exceedance: bool = True
) -> Dict[str, float]:
    """P10/P50/P90 of `values` under the app's percentile convention.

    use_exceedance=True  -> P10 is the 90th percentile (high case), P90 the 10th.
    use_exceedance=False -> P10 is the 10th percentile, P90 the 90th.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"P10": np.nan, "P50": np.nan, "P90": np.nan}

    low, mid, high = np.percentile(values, [10, 50, 90])
    if use_exceedance:
        return {"P10": float(high), "P50": float(mid), "P90": float(low)}
    return {"P10": float(low), "P50": float(mid), "P90": float(high)}


def suggest_bin_width(x, target_bins: int = 20, depth_like: bool = False) -> float:
    """A round bin width: 10 m for depths, else 1/2/5 x 10^k closest to range/target."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 10.0 if depth_like else 1.0

    span = float(np.max(x) - np.min(x))
    if span <= 0:
        return 10.0 if depth_like else 1.0

    if depth_like and span > 50.0:
        return 10.0

    raw = span / max(int(target_bins), 1)
    exponent = np.floor(np.log10(raw))
    base = 10.0**exponent
    for step in (1.0, 2.0, 5.0, 10.0):
        if raw <= step * base:
            return float(step * base)
    return float(10.0 * base)


def bin_edges_for(
    x: np.ndarray, bin_width: float, origin: Optional[float] = None
) -> np.ndarray:
    """Edges anchored on a round multiple of the width, so a 10 m bin starts at 2200."""
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be a positive, finite number")

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.asarray([], dtype=float)

    lo, hi = float(np.min(x)), float(np.max(x))
    start = (
        float(origin)
        if origin is not None
        else float(np.floor(lo / bin_width) * bin_width)
    )
    n_bins = int(np.ceil((hi - start) / bin_width))
    n_bins = max(n_bins, 1)
    if n_bins > MAX_BINS:
        raise ValueError(
            f"bin width {bin_width:g} gives {n_bins} bins over a range of "
            f"{hi - lo:g} (limit {MAX_BINS}) - use a wider bin"
        )
    return start + bin_width * np.arange(n_bins + 1, dtype=float)


def bin_statistics(
    x,
    y,
    bin_width: float,
    use_exceedance: bool = True,
    origin: Optional[float] = None,
) -> pd.DataFrame:
    """Statistics of y within fixed-width intervals of x.

    One row per interval, empty intervals included with count 0 so the table has no
    gaps. `mean_per_unit_x` is mean / bin_width - the literal "mean STOIIP per metre
    of depth" reading.
    """
    x, y = finite_pairs(x, y)
    if x.size == 0:
        return pd.DataFrame(columns=list(BIN_COLUMNS))

    edges = bin_edges_for(x, bin_width, origin=origin)

    # An explicit origin can sit above the data; anything outside the edges is not ours
    # to lump into the end bins.
    inside = (x >= edges[0]) & (x <= edges[-1])
    x, y = x[inside], y[inside]
    if x.size == 0:
        return pd.DataFrame(columns=list(BIN_COLUMNS))

    n_bins = len(edges) - 1
    # Bins are [min, max) except the last, which is closed on the right so the single
    # maximum value lands in it rather than in a bin of its own.
    idx = (
        np.digitize(x, edges[1:-1], right=False)
        if n_bins > 1
        else np.zeros_like(x, dtype=int)
    )
    idx = np.clip(idx, 0, n_bins - 1)

    rows: List[Dict[str, float]] = []
    for b in range(n_bins):
        in_bin = y[idx == b]
        count = int(in_bin.size)
        row = {
            "bin_min": float(edges[b]),
            "bin_max": float(edges[b + 1]),
            "bin_mid": float((edges[b] + edges[b + 1]) / 2.0),
            "count": count,
        }
        if count == 0:
            row.update(
                {
                    "mean": np.nan,
                    "std": np.nan,
                    "P10": np.nan,
                    "P50": np.nan,
                    "P90": np.nan,
                    "mean_per_unit_x": np.nan,
                }
            )
        else:
            mean_v = float(np.mean(in_bin))
            row.update(
                {
                    "mean": mean_v,
                    "std": float(np.std(in_bin, ddof=1)) if count > 1 else 0.0,
                    "mean_per_unit_x": mean_v / float(bin_width),
                }
            )
            row.update(percentiles_by_convention(in_bin, use_exceedance))
        rows.append(row)

    return pd.DataFrame(rows, columns=list(BIN_COLUMNS))


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> Tuple[float, float]:
    """Pearson or Spearman with p-value, NaN instead of an exception on degenerate input."""
    if x.size < 3:
        return np.nan, np.nan
    if method == "pearson":
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan, np.nan
        from scipy.stats import pearsonr

        stat, p = pearsonr(x, y)
    else:
        # Spearman is rank-based, so it is the ranks that must vary.
        from scipy.stats import rankdata, spearmanr

        if np.std(rankdata(x)) == 0 or np.std(rankdata(y)) == 0:
            return np.nan, np.nan
        stat, p = spearmanr(x, y)
    return float(stat), float(p)


def pair_statistics(x, y) -> Dict[str, float]:
    """Correlation, R^2 and least-squares slope for one pair, plus the log-log variant."""
    x, y = finite_pairs(x, y)
    out: Dict[str, float] = {
        "n": int(x.size),
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "r_squared": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "log_pearson_r": np.nan,
        "log_r_squared": np.nan,
        "log_slope": np.nan,
    }
    if x.size < 3:
        return out

    out["pearson_r"], out["pearson_p"] = _safe_corr(x, y, "pearson")
    out["spearman_rho"], out["spearman_p"] = _safe_corr(x, y, "spearman")

    if np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        out["slope"] = float(slope)
        out["intercept"] = float(intercept)
    if np.isfinite(out["pearson_r"]):
        out["r_squared"] = float(out["pearson_r"] ** 2)

    # A power-law fit is only meaningful where both axes are strictly positive.
    if np.all(x > 0) and np.all(y > 0):
        lx, ly = np.log10(x), np.log10(y)
        r, _ = _safe_corr(lx, ly, "pearson")
        out["log_pearson_r"] = r
        if np.isfinite(r):
            out["log_r_squared"] = float(r**2)
        if np.std(lx) > 0:
            out["log_slope"] = float(np.polyfit(lx, ly, 1)[0])

    return out


def fit_line(
    x, y, log_x: bool = False, log_y: bool = False, n_points: int = 50
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Least-squares trend line, returned in the axes' own coordinates.

    With log_x/log_y the fit is done on log10 values and converted back, so the line
    is straight on a log axis. Returns None when there is nothing to fit.

    np.polyfit rather than a plotly trendline: px.scatter(trendline="ols") needs
    statsmodels, which is not a dependency of this app.
    """
    x, y = finite_pairs(x, y)
    if log_x:
        keep = x > 0
        x, y = x[keep], y[keep]
    if log_y:
        keep = y > 0
        x, y = x[keep], y[keep]
    if x.size < 3:
        return None

    xf = np.log10(x) if log_x else x
    yf = np.log10(y) if log_y else y
    if np.std(xf) == 0:
        return None

    slope, intercept = np.polyfit(xf, yf, 1)
    xs = np.linspace(float(np.min(xf)), float(np.max(xf)), max(int(n_points), 2))
    ys = slope * xs + intercept
    return (10.0**xs if log_x else xs), (10.0**ys if log_y else ys)


def correlation_matrix(
    df: pd.DataFrame, columns: Sequence[str], method: str = "spearman"
) -> pd.DataFrame:
    """Square correlation matrix over `columns`, skipping ones that do not vary."""
    usable = [
        c
        for c in columns
        if c in df.columns and df[c].notna().sum() >= 3 and df[c].std(skipna=True) > 0
    ]
    if len(usable) < 2:
        return pd.DataFrame()
    return df[usable].corr(method=method, numeric_only=True)


def rank_drivers(
    df: pd.DataFrame,
    target: str,
    columns: Iterable[str],
    method: str = "spearman",
) -> pd.DataFrame:
    """Correlation of each column against `target`, strongest absolute value first."""
    if target not in df.columns:
        return pd.DataFrame(columns=["variable", "correlation", "abs_correlation"])

    y = df[target].to_numpy(dtype=float)
    rows = []
    for col in columns:
        if col == target or col not in df.columns:
            continue
        x = df[col].to_numpy(dtype=float)
        try:
            xf, yf = finite_pairs(x, y)
        except ValueError:
            continue
        stat, _ = _safe_corr(xf, yf, "pearson" if method == "pearson" else "spearman")
        if np.isfinite(stat):
            rows.append(
                {
                    "variable": col,
                    "correlation": stat,
                    "abs_correlation": abs(stat),
                }
            )

    out = pd.DataFrame(rows, columns=["variable", "correlation", "abs_correlation"])
    if not out.empty:
        out = out.sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    return out
