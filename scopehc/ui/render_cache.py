"""Content-addressed memo cache for the figures and statistics on a page.

Why this exists
---------------
Streamlit re-runs the whole script on any widget interaction, so a sidebar
checkbox that changes nothing on the Results page still rebuilt every figure on
it. Measurement (see the PR that introduced this module) put the Monte Carlo
arithmetic at 0.2% of a 10,000-trial Results render and 1% at 50,000. The cost
is elsewhere: Plotly figure construction, about 20 ms per figure and almost
independent of trial count, and the SciPy rank correlations behind the
cross-plot. This cache skips that work when - and only when - every input to it
is byte-for-byte unchanged.

Why not @st.cache_data
----------------------
Two reasons, both measured.

* ``st.cache_data`` serialises what it stores. A finished histogram/CDF figure
  at 50,000 trials pickles to 1.6 MB and costs 10 ms to round-trip, against
  21 ms to build from scratch - half the saving spent on the cache itself.
  This module stores the live object in ``st.session_state``.
* ``st.cache_data`` is shared between browser sessions. Under ``streamlit run``,
  session state keeps one session's figures to itself, so even a key collision
  could not put one user's numbers on another user's screen. (Outside a server -
  a plain script, or the test suite - ``st.session_state`` falls back to a single
  process-global store. That is harmless: the keys are content-addressed, so a
  shared store still answers with the value that belongs to the key.)

Why not the existing compute_input_hash
---------------------------------------
``common.compute_input_hash`` hashes the dict from
``collect_all_inputs_from_session``, which records only the mean, std, min, max
and length of each sampled array. That is built to answer "has the user changed
an input since the last run?", and it is lossy by design: two different sample
sets with the same first four moments hash alike. It is therefore not safe as a
key for anything that has to match the numbers on screen. Keys here come from
the actual array bytes instead, which also means every upstream input - fluid
type, GRV option, saturation mode, the gas-to-BOE factor, the seed, the trial
count - is covered automatically, because any of them changing changes the
values being plotted.

The contract
------------
1. A key must name every value that can change the result. For a figure that
   means the plotted arrays, every label and unit string that appears in it, the
   colour inputs, and the percentile convention - which flips both the CDF
   direction and the P10/P90 lines without touching the data.
2. **A cached object must never be mutated by its caller.** Mutating one means
   every later render sees the mutation: a figure that collects a further set of
   P10/P50/P90 lines on every rerun, for instance. Anything of the
   build-then-adjust shape has to do the adjusting inside the builder.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")

# Enough for every figure and table on the Results page a few times over, so
# flipping a setting back and forth still hits, while a session cannot grow
# without bound. The largest entry is a figure holding a few hundred kB of
# trial data.
_MAX_ENTRIES = 96

_STORE_KEY = "_render_cache"
_STATS_KEY = "_render_cache_stats"

_SEP = b"\x1f"  # between parts, so ("ab", "c") and ("a", "bc") differ
_MISS = object()


def _store() -> "OrderedDict[str, Any] | None":
    """The cache for this session, or None if session state cannot be reached.

    Under ``streamlit run`` this is per browser session. Without a server it is
    one store for the process, which is safe here because keys are
    content-addressed - see the module docstring. A module-level dict is
    deliberately not used as a fallback, so that the server case stays
    per-session.
    """
    try:
        import streamlit as st

        state = st.session_state
        if _STORE_KEY not in state:
            state[_STORE_KEY] = OrderedDict()
        return state[_STORE_KEY]
    except Exception:
        return None


def _bump(field: str) -> None:
    try:
        import streamlit as st

        stats = st.session_state.setdefault(_STATS_KEY, {"hits": 0, "misses": 0})
        stats[field] = stats.get(field, 0) + 1
    except Exception:
        pass


def _feed(h: "hashlib._Hash", part: Any) -> None:
    """Fold one value into the digest, by content and not by identity."""
    if part is None:
        h.update(b"None")
        return
    if isinstance(part, np.ndarray):
        # dtype and shape go in as well, so arrays that share bytes but differ
        # in either are distinguishable. Object/string arrays have no stable
        # buffer, so fall back to their repr.
        h.update(f"ndarray:{part.dtype.str}:{part.shape}:".encode("utf-8"))
        if part.dtype.hasobject:
            h.update(repr(part.tolist()).encode("utf-8"))
        else:
            h.update(np.ascontiguousarray(part).tobytes())
        return
    if isinstance(part, pd.DataFrame):
        h.update(f"frame:{list(part.columns)}:".encode("utf-8"))
        _feed(h, part.to_numpy())
        return
    if isinstance(part, pd.Series):
        h.update(f"series:{part.name}:".encode("utf-8"))
        _feed(h, part.to_numpy())
        return
    if isinstance(part, (list, tuple)):
        h.update(f"seq:{len(part)}:".encode("utf-8"))
        for item in part:
            _feed(h, item)
            h.update(_SEP)
        return
    if isinstance(part, dict):
        h.update(b"map:")
        for key in sorted(part, key=repr):
            _feed(h, key)
            h.update(b"=")
            _feed(h, part[key])
            h.update(_SEP)
        return
    # bool before int: True and 1 must not collide, since several of these keys
    # carry a convention flag next to a count.
    if isinstance(part, bool):
        h.update(b"bool:1" if part else b"bool:0")
        return
    if isinstance(part, (int, np.integer)):
        h.update(f"int:{int(part)}".encode("utf-8"))
        return
    if isinstance(part, (float, np.floating)):
        # repr round-trips a float exactly, unlike a formatted string.
        h.update(f"float:{float(part)!r}".encode("utf-8"))
        return
    if isinstance(part, str):
        h.update(b"str:")
        h.update(part.encode("utf-8"))
        return
    if isinstance(part, bytes):
        h.update(b"bytes:")
        h.update(part)
        return
    # Anything else: repr is the honest option. A type whose repr is its memory
    # address would key by identity, so do not pass such a thing as a key part.
    h.update(f"repr:{part!r}".encode("utf-8"))


def fingerprint(*parts: Any) -> str:
    """A digest of the given values, by content.

    Hashing the full bytes of a 50,000-trial array costs about 0.6 ms, against
    the 20 ms the figure built from it costs - see the module docstring.
    """
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        _feed(h, part)
        h.update(_SEP)
    return h.hexdigest()


def memo(key: str, build: Callable[[], T]) -> T:
    """Return the value for ``key``, calling ``build`` only on a miss.

    The returned object belongs to the cache: see the contract in the module
    docstring and do not mutate it.
    """
    store = _store()
    if store is None:
        return build()

    cached = store.get(key, _MISS)
    if cached is not _MISS:
        store.move_to_end(key)
        _bump("hits")
        return cached  # type: ignore[return-value]

    value = build()
    store[key] = value
    while len(store) > _MAX_ENTRIES:
        store.popitem(last=False)
    _bump("misses")
    return value


def cached(*key_parts: Any):
    """``memo`` as a decorator-free helper: ``cached(a, b)(lambda: build())``."""
    key = fingerprint(*key_parts)

    def run(build: Callable[[], T]) -> T:
        return memo(key, build)

    return run


def cache_stats() -> dict:
    """Hits, misses and current size - for diagnostics and tests."""
    store = _store()
    try:
        import streamlit as st

        stats = dict(st.session_state.get(_STATS_KEY, {"hits": 0, "misses": 0}))
    except Exception:
        stats = {"hits": 0, "misses": 0}
    stats["size"] = 0 if store is None else len(store)
    return stats


def clear() -> None:
    """Drop everything cached for this session.

    Nothing in the app needs this for correctness - every key names its own
    inputs, so a changed input lands on a different key rather than
    invalidating an old one. It exists for tests and for a deliberate reset.
    """
    store = _store()
    if store is not None:
        store.clear()
    try:
        import streamlit as st

        st.session_state[_STATS_KEY] = {"hits": 0, "misses": 0}
    except Exception:
        pass
