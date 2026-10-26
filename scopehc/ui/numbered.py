"""
Rendering helpers for numbered headings, figures and tables.

The registry in scopehc/numbering.py owns the numbers and the hierarchy. These
three functions are the only place the app turns them into Streamlit output, so
a heading's depth on screen always matches its depth in the registry rather than
whatever number of '#' characters the original author happened to type.

That matters here: the GRV page renders its geometric-correction calculator as
'###' and its "GRV results by fluid type" as '###' too, although one is nested a
level deeper than the other. Taking the level from the registry normalises that
without anyone having to audit hash marks by hand.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from scopehc import numbering

# Markdown runs out of heading levels at 6; the registry is three deep today.
_MAX_HASHES = 6


def heading(key: str, override: Optional[str] = None) -> None:
    """Render '2.1.5 GRV results by fluid type' at the registry's level.

    `override` replaces the title where a heading is built at runtime from
    something the registry cannot know - the GRV page titles one section after
    the chosen method - without moving the number.
    """
    hashes = "#" * min(numbering.level(key), _MAX_HASHES)
    st.markdown(f"{hashes} {numbering.heading(key, override)}")


def figure_caption(item_key: str, override: Optional[str] = None) -> None:
    """Render 'Figure 2.1.5-1 — Oil GRV distribution' under a chart."""
    st.caption(numbering.caption(item_key, override))


def table_caption(item_key: str, override: Optional[str] = None) -> None:
    """Render 'Table 2.1.5-1 — Oil GRV statistics' under a table."""
    st.caption(numbering.caption(item_key, override))
