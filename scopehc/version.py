"""
What is actually running: the repository commit, and whether the files on disk
have moved on since the modules were imported.

Why the second half exists
--------------------------
The live app once served a mixture: the page files under _pages_disabled/ were
current, because streamlit_app.py re-executes them from disk on every rerun,
while everything under scopehc/ was several commits behind. Rebooting did not
clear it. Working out which commit was actually running took reading three
separate markers off three pages and reasoning backwards from which of them had
landed.

A commit SHA alone would not have caught it, because the SHA is read from .git
at render time and the clone on disk was perfectly up to date - it was the
imported modules that were stale. So this records the modification times of the
package's own source files when it is first imported, and compares them against
the files on disk when asked. If a file has changed since import, the module in
memory is not the file on disk, and the marker says so.

Nothing here raises. A version marker that breaks the page it is reporting on
would be worse than no marker, so every lookup degrades to "unknown".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def commit(short: bool = True) -> Optional[str]:
    """The checked-out commit, read straight from .git - no subprocess.

    Streamlit Cloud deploys a git clone, so this works there; a zip deployment
    has no .git and gets None.
    """
    git_dir = REPO_ROOT / ".git"
    head = _read(git_dir / "HEAD")
    if head is None:
        return None

    if head.startswith("ref:"):
        ref = head.partition(" ")[2].strip()
        sha = _read(git_dir / ref)
        if sha is None:
            # A packed ref - the loose file is absent once git has packed it.
            packed = _read(git_dir / "packed-refs") or ""
            for line in packed.splitlines():
                if line.startswith(("#", "^")):
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2 and parts[1].strip() == ref:
                    sha = parts[0]
                    break
    else:
        sha = head  # detached HEAD

    if not sha:
        return None
    sha = sha.strip()
    return sha[:7] if short else sha


def branch() -> Optional[str]:
    head = _read(REPO_ROOT / ".git" / "HEAD")
    if head and head.startswith("ref: refs/heads/"):
        return head[len("ref: refs/heads/"):].strip()
    return None


def _source_files() -> List[Path]:
    try:
        return sorted(PACKAGE_ROOT.rglob("*.py"))
    except OSError:
        return []


def _mtimes() -> Dict[str, float]:
    out: Dict[str, float] = {}
    for path in _source_files():
        try:
            out[str(path)] = os.path.getmtime(path)
        except OSError:
            continue
    return out


# Captured once, when this package is first imported into the running process.
_AT_IMPORT: Dict[str, float] = _mtimes()


def changed_since_import() -> List[str]:
    """Source files that differ on disk from when they were imported.

    A non-empty list means the running process is not what the checkout says:
    the code was updated underneath it and nothing reloaded. That is the exact
    state the live app was in, and the one a commit SHA cannot show.
    """
    now = _mtimes()
    changed = [
        Path(name).name
        for name, mtime in now.items()
        if name in _AT_IMPORT and mtime != _AT_IMPORT[name]
    ]
    added = [Path(name).name for name in now if name not in _AT_IMPORT]
    return sorted(changed + added)


def marker() -> str:
    """One line for the sidebar."""
    sha = commit()
    name = branch()
    parts = []
    if sha:
        parts.append(f"{name} · {sha}" if name else sha)
    else:
        parts.append("version unknown")

    stale = changed_since_import()
    if stale:
        count = len(stale)
        noun = "file" if count == 1 else "files"
        parts.append(f"⚠ {count} {noun} changed since start - restart to pick up")
    return " | ".join(parts)
