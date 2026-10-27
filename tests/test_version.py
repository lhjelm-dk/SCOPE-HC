"""Tests for the deployed-version marker.

What this pins is mostly the failure behaviour. The marker reports on the app;
if it can raise, it takes down the page it is meant to describe, so every lookup
has to degrade to "unknown" instead.

The staleness check is the part that earns its place. The live app once served
page files from a current checkout while every module under scopehc/ was several
commits behind, and a reboot did not clear it. A commit SHA could not show that -
the SHA on disk was correct - so the marker compares the source files against
their state when the package was imported.
"""

import os
from pathlib import Path

import pytest

from scopehc import version


def test_the_commit_is_read_from_the_checkout():
    sha = version.commit()
    assert sha is not None, "these tests run inside the git checkout"
    assert len(sha) == 7
    assert all(c in "0123456789abcdef" for c in sha)


def test_the_full_sha_is_available_too():
    full = version.commit(short=False)
    assert len(full) == 40
    assert full.startswith(version.commit())


def test_the_branch_is_reported_when_head_points_at_one():
    # Detached HEAD gives None, which is why the marker tolerates it.
    name = version.branch()
    assert name is None or isinstance(name, str)


def test_nothing_has_changed_in_a_quiet_checkout():
    assert version.changed_since_import() == []


def test_the_marker_names_the_commit():
    marker = version.marker()
    assert version.commit() in marker


def test_a_file_touched_after_import_is_reported(tmp_path):
    """The live-app failure: source replaced underneath a running process."""
    target = Path(version.PACKAGE_ROOT) / "utils.py"
    before = os.stat(target)
    try:
        os.utime(target, (before.st_atime, before.st_mtime + 120))
        changed = version.changed_since_import()
        assert "utils.py" in changed
        marker = version.marker()
        assert "changed since start" in marker
        assert "1 file changed" in marker
    finally:
        os.utime(target, (before.st_atime, before.st_mtime))

    assert version.changed_since_import() == []


def test_several_changed_files_are_counted_in_the_plural():
    targets = [Path(version.PACKAGE_ROOT) / name for name in ("utils.py", "geom.py")]
    saved = [(t, os.stat(t)) for t in targets]
    try:
        for target, st in saved:
            os.utime(target, (st.st_atime, st.st_mtime + 120))
        assert "2 files changed" in version.marker()
    finally:
        for target, st in saved:
            os.utime(target, (st.st_atime, st.st_mtime))


def test_a_missing_git_directory_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(version, "REPO_ROOT", tmp_path)
    assert version.commit() is None
    assert version.branch() is None
    assert "version unknown" in version.marker()


def test_a_detached_head_reports_the_sha_and_no_branch(monkeypatch, tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("a" * 40 + "\n", encoding="utf-8")
    monkeypatch.setattr(version, "REPO_ROOT", tmp_path)

    assert version.commit() == "aaaaaaa"
    assert version.branch() is None


def test_a_packed_ref_is_found(monkeypatch, tmp_path):
    """Once git packs refs the loose file is gone; the SHA is still readable."""
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n"
        + "b" * 40 + " refs/heads/main\n"
        + "c" * 40 + " refs/tags/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(version, "REPO_ROOT", tmp_path)

    assert version.commit() == "bbbbbbb"


def test_an_unreadable_source_tree_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(version, "PACKAGE_ROOT", tmp_path / "gone")
    assert version.changed_since_import() == []
    assert isinstance(version.marker(), str)
