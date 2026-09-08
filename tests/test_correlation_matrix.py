"""Regression tests for the nearest-correlation-matrix projection.

The bug these pin: the previous implementation did not converge and could return an indefinite
matrix, which validate_dependency_matrix then reported as valid. An inconsistent elicitation was
therefore accepted at the input screen and raised LinAlgError later, from inside
correlated_samples. On the matrix below its output still had a smallest eigenvalue of -0.267.

Causes were (a) the Dykstra correction computed after the unit-diagonal fill, so it absorbed both
projections, and (b) a final clip of the whole matrix to +/-0.999, which could reintroduce
indefiniteness into a matrix that had just been repaired.
"""

import numpy as np
import pytest

from scopehc.sampling import (
    _nearest_correlation_matrix,
    correlated_samples,
    fix_correlation_matrix,
    rng_from_seed,
    validate_dependency_matrix,
)

# A-B 0.9, B-C 0.9, A-C -0.9. No set of random variables has this matrix: if A and B move together
# and B and C move together, A and C cannot move oppositely. Smallest eigenvalue -0.8.
INCONSISTENT = np.array([
    [1.0, 0.9, -0.9],
    [0.9, 1.0, 0.9],
    [-0.9, 0.9, 1.0],
])

VALID = np.array([
    [1.0, 0.3, 0.1],
    [0.3, 1.0, 0.2],
    [0.1, 0.2, 1.0],
])


def _min_eig(m):
    return float(np.linalg.eigvalsh(m).min())


def test_the_inconsistent_matrix_is_repaired_not_merely_adjusted():
    assert _min_eig(INCONSISTENT) < -0.5          # the input really is impossible
    assert _min_eig(_nearest_correlation_matrix(INCONSISTENT)) >= 0.0


@pytest.mark.parametrize("A", [INCONSISTENT, VALID, np.eye(4),
                               np.full((4, 4), 0.99) + np.eye(4) * 0.01,
                               np.array([[1.0, -0.999], [-0.999, 1.0]])])
def test_the_result_always_has_a_cholesky_factor(A):
    """The property that matters: everything downstream factors the result."""
    np.linalg.cholesky(_nearest_correlation_matrix(A))


@pytest.mark.parametrize("A", [INCONSISTENT, VALID, np.eye(4)])
def test_the_result_is_a_correlation_matrix(A):
    X = _nearest_correlation_matrix(A)
    assert np.allclose(np.diag(X), 1.0)
    assert np.allclose(X, X.T)
    off = X[~np.eye(X.shape[0], dtype=bool)]
    assert np.all(np.abs(off) <= 0.999 + 1e-12)


@pytest.mark.parametrize("A", [VALID, np.eye(3),
                               np.array([[1.0, 0.7, 0.6, 0.5], [0.7, 1.0, 0.4, 0.3],
                                         [0.6, 0.4, 1.0, 0.2], [0.5, 0.3, 0.2, 1.0]])])
def test_an_already_valid_matrix_passes_through_untouched(A):
    """A projection that distorted valid input would silently change every correlated run."""
    assert np.allclose(_nearest_correlation_matrix(A), A, atol=1e-12)


def test_the_repair_is_a_compromise_rather_than_a_surrender():
    """The three requested pairs cannot all hold, so the projection splits the difference. What it
    must not do is discard the *signs* the assessor stated."""
    X = _nearest_correlation_matrix(INCONSISTENT)
    assert X[0, 1] > 0.3 and X[1, 2] > 0.3 and X[0, 2] < -0.3


def test_validation_and_sampling_no_longer_disagree():
    """The actual defect, end to end: validation used to pass and sampling then raised."""
    ok, message = validate_dependency_matrix(INCONSISTENT, ["a", "b", "c"])
    assert ok, message

    names = ["a", "b", "c"]
    cfg = {n: {"distribution": "normal", "mean": 0.0, "std": 1.0} for n in names}
    samples = correlated_samples(rng_from_seed(0), cfg, INCONSISTENT, names, 20_000)
    assert set(samples) == set(names)
    assert all(np.isfinite(samples[n]).all() for n in names)


def test_fix_correlation_matrix_does_not_reintroduce_indefiniteness():
    """It used to clip after the projection, which is what could undo the repair."""
    assert _min_eig(fix_correlation_matrix(INCONSISTENT)) >= 0.0


def test_the_projection_is_idempotent():
    """Projecting a projected matrix must be a no-op, or the iteration has not converged."""
    once = _nearest_correlation_matrix(INCONSISTENT)
    assert np.allclose(_nearest_correlation_matrix(once), once, atol=1e-10)
