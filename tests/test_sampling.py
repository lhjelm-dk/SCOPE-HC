import numpy as np

from scopehc.sampling import (
    sample_beta_subjective,
    sample_truncated_normal,
    sample_truncated_lognormal,
    sample_burr,
    sample_johnson_su,
    rng_from_seed,
)


def test_subjective_beta_quantiles():
    rng = rng_from_seed(1234)
    samples = sample_beta_subjective(rng, 0.0, 1.0, 0.2, 0.5, 0.8, 100_000)
    q10, q50, q90 = np.percentile(samples, [10, 50, 90])

    assert np.isclose(q10, 0.2, atol=0.02)
    assert np.isclose(q50, 0.5, atol=0.02)
    assert np.isclose(q90, 0.8, atol=0.02)


def test_truncated_normal_bounds():
    rng = rng_from_seed(42)
    samples = sample_truncated_normal(rng, mean=0.0, sd=1.0, min_v=-2.0, max_v=2.0, n=100_000)
    assert np.all(samples >= -2.0 - 1e-6)
    assert np.all(samples <= 2.0 + 1e-6)


def test_truncated_lognormal_bounds():
    rng = rng_from_seed(99)
    samples = sample_truncated_lognormal(
        rng,
        mean=10.0,
        sd=3.0,
        min_v=5.0,
        max_v=25.0,
        n=100_000,
    )
    assert np.all(samples >= 5.0 - 1e-6)
    assert np.all(samples <= 25.0 + 1e-6)
    assert np.isfinite(samples).all()


def test_burr_sampling_positive():
    rng = rng_from_seed(7)
    samples = sample_burr(rng, c=2.0, d=5.0, scale=1.5, loc=0.0, n=50_000)
    assert np.all(samples >= 0.0)
    assert np.isfinite(samples).all()
    assert np.mean(samples) > 0


def test_johnson_su_sampling():
    rng = rng_from_seed(11)
    samples = sample_johnson_su(rng, gamma=0.5, delta=1.5, loc=0.0, scale=2.0, n=50_000)
    assert np.isfinite(samples).all()
    assert not np.allclose(np.mean(samples), 0.0)

