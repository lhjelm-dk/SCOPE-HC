"""
Sampling: samplers, correlated sampling, dependency matrix logic
"""
import numpy as np
from typing import Dict, Any, Tuple, List
from scipy.stats import norm, beta as beta_dist, lognorm, truncnorm, burr, johnsonsu
from scipy.optimize import least_squares


def rng_from_seed(seed: int):
    """Create random number generator from seed."""
    return np.random.default_rng(int(seed))


def sample_uniform(rng, low, high, n):
    """Sample from uniform distribution."""
    return rng.uniform(low, high, n)


def sample_triangular(rng, low, mode, high, n):
    """Sample from triangular distribution."""
    return rng.triangular(low, mode, high, n)


def sample_pert(rng, min_v, mode_v, max_v, n, lam=4.0):
    """
    Sample from PERT distribution.
    
    Args:
        rng: Random number generator
        min_v: Minimum value
        mode_v: Mode value
        max_v: Maximum value
        n: Number of samples
        lam: Lambda parameter (default 4.0)
        
    Returns:
        Array of samples
    """
    a, b, c = float(min_v), float(mode_v), float(max_v)
    if c == a:
        return np.full(n, a)
    
    alpha = 1.0 + lam * (mode_v - a) / (c - a)
    beta = 1.0 + lam * (c - mode_v) / (c - a)
    y = rng.beta(alpha, beta, n)
    return a + y * (c - a)


def sample_lognormal_mean_sd(rng, mean, sd, n):
    """
    Sample from lognormal distribution with mean and standard deviation.
    
    Args:
        rng: Random number generator
        mean: Mean of the lognormal distribution
        sd: Standard deviation of the lognormal distribution
        n: Number of samples
        
    Returns:
        Array of samples
    """
    sigma2 = np.log(1.0 + (sd * sd) / (mean * mean))
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - 0.5 * sigma2
    return rng.lognormal(mean=mu, sigma=sigma, size=n)


def sample_truncated_normal(rng, mean, sd, min_v, max_v, n):
    """
    Sample from a truncated normal distribution.
    """
    mean = float(mean)
    sd = max(float(sd), 1e-12)
    min_v = float(min_v)
    max_v = float(max_v)
    if max_v <= min_v:
        return np.full(n, min_v)
    a = (min_v - mean) / sd
    b = (max_v - mean) / sd
    u = rng.uniform(1e-12, 1.0 - 1e-12, size=n)
    samples = truncnorm.ppf(u, a, b, loc=mean, scale=sd)
    return np.clip(samples, float(min_v), float(max_v))


def sample_truncated_lognormal(rng, mean, sd, min_v, max_v, n):
    """
    Sample from a truncated lognormal distribution defined by arithmetic mean/sd.
    """
    mean = max(float(mean), 1e-12)
    sd = max(float(sd), 1e-12)
    min_v = max(float(min_v), 1e-12)
    max_v = max(float(max_v), min_v + 1e-12)
    if max_v <= min_v:
        return np.full(n, min_v)

    sigma2 = np.log(1.0 + (sd * sd) / (mean * mean))
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - 0.5 * sigma2

    log_min = (np.log(min_v) - mu) / sigma
    log_max = (np.log(max_v) - mu) / sigma

    u = rng.uniform(1e-12, 1.0 - 1e-12, size=n)
    log_samples = truncnorm.ppf(u, log_min, log_max, loc=0.0, scale=1.0)
    samples = np.exp(mu + sigma * log_samples)
    return np.clip(samples, min_v, max_v)


def sample_burr(rng, c, d, scale, loc, n):
    """
    Sample from Burr distribution (SciPy burr, a.k.a. Burr XII).
    """
    c = max(float(c), 1e-9)
    d = max(float(d), 1e-9)
    scale = max(float(scale), 1e-12)
    loc = float(loc)
    dist = burr(c, d, loc=loc, scale=scale)
    u = rng.uniform(1e-12, 1.0 - 1e-12, size=n)
    samples = dist.ppf(u)
    return np.asarray(samples, dtype=float)


def sample_johnson_su(rng, gamma, delta, loc, scale, n):
    """
    Sample from Johnson SU distribution.
    """
    delta = max(float(delta), 1e-9)
    scale = max(float(scale), 1e-12)
    gamma = float(gamma)
    loc = float(loc)
    dist = johnsonsu(gamma, delta, loc=loc, scale=scale)
    u = rng.uniform(1e-12, 1.0 - 1e-12, size=n)
    samples = dist.ppf(u)
    return np.asarray(samples, dtype=float)


def _fit_beta_from_quantiles(p10, p50, p90, eps=1e-9):
    """
    Fit beta(alpha, beta) parameters to match specified quantiles in [0, 1].
    """
    qs = np.array([0.10, 0.50, 0.90], dtype=float)
    targets = np.clip(np.array([p10, p50, p90], dtype=float), eps, 1.0 - eps)

    x0 = np.array([2.0, 2.0], dtype=float)

    def resid(x):
        a, b = np.maximum(x, eps)
        return beta_dist.ppf(qs, a, b) - targets

    bounds = (np.array([eps, eps]), np.array([100.0, 100.0]))
    res = least_squares(resid, x0, bounds=bounds)
    a, b = np.maximum(res.x, eps)
    return float(a), float(b)


def sample_beta_subjective(rng, min_v, max_v, p10, p50, p90, n):
    """
    Sample from subjective beta distribution using P10, P50, P90.
    """
    denom = max_v - min_v
    if denom <= 0:
        return np.full(n, float(min_v))

    p10n = (p10 - min_v) / denom
    p50n = (p50 - min_v) / denom
    p90n = (p90 - min_v) / denom

    a, b = _fit_beta_from_quantiles(p10n, p50n, p90n)
    y = rng.beta(a, b, size=n)
    return min_v + y * denom


def sample_stretched_beta(rng, min_v, max_v, mode, n, stretch=1.0):
    """
    Sample from stretched beta distribution.
    
    Args:
        rng: Random number generator
        min_v: Minimum value
        max_v: Maximum value
        mode: Mode value
        n: Number of samples
        stretch: Stretch parameter
        
    Returns:
        Array of samples
    """
    # Normalize mode to [0, 1]
    mode_norm = (mode - min_v) / (max_v - min_v)
    
    # Calculate alpha and beta parameters
    alpha = 1.0 + stretch * mode_norm
    beta = 1.0 + stretch * (1.0 - mode_norm)
    
    y = rng.beta(alpha, beta, n)
    return min_v + y * (max_v - min_v)


def _rearrange_perfect_dependence(x: np.ndarray, y: np.ndarray, sign: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rearrange y to achieve perfect dependence with x while preserving marginals.
    
    Args:
        x: First array
        y: Second array to rearrange
        sign: >0 for comonotonic (both ascending), <0 for countermonotonic
        
    Returns:
        Tuple of (x_sorted, y_rearranged)
    """
    order_x = np.argsort(x)
    order_y = np.argsort(y)
    if sign >= 0:
        y_re = y[order_y]
    else:
        y_re = y[order_y[::-1]]
    x_sorted = x[order_x]
    return x_sorted, y_re


def correlated_samples(rng, params_cfg, corr_matrix, param_names, n):
    """
    Generate correlated samples using Cholesky decomposition with Higham projection.
    """
    if not param_names or n <= 0:
        return {}

    corr = _nearest_correlation_matrix(corr_matrix)

    extreme_pairs: List[Tuple[int, int, float]] = []
    for i in range(len(param_names)):
        for j in range(i + 1, len(param_names)):
            rho = corr[i, j]
            if abs(rho) >= 0.999:
                sign = float(np.sign(rho) or 1.0)
                extreme_pairs.append((i, j, sign))
                corr[i, j] = corr[j, i] = sign * 0.98

    z = rng.standard_normal((n, len(param_names)))

    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("Unable to obtain Cholesky factor for correlation matrix") from exc

    correlated_z = z @ L.T
    uniform_samples = norm.cdf(correlated_z)

    samples: Dict[str, np.ndarray] = {}
    for i, param_name in enumerate(param_names):
        cfg = params_cfg.get(param_name)
        if not cfg:
            continue

        dist_type = str(cfg.get('distribution', 'uniform')).lower()
        u = np.clip(uniform_samples[:, i], 1e-12, 1.0 - 1e-12)

        if dist_type == 'uniform':
            a, b = cfg['min'], cfg['max']
            samples[param_name] = a + u * (b - a)
        elif dist_type == 'pert':
            a, m, b = cfg['min'], cfg['mode'], cfg['max']
            lam = cfg.get('lam', 4.0)
            alpha = 1.0 + lam * (m - a) / (b - a)
            beta_param = 1.0 + lam * (b - m) / (b - a)
            samples[param_name] = a + beta_dist.ppf(u, alpha, beta_param) * (b - a)
        elif dist_type == 'triangular':
            a, m, b = cfg['min'], cfg['mode'], cfg['max']
            mode_pos = (m - a) / (b - a)
            samples[param_name] = np.where(
                u <= mode_pos,
                a + np.sqrt(u * (b - a) * (m - a)),
                b - np.sqrt((1.0 - u) * (b - a) * (b - m)),
            )
        elif dist_type.startswith('lognormal'):
            mean, sd = cfg['mean'], cfg['sd']
            sigma2 = np.log(1.0 + (sd * sd) / (mean * mean))
            sigma = np.sqrt(sigma2)
            mu = np.log(mean) - 0.5 * sigma2
            samples[param_name] = lognorm.ppf(u, sigma, scale=np.exp(mu))
        elif dist_type == 'stretched beta':
            a, m, b = cfg['min'], cfg['mode'], cfg['max']
            stretch = cfg.get('stretch', cfg.get('lam', 4.0))
            mode_norm = (m - a) / (b - a)
            alpha = 1.0 + stretch * mode_norm
            beta_param = 1.0 + stretch * (1.0 - mode_norm)
            samples[param_name] = a + beta_dist.ppf(u, alpha, beta_param) * (b - a)
        elif dist_type == 'truncated normal':
            mean = cfg['mean']
            sd = max(cfg['sd'], 1e-12)
            a = (cfg['min'] - mean) / sd
            b = (cfg['max'] - mean) / sd
            samples[param_name] = truncnorm.ppf(u, a, b, loc=mean, scale=sd)
        elif dist_type == 'truncated lognormal':
            mean, sd = cfg['mean'], cfg['sd']
            mean = max(mean, 1e-12)
            sd = max(sd, 1e-12)
            sigma2 = np.log(1.0 + (sd * sd) / (mean * mean))
            sigma = np.sqrt(sigma2)
            mu = np.log(mean) - 0.5 * sigma2
            log_min = (np.log(max(cfg['min'], 1e-12)) - mu) / sigma
            log_max = (np.log(max(cfg['max'], 1e-12)) - mu) / sigma
            truncated = truncnorm.ppf(u, log_min, log_max)
            samples[param_name] = np.clip(
                np.exp(mu + sigma * truncated),
                cfg['min'],
                cfg['max'],
            )
        elif 'burr' in dist_type:
            c = max(cfg.get('c', 1.0), 1e-9)
            d = max(cfg.get('d', 1.0), 1e-9)
            scale = max(cfg.get('scale', 1.0), 1e-12)
            loc = float(cfg.get('loc', 0.0))
            samples[param_name] = burr.ppf(u, c, d, loc=loc, scale=scale)
        elif 'johnson' in dist_type:
            gamma = float(cfg.get('gamma', 0.0))
            delta = max(cfg.get('delta', 1.0), 1e-9)
            loc = float(cfg.get('loc', 0.0))
            scale = max(cfg.get('scale', 1.0), 1e-12)
            samples[param_name] = johnsonsu.ppf(u, gamma, delta, loc=loc, scale=scale)
        else:
            samples[param_name] = u

    for i, j, sign in extreme_pairs:
        key_i, key_j = param_names[i], param_names[j]
        if key_i in samples and key_j in samples:
            x_orig = samples[key_i].copy()
            y_orig = samples[key_j].copy()
            x_sorted, y_rearranged = _rearrange_perfect_dependence(x_orig, y_orig, sign)
            # Map y_rearranged back to original order of x
            order_x = np.argsort(x_orig)
            inv_order = np.empty_like(order_x)
            inv_order[order_x] = np.arange(len(order_x))
            samples[key_j] = y_rearranged[inv_order]

    return samples


def _nearest_correlation_matrix(A, tol=1e-12, max_iter=100):
    """
    Higham (2002) nearest correlation matrix projection.
    """
    X = np.array(A, dtype=float, copy=True)
    X = 0.5 * (X + X.T)
    np.fill_diagonal(X, 1.0)
    Y = X.copy()
    Delta_S = np.zeros_like(X)

    for _ in range(max_iter):
        R = Y - Delta_S
        eigval, eigvec = np.linalg.eigh(R)
        eigval = np.clip(eigval, 0.0, None)
        X = (eigvec * eigval) @ eigvec.T
        X = 0.5 * (X + X.T)
        np.fill_diagonal(X, 1.0)
        Delta_S = X - R
        Y = X.copy()
        if np.linalg.norm(X - R, ord='fro') <= tol:
            break

    X = np.clip(X, -0.999, 0.999)
    np.fill_diagonal(X, 1.0)
    return X


def validate_dependency_matrix(dep_matrix: np.ndarray, param_names: List[str]) -> Tuple[bool, str]:
    """
    Validate dependency matrix for reasonable values.
    """
    dep_matrix = np.asarray(dep_matrix, dtype=float)
    n = len(param_names)

    if dep_matrix.shape != (n, n):
        return False, "Dependency matrix shape does not match parameter count"

    if not np.allclose(dep_matrix, dep_matrix.T, atol=1e-8):
        return False, "Dependency matrix must be symmetric"

    if not np.allclose(np.diag(dep_matrix), 1.0, atol=1e-8):
        return False, "Diagonal entries must equal 1.0"

    off_diag = dep_matrix[np.triu_indices_from(dep_matrix, k=1)]
    if np.any(off_diag < -0.999) or np.any(off_diag > 0.999):
        return False, "Off-diagonal elements must be within [-0.999, 0.999]"

    try:
        _nearest_correlation_matrix(dep_matrix)
    except np.linalg.LinAlgError as exc:
        return False, f"Dependency matrix could not be stabilised: {exc}"

    return True, ""


def fix_correlation_matrix(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Project a matrix to the nearest correlation matrix.
    """
    fixed = _nearest_correlation_matrix(corr_matrix)
    fixed = np.clip(fixed, -0.999, 0.999)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def apply_correlation(x: np.ndarray, y: np.ndarray, correlation: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply correlation between two arrays using Cholesky decomposition.
    
    Args:
        x: First array of samples
        y: Second array of samples  
        correlation: Correlation coefficient between -1 and 1
        
    Returns:
        Tuple of (correlated_x, correlated_y)
    """
    if abs(correlation) < 1e-6:
        return x, y
    
    # Ensure correlation is within bounds
    correlation = max(-0.99, min(0.99, correlation))
    
    # Create correlation matrix
    corr_matrix = np.array([[1.0, correlation], [correlation, 1.0]])
    
    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # If not positive definite, use SVD
        U, s, Vt = np.linalg.svd(corr_matrix)
        L = U @ np.sqrt(np.diag(s)) @ Vt
    
    # Generate correlated normal random variables
    z = np.random.standard_normal((2, len(x)))
    correlated_normal = L @ z
    
    # Transform to uniform [0,1] using normal CDF
    u1 = norm.cdf(correlated_normal[0])
    u2 = norm.cdf(correlated_normal[1])
    
    # Apply inverse CDF to get correlated samples
    x_sorted = np.sort(x)
    x_indices = (u1 * (len(x) - 1)).astype(int)
    correlated_x = x_sorted[x_indices]
    
    y_sorted = np.sort(y)
    y_indices = (u2 * (len(y) - 1)).astype(int)
    correlated_y = y_sorted[y_indices]
    
    return correlated_x, correlated_y


def sample_scalar_dist(rng, dist_name: str, params: dict, n: int):
    """
    Sample a scalar from one of the supported distributions.
    
    Args:
        rng: Random number generator
        dist_name: Distribution name ("PERT", "Triangular", "Uniform")
        params: Dictionary with distribution parameters
        n: Number of samples
        
    Returns:
        Array of samples
    """
    dn = (dist_name or "PERT").lower()
    
    if dn == "pert":
        a, m, b = params.get("min"), params.get("mode"), params.get("max")
        return sample_pert(rng, a, m, b, n)
    if dn == "triangular":
        a, m, b = params.get("min"), params.get("mode"), params.get("max")
        return rng.triangular(a, m, b, n)
    if dn == "uniform":
        a, b = params.get("min"), params.get("max")
        return rng.uniform(a, b, n)
    if dn.startswith("lognormal"):
        mean, sd = params.get("mean"), params.get("sd")
        return sample_lognormal_mean_sd(rng, mean, sd, n)
    if dn == "subjective beta (vose)":
        return sample_beta_subjective(
            rng,
            params.get("min"),
            params.get("max"),
            params.get("p10", params.get("min")),
            params.get("p50", params.get("mode")),
            params.get("p90", params.get("max")),
            n,
        )
    if dn == "stretched beta":
        return sample_stretched_beta(
            rng,
            params.get("min"),
            params.get("max"),
            params.get("mode"),
            n,
            params.get("stretch", 1.0),
        )
    if dn == "truncated normal":
        return sample_truncated_normal(
            rng,
            params.get("mean"),
            params.get("sd"),
            params.get("min"),
            params.get("max"),
            n,
        )
    if dn == "truncated lognormal":
        return sample_truncated_lognormal(
            rng,
            params.get("mean"),
            params.get("sd"),
            params.get("min"),
            params.get("max"),
            n,
        )
    if dn.startswith("burr"):
        return sample_burr(
            rng,
            params.get("c", 1.0),
            params.get("d", 1.0),
            params.get("scale", 1.0),
            params.get("loc", 0.0),
            n,
        )
    if dn.startswith("johnson su") or "johnson" in dn:
        return sample_johnson_su(
            rng,
            params.get("gamma", 0.0),
            params.get("delta", 1.0),
            params.get("loc", 0.0),
            params.get("scale", 1.0),
            n,
        )

    # Fallback to constant 1.0 if unknown
    return np.ones(n, dtype=float)
