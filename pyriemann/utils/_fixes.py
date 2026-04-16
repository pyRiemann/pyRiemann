"""Compatibility fixes for sklearn covariance estimators.

Provides array-API-compatible versions of sklearn covariance functions
(ledoit_wolf, oas, fast_mcd) that work with both numpy and torch tensors.

When sklearn adds native array API support for these functions, this module
can be removed. Check sklearn tags with:

    >>> from sklearn.covariance import LedoitWolf
    >>> LedoitWolf().__sklearn_tags__().array_api_support

If you add content to this file, please give the version of the package
at which the fix is no longer needed.
"""

# Authors: pyRiemann developers
# License: BSD-3-Clause

from sklearn.covariance import (
    fast_mcd as _sklearn_mcd,
    ledoit_wolf as _sklearn_lw,
    oas as _sklearn_oas,
)

from ._backend import (
    diag_indices, get_namespace, is_numpy_namespace, to_numpy, from_numpy,
)


# --------------------------------------------------------------------------
# Empirical covariance (needed by ledoit_wolf and oas)
# --------------------------------------------------------------------------

def _empirical_covariance(X, assume_centered=False, xp=None):
    """Empirical covariance, array-API compatible.

    Uses post-hoc centering to avoid allocating a centered copy of X:
    ``cov = (X.T @ X - n * outer(mean, mean)) / n``. This matches the
    approach taken by sklearn PR #33573 for non-numpy backends. Cancellation
    error is negligible for signals with mean close to zero (e.g. bandpass-
    filtered EEG/MEG).
    """
    if xp is None:
        xp = get_namespace(X)
    n_samples = X.shape[0]
    cov = X.mT @ X
    if not assume_centered:
        mean = xp.mean(X, axis=0)
        cov = cov - n_samples * (mean[:, None] * mean[None, :])
    return cov / n_samples


# --------------------------------------------------------------------------
# Ledoit-Wolf shrinkage (fix needed until sklearn >= ?.?)
# --------------------------------------------------------------------------

def _ledoit_wolf_shrinkage(X, assume_centered=False, xp=None):
    """Ledoit-Wolf shrinkage coefficient, array-API compatible.

    Non-blocked GPU-friendly implementation (single large matmul).
    """
    if xp is None:
        xp = get_namespace(X)
    n_samples, n_features = X.shape

    if n_features == 1:
        return 0.0

    if not assume_centered:
        X = X - xp.mean(X, axis=0)

    X2 = X ** 2
    emp_cov_trace = xp.sum(X2, axis=0) / n_samples
    mu = float(xp.sum(emp_cov_trace)) / n_features

    # GPU-friendly: single matmul for delta_ and a row-sum identity for beta_.
    # Identity: sum(X2.T @ X2) == sum(sum(X2, axis=1) ** 2), which avoids
    # allocating the (n_features, n_features) intermediate.
    beta_ = float(xp.sum(xp.sum(X2, axis=1) ** 2))
    delta_ = float(xp.sum((X.mT @ X) ** 2)) / n_samples ** 2

    beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)
    delta = delta_ - 2.0 * mu * float(xp.sum(emp_cov_trace)) + \
        n_features * mu ** 2
    delta /= n_features

    beta = min(beta, delta)
    shrinkage = 0 if delta == 0 else beta / delta
    return shrinkage


def ledoit_wolf(X, assume_centered=False, block_size=1000):
    """Ledoit-Wolf shrunk covariance, array-API compatible.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data from which to compute the covariance estimate.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split
        for numpy. Ignored for non-numpy backends.

    Returns
    -------
    shrunk_cov : ndarray, shape (n_features, n_features)
        Shrunk covariance.
    shrinkage : float
        Coefficient in the convex combination.
    """
    xp = get_namespace(X)

    if is_numpy_namespace(xp):
        return _sklearn_lw(X, assume_centered=assume_centered,
                           block_size=block_size)

    # Non-numpy: use our array-API implementation
    n_features = X.shape[1]
    if n_features == 1:
        if not assume_centered:
            X = X - xp.mean(X)
        return xp.reshape(xp.mean(X ** 2), (1, 1)), 0.0

    shrinkage = _ledoit_wolf_shrinkage(
        X, assume_centered=assume_centered, xp=xp)
    emp_cov = _empirical_covariance(X, assume_centered=assume_centered, xp=xp)
    mu = float(xp.linalg.trace(emp_cov)) / n_features
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    idx = diag_indices(n_features, like=shrunk_cov)
    shrunk_cov[..., idx[0], idx[1]] += shrinkage * mu
    return shrunk_cov, shrinkage


# --------------------------------------------------------------------------
# OAS shrinkage (fix needed until sklearn >= ?.?)
# --------------------------------------------------------------------------

def oas(X, assume_centered=False):
    """OAS shrunk covariance, array-API compatible.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data from which to compute the covariance estimate.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.

    Returns
    -------
    shrunk_cov : ndarray, shape (n_features, n_features)
        Shrunk covariance.
    shrinkage : float
        Coefficient in the convex combination.
    """
    xp = get_namespace(X)

    if is_numpy_namespace(xp):
        return _sklearn_oas(X, assume_centered=assume_centered)

    # Non-numpy: our array-API implementation
    n_samples, n_features = X.shape

    if n_features == 1:
        if not assume_centered:
            X = X - xp.mean(X)
        return xp.reshape(xp.mean(X ** 2), (1, 1)), 0.0

    emp_cov = _empirical_covariance(X, assume_centered=assume_centered, xp=xp)

    alpha = float(xp.mean(emp_cov ** 2))
    mu = float(xp.linalg.trace(emp_cov)) / n_features
    mu_squared = mu ** 2

    num = alpha + mu_squared
    den = (n_samples + 1) * (alpha - mu_squared / n_features)
    shrinkage = 1.0 if den == 0 else min(num / den, 1.0)

    shrunk_cov = (1.0 - shrinkage) * emp_cov
    idx = diag_indices(n_features, like=shrunk_cov)
    shrunk_cov[..., idx[0], idx[1]] += shrinkage * mu
    return shrunk_cov, shrinkage


# --------------------------------------------------------------------------
# fast_mcd — no torch implementation exists anywhere (checked March 2026).
# Falls back to numpy via sklearn. Remove when sklearn adds array API support
# for MinCovDet (currently array_api_support=False).
# --------------------------------------------------------------------------

def fast_mcd(X, **kwds):
    """Minimum Covariance Determinant estimator.

    For numpy, delegates to sklearn. For non-numpy backends (e.g. torch),
    converts to numpy and back — no native torch MCD implementation exists.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data from which to compute the MCD estimate.

    Returns
    -------
    location : ndarray, shape (n_features,)
        Robust location.
    covariance : ndarray, shape (n_features, n_features)
        Robust covariance.
    support : ndarray, shape (n_samples,), dtype=bool
        Mask of observations used for the estimate.
    dist : ndarray, shape (n_samples,)
        Mahalanobis distances.
    """
    xp = get_namespace(X)
    if is_numpy_namespace(xp):
        return _sklearn_mcd(X, **kwds)

    # No torch MCD exists — convert to numpy and back
    X_np = to_numpy(X)
    result = _sklearn_mcd(X_np, **kwds)
    return tuple(
        from_numpy(r, like=X) if hasattr(r, 'shape') else r
        for r in result
    )
