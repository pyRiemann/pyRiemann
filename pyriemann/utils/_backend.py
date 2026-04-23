"""Backend helpers using array-api-compat for NumPy/PyTorch support."""

from array_api_compat import (
    array_namespace as get_namespace,
    device as xpd,
    is_numpy_array,
    is_numpy_namespace,
    is_torch_array,
)
from array_api_extra import atleast_nd
import numpy as np


__all__ = [
    "as_numpy",
    "from_numpy",
    "diag_indices",
    "hann_window",
    "nanmean",
    "tril_indices",
    "triu_indices",
    "apply_xp_cov",
    "torch",
]

try:
    import torch
except ImportError:
    torch = None


def as_numpy(X):
    """Convert array to numpy, detaching from torch if needed.

    For torch tensors, ``resolve_conj`` and ``resolve_neg`` are called first
    to materialize lazy conjugate/negative-stride views (no array-API
    equivalent: ``xp.conj`` computes a new conjugate, it does not resolve
    an existing conjugate bit).

    Notes
    -----
    .. versionadded:: 0.12
    """
    if is_numpy_array(X):
        return X
    if is_torch_array(X):
        return X.resolve_conj().resolve_neg().detach().cpu().numpy()
    raise TypeError(
        f"as_numpy expects a numpy.ndarray or torch.Tensor, got "
        f"{type(X).__name__}"
    )


def from_numpy(X, *, like):
    """Convert numpy array to the same backend/device as ``like``.

    Notes
    -----
    .. versionadded:: 0.12
    """
    xp = get_namespace(like)
    return xp.asarray(X, dtype=like.dtype, device=xpd(like))


def diag_indices(n, *, like=None):
    """Indices of the main diagonal of an n-by-n array.

    Notes
    -----
    .. versionadded:: 0.12
    """
    if is_torch_array(like):
        idx = torch.arange(n, device=xpd(like))
        return idx, idx
    return np.diag_indices(n)


def tril_indices(n, k=0, *, like=None):
    """Indices of the lower triangle of an n-by-n array.

    Notes
    -----
    .. versionadded:: 0.12
    """
    if is_torch_array(like):
        return torch.tril_indices(n, n, offset=k, device=xpd(like))
    return np.tril_indices(n, k)


def triu_indices(n, k=0, *, like=None):
    """Indices of the upper triangle of an n-by-n array.

    Notes
    -----
    .. versionadded:: 0.12
    """
    if is_torch_array(like):
        return torch.triu_indices(n, n, offset=k, device=xpd(like))
    return np.triu_indices(n, k)


def hann_window(n, *, like):
    """Symmetric Hann window of length ``n`` in the namespace of ``like``.

    Matches ``numpy.hanning(n)`` element-wise but inherits backend, dtype
    and device from ``like``. Used internally by spectral estimators.

    Parameters
    ----------
    n : int
        Window length.
    like : ndarray
        Reference array used to infer the array-API namespace, dtype and
        device of the returned window.

    Returns
    -------
    win : ndarray, shape (n,)
        Hann window.

    Notes
    -----
    .. versionadded:: 0.12
    """
    xp = get_namespace(like)
    dev = xpd(like)
    if n == 1:
        return xp.ones(1, dtype=like.dtype, device=dev)
    k = xp.arange(n, dtype=like.dtype, device=dev)
    return 0.5 - 0.5 * xp.cos(2 * np.pi * k / (n - 1))


def nanmean(X, axis=0):
    """Mean along ``axis`` ignoring NaN values; backend-agnostic.

    Equivalent to ``numpy.nanmean(X, axis=axis)`` but works for any
    array-API namespace. When the input contains no NaN, the result
    matches ``xp.mean(X, axis=axis)``.

    Parameters
    ----------
    X : ndarray
        Input array.
    axis : int, default=0
        Axis along which to compute the mean.

    Notes
    -----
    .. versionadded:: 0.12
    """
    xp = get_namespace(X)
    isnan = xp.isnan(X)
    X_clean = xp.where(isnan, xp.zeros_like(X), X)
    return xp.sum(X_clean, axis=axis) / xp.sum(~isnan, axis=axis)


# ``np.cov`` / ``np.corrcoef`` use ``bias``/``ddof``; ``torch.cov`` /
# ``torch.corrcoef`` use ``correction``. Translate the kwargs so callers
# need not care about the backend.
def _cov_kwargs_to_xp(kwds):
    out = {}
    if "bias" in kwds:
        out["correction"] = 0 if kwds.pop("bias") else 1
    if "ddof" in kwds:
        out["correction"] = kwds.pop("ddof")
    for k in ("fweights", "aweights"):
        if k in kwds:
            out[k] = kwds.pop(k)
    return out


def apply_xp_cov(func, X, **kwds):
    """Call an array-API ``cov``/``corrcoef``, translating numpy kwargs.

    Numpy and torch use different keyword names for the unbiased correction
    (``bias``/``ddof`` vs ``correction``); this dispatches to ``func`` with
    the right kwargs and ensures the result is at least 2D.

    Notes
    -----
    .. versionadded:: 0.12
    """
    xp = get_namespace(X)
    if is_numpy_namespace(xp):
        C = func(X, **kwds)
    else:
        C = func(X, **_cov_kwargs_to_xp(kwds))
    return atleast_nd(C, ndim=2)


###############################################################################


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

from sklearn.covariance import (
    fast_mcd as _sklearn_mcd,
    ledoit_wolf as _sklearn_lw,
    oas as _sklearn_oas,
)


def _add_to_diagonal(X, value, xp):
    """Add a scalar value to the diagonal of a matrix."""
    n = X.shape[-1]
    idx = xp.arange(n, device=getattr(X, 'device', None))
    X[..., idx, idx] += value


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
    mu = xp.sum(emp_cov_trace) / n_features

    # GPU-friendly: single matmul for delta_ and a row-sum identity for beta_.
    # Identity: sum(X2.T @ X2) == sum(sum(X2, axis=1) ** 2), which avoids
    # allocating the (n_features, n_features) intermediate.
    beta_ = xp.sum(xp.sum(X2, axis=1) ** 2)
    delta_ = xp.sum((X.mT @ X) ** 2) / n_samples ** 2

    beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)
    delta = delta_ - 2.0 * mu * xp.sum(emp_cov_trace) + \
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
    mu = xp.linalg.trace(emp_cov) / n_features
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    _add_to_diagonal(shrunk_cov, shrinkage * mu, xp)
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

    alpha = xp.mean(emp_cov ** 2)
    mu = xp.linalg.trace(emp_cov) / n_features
    mu_squared = mu ** 2

    num = alpha + mu_squared
    den = (n_samples + 1) * (alpha - mu_squared / n_features)
    shrinkage = 1.0 if den == 0 else min(num / den, 1.0)

    shrunk_cov = (1.0 - shrinkage) * emp_cov
    _add_to_diagonal(shrunk_cov, shrinkage * mu, xp)
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
    X_np = as_numpy(X)
    result = _sklearn_mcd(X_np, **kwds)
    return tuple(
        from_numpy(r, like=X) if hasattr(r, 'shape') else r
        for r in result
    )
