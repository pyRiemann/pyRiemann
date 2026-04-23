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
