"""Backend helpers using array-api-compat for NumPy/PyTorch support."""

from array_api_compat import (
    array_namespace as get_namespace,
    device as xpd,
    is_numpy_array,
    is_torch_array,
)
import numpy as np


__all__ = [
    "as_numpy",
    "from_numpy",
    "diag_indices",
    "tril_indices",
    "triu_indices",
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
    if is_torch_array(like):
        idx = torch.arange(n, device=xpd(like))
        return idx, idx
    return np.diag_indices(n)


def tril_indices(n, k=0, *, like=None):
    if is_torch_array(like):
        return torch.tril_indices(n, n, offset=k, device=xpd(like))
    return np.tril_indices(n, k)


def triu_indices(n, k=0, *, like=None):
    if is_torch_array(like):
        return torch.triu_indices(n, n, offset=k, device=xpd(like))
    return np.triu_indices(n, k)
