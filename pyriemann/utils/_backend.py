"""Backend helpers using array-api-compat for NumPy/PyTorch support."""

from array_api_compat import (
    array_namespace as get_namespace,
    device as xpd,
    is_torch_namespace,
)
import numpy as np


__all__ = [
    "check_matrix_pair",
    "to_numpy",
    "from_numpy",
    "check_like",
    "diag_indices",
    "tril_indices",
    "triu_indices",
    "torch",
]

try:
    import torch
except ImportError:
    torch = None


def to_numpy(X):
    """Convert array to numpy, detaching from torch if needed."""
    return X if isinstance(X, np.ndarray) else X.detach().cpu().numpy()


def from_numpy(X, *, like):
    """Convert numpy array to the same backend/device as ``like``."""
    xp = get_namespace(like)
    return xp.asarray(X, dtype=like.dtype, device=xpd(like))


def check_like(like):
    if like is None:
        xp, dev = np, None
    else:
        xp, dev = get_namespace(like), xpd(like)
    return xp, dev


def check_matrix_pair(A, B, *, require_square=False):
    xp = get_namespace(A, B)
    if A.ndim < 2 or B.ndim < 2:
        raise ValueError("Inputs must be at least 2D arrays")
    if A.shape[-2:] != B.shape[-2:]:
        raise ValueError("Inputs must have equal matrix dimensions")
    if require_square and A.shape[-2] != A.shape[-1]:
        raise ValueError("Inputs must contain square matrices")
    if A.shape != B.shape:
        try:
            xp.broadcast_shapes(A.shape[:-2], B.shape[:-2])
        except Exception as exc:
            raise ValueError("Inputs have incompatible dimensions.") from exc
    return xp


def diag_indices(n, *, like=None):
    xp, dev = check_like(like)
    if is_torch_namespace(xp):
        idx = torch.arange(n, device=dev)
        return idx, idx
    return np.diag_indices(n)


def tril_indices(n, k=0, *, like=None):
    xp, dev = check_like(like)
    if is_torch_namespace(xp):
        return torch.tril_indices(n, n, offset=k, device=dev)
    return np.tril_indices(n, k)


def triu_indices(n, k=0, *, like=None):
    xp, dev = check_like(like)
    if is_torch_namespace(xp):
        return torch.triu_indices(n, n, offset=k, device=dev)
    return np.triu_indices(n, k)
