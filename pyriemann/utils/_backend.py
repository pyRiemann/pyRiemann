"""Backend helpers using array-api-compat for NumPy/PyTorch support."""

from array_api_compat import (
    array_namespace as get_namespace,
    device as xpd,
    is_numpy_namespace,
    is_torch_namespace,
)
from array_api_extra import (
    create_diagonal,
)
import numpy as np
from scipy.linalg import eigvalsh


__all__ = [
    # Re-exported from array-api-compat
    "get_namespace",
    "xpd",
    "is_numpy_namespace",
    "is_torch_namespace",
    # Re-exported from array-api-extra
    "create_diagonal",
    # Custom
    "check_matrix_pair",
    "to_numpy",
    "from_numpy",
    "check_like",
    "weighted_average",
    "diag_indices",
    "tril_indices",
    "triu_indices",
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


# --- Custom extensions not in Array API or array-api-extra ---


def joint_eigvalsh(A, B, *, xp):
    """Joint eigenvalues of A and B, ie generalized eigvalsh(A, B).

    Uses scipy for numpy (direct generalized eigvalsh), Cholesky reduction
    for torch: L = chol(B), eigvalsh(L^{-1} A L^{-H}).
    """
    if is_numpy_namespace(xp) and A.shape == B.shape:
        # local import to avoid circular dependency (_backend <- base)
        from .base import _recursive
        return _recursive(eigvalsh, A, B)
    # local import to avoid circular dependency (_backend <- base)
    from .base import ctranspose
    L = xp.linalg.cholesky(B)
    Y = xp.linalg.solve(L, A)
    Z = ctranspose(xp.linalg.solve(L, ctranspose(Y)))
    return xp.linalg.eigvalsh(Z)


def pairwise_euclidean(X, Y, *, xp):
    """Pairwise Euclidean distance matrix, array-API compatible.

    Uses sklearn for numpy (optimized BLAS), torch.cdist for torch (GPU).
    """
    if is_numpy_namespace(xp):
        from sklearn.metrics import euclidean_distances
        return euclidean_distances(X, Y)
    return torch.cdist(X, Y, p=2)


def weighted_average(x, weights=None, axis=0, *, xp):
    """Weighted average along an axis, matching np.average behavior."""
    if weights is None:
        return xp.mean(x, axis=axis)
    weights = xp.asarray(weights, dtype=x.dtype, device=xpd(x))
    if weights.shape[0] != x.shape[axis]:
        raise ValueError(
            "Shape of weights must be consistent with shape of a "
            "along specified axis."
        )
    weights = weights / xp.sum(weights)
    return xp.tensordot(weights, x, axes=([0], [axis]))


def diag_indices(n, *, xp, like=None):
    if is_torch_namespace(xp):
        dev = None if like is None else xpd(like)
        idx = torch.arange(n, device=dev)
        return idx, idx
    return np.diag_indices(n)


def tril_indices(n, k=0, *, xp, like=None):
    if is_torch_namespace(xp):
        dev = None if like is None else xpd(like)
        return torch.tril_indices(n, n, offset=k, device=dev)
    return np.tril_indices(n, k)


def triu_indices(n, k=0, *, xp, like=None):
    if is_torch_namespace(xp):
        dev = None if like is None else xpd(like)
        return torch.triu_indices(n, n, offset=k, device=dev)
    return np.triu_indices(n, k)
