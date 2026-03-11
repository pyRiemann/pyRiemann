"""Backend helpers using array-api-compat for NumPy/torch support."""

import numpy as np
from array_api_compat import (
    array_namespace,
    device as xpd,
    is_numpy_namespace,
    is_torch_namespace,
)

try:  # pragma: no cover - torch is optional
    import torch
except ImportError:  # pragma: no cover - torch is optional
    torch = None


def is_torch_tensor(x):
    """Return True if x is a torch tensor."""
    return torch is not None and isinstance(x, torch.Tensor)


def is_numpy_array(x):
    """Return True if x is a NumPy ndarray."""
    return isinstance(x, np.ndarray)


def is_backend_array(x):
    """Return True if x is a supported array type."""
    return is_numpy_array(x) or is_torch_tensor(x)


def _broadcast_batch_shapes(*arrays):
    """Raise a ValueError if matrix batch dimensions are not broadcastable."""
    batch_shapes = [x.shape[:-2] for x in arrays]
    try:
        return np.broadcast_shapes(*batch_shapes)
    except ValueError as exc:
        raise ValueError("Inputs have incompatible dimensions.") from exc


def get_namespace(*xs):
    """Resolve the array namespace from array inputs.

    Filters out None values and delegates to array_api_compat.array_namespace.
    Falls back to NumPy when no array inputs are provided.
    """
    arrays = [x for x in xs if x is not None and is_backend_array(x)]
    if not arrays:
        return array_namespace(np.empty(0))
    return array_namespace(*arrays)


def check_matrix_pair(A, B, *, require_square=False):
    """Check two arrays/tensors for pairwise matrix operations."""
    xp = get_namespace(A, B)
    if not is_backend_array(A) or not is_backend_array(B):
        raise ValueError("Inputs must be ndarrays or tensors")
    if A.ndim < 2 or B.ndim < 2:
        raise ValueError("Inputs must be at least 2D arrays")
    if A.shape[-2:] != B.shape[-2:]:
        raise ValueError("Inputs must have equal matrix dimensions")
    if require_square and A.shape[-2] != A.shape[-1]:
        raise ValueError("Inputs must contain square matrices")
    if A.shape != B.shape:
        _broadcast_batch_shapes(A, B)
    return xp


# --- Custom extensions not in Array API ---


def weighted_average(x, weights=None, axis=0, *, xp):
    """Weighted average along an axis."""
    if weights is None:
        return xp.mean(x, axis=axis)
    weights = xp.asarray(weights, dtype=x.dtype, device=xpd(x))
    return xp.tensordot(weights, x, axes=([0], [axis]))


def diag_embed(x, *, xp):
    """Create diagonal matrices from the last dimension of x."""
    if is_torch_namespace(xp):
        return torch.diag_embed(x)
    return xp.eye(x.shape[-1], dtype=x.dtype, device=xpd(x)) * x[..., None]


def diag_indices(n, *, xp, like=None):
    """Return indices of diagonal elements."""
    if is_torch_namespace(xp):
        dev = None if like is None else xpd(like)
        idx = torch.arange(n, device=dev)
        return idx, idx
    return np.diag_indices(n)


def tril_indices(n, k=0, *, xp, like=None):
    """Return lower-triangular indices."""
    if is_torch_namespace(xp):
        dev = None if like is None else xpd(like)
        return torch.tril_indices(n, n, offset=k, device=dev)
    return np.tril_indices(n, k)


def triu_indices(n, k=0, *, xp, like=None):
    """Return upper-triangular indices."""
    if is_torch_namespace(xp):
        dev = None if like is None else xpd(like)
        return torch.triu_indices(n, n, offset=k, device=dev)
    return np.triu_indices(n, k)
