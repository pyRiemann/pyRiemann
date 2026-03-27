"""Backend helpers using array-api-compat for NumPy/torch support."""

import numpy as np
from array_api_compat import (  # noqa: F401 - re-exported
    array_namespace as get_namespace,
    device as xpd,
    is_numpy_namespace,
    is_torch_namespace,
)
from array_api_extra import (  # noqa: F401 - re-exported
    broadcast_shapes,
    create_diagonal,
)

try:
    import torch
except ImportError:
    torch = None


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
            broadcast_shapes(A.shape[:-2], B.shape[:-2])
        except Exception as exc:
            raise ValueError("Inputs have incompatible dimensions.") from exc
    return xp


# --- Custom extensions not in Array API or array-api-extra ---


def weighted_average(x, weights=None, axis=0, *, xp):
    if weights is None:
        return xp.mean(x, axis=axis)
    weights = xp.asarray(weights, dtype=x.dtype, device=xpd(x))
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
