"""Backend helpers using array-api-compat for NumPy/PyTorch support."""

from array_api_compat import (
    array_namespace as get_namespace,
    device as xpd,
    is_numpy_namespace,
    is_torch_namespace,
)
from array_api_extra import (
    atleast_nd,
    create_diagonal,
    expand_dims,
)
import numpy as np


__all__ = [
    # Re-exported from array-api-compat
    "get_namespace",
    "xpd",
    "is_numpy_namespace",
    "is_torch_namespace",
    # Re-exported from array-api-extra
    "create_diagonal",
    "expand_dims",
    # Custom
    "check_matrix_pair",
    "to_numpy",
    "from_numpy",
    "check_like",
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


def _numpy_to_xp_kwargs(kwds):
    """Map numpy cov/corrcoef kwargs to torch-compatible kwargs.

    numpy uses ``bias`` and ``ddof``, torch uses ``correction``.
    ``fweights`` and ``aweights`` have the same name in both.
    """
    out = {}
    if "bias" in kwds:
        out["correction"] = 0 if kwds.pop("bias") else 1
    if "ddof" in kwds:
        out["correction"] = kwds.pop("ddof")
    # fweights/aweights: same name
    for k in ("fweights", "aweights"):
        if k in kwds:
            out[k] = kwds.pop(k)
    # rowvar, dtype, y: numpy-only, drop silently
    return out


def _apply_xp(func, X, **kwds):
    """Call an array-api function, translating kwargs across backends."""
    xp = get_namespace(X)
    if is_numpy_namespace(xp):
        C = func(X, **kwds)
    else:
        C = func(X, **_numpy_to_xp_kwargs(kwds))
    return atleast_nd(C, ndim=2)


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
