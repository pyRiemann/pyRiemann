"""Private backend helpers for NumPy and optional torch support."""

from dataclasses import dataclass

import numpy as np

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


def _require_torch():
    if torch is None:  # pragma: no cover - torch is optional
        raise ImportError(
            "Torch backend requested but torch is not installed."
        )


def _like_device(like):
    return None if like is None else like.device


def _module_ops(module, names):
    return {name: getattr(module, name) for name in names}


def _numpy_asarray(x, *, like=None, dtype=None):
    return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)


def _torch_asarray(x, *, like=None, dtype=None):
    _require_torch()
    device = _like_device(like)
    if isinstance(x, torch.Tensor):
        if dtype is None:
            dtype = x.dtype
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _numpy_real_dtype(x=None):
    if x is None:
        return np.float64
    return np.asarray(x).real.dtype


def _torch_real_dtype(x=None):
    _require_torch()
    if x is None:
        return torch.get_default_dtype()
    return x.real.dtype


def _numpy_diag_embed(x):
    return np.eye(x.shape[-1], dtype=x.dtype) * x[..., None]


def _numpy_weighted_average(x, weights=None, axis=0):
    if weights is None:
        return np.mean(x, axis=axis)
    return np.tensordot(weights, x, axes=([0], [axis]))


def _torch_weighted_average(x, weights=None, axis=0):
    if weights is None:
        return torch.mean(x, dim=axis)
    return torch.tensordot(weights, x, dims=([0], [axis]))


def _numpy_eye(n, *, like=None):
    dtype = None if like is None else like.dtype
    return np.eye(n, dtype=dtype)


def _torch_eye(n, *, like=None):
    _require_torch()
    return torch.eye(
        n,
        dtype=None if like is None else like.dtype,
        device=_like_device(like),
    )


def _numpy_ones(shape, *, like=None, dtype=None):
    if dtype is None and like is not None:
        dtype = _numpy_real_dtype(like)
    return np.ones(shape, dtype=dtype)


def _torch_ones(shape, *, like=None, dtype=None):
    _require_torch()
    if dtype is None and like is not None:
        dtype = _torch_real_dtype(like)
    return torch.ones(shape, dtype=dtype, device=_like_device(like))


def _numpy_zeros(shape, *, like=None, dtype=None):
    if dtype is None and like is not None:
        dtype = like.dtype
    return np.zeros(shape, dtype=dtype)


def _torch_zeros(shape, *, like=None, dtype=None):
    _require_torch()
    if dtype is None and like is not None:
        dtype = like.dtype
    return torch.zeros(shape, dtype=dtype, device=_like_device(like))


def _numpy_diag_indices(n, *, like=None):
    return np.diag_indices(n)


def _torch_diag_indices(n, *, like=None):
    _require_torch()
    idx = torch.arange(n, device=_like_device(like))
    return idx, idx


def _numpy_tril_indices(n, k=0, *, like=None):
    return np.tril_indices(n, k)


def _torch_tril_indices(n, k=0, *, like=None):
    _require_torch()
    return torch.tril_indices(n, n, offset=k, device=_like_device(like))


def _numpy_triu_indices(n, k=0, *, like=None):
    return np.triu_indices(n, k)


def _torch_triu_indices(n, k=0, *, like=None):
    _require_torch()
    return torch.triu_indices(n, n, offset=k, device=_like_device(like))


_SHARED_UNARY_OPS = (
    "conj",
    "real",
    "sqrt",
    "log",
    "exp",
    "abs",
    "imag",
    "cos",
    "sin",
)
_LINALG_OPS = (
    "eig",
    "eigh",
    "eigvalsh",
    "cholesky",
    "inv",
    "solve",
    "slogdet",
    "svd",
)

_NUMPY_OPS = {
    **_module_ops(np, _SHARED_UNARY_OPS),
    **_module_ops(np.linalg, _LINALG_OPS),
    "all": lambda x, axis=None: bool(np.all(x)) if axis is None else np.all(
        x,
        axis=axis,
    ),
    "all_finite": lambda x: np.isfinite(x).all(),
    "any": lambda x: bool(np.any(x)),
    "arctan2": np.arctan2,
    "concatenate": lambda xs, axis=0: np.concatenate(xs, axis=axis),
    "isclose": np.isclose,
    "isnan": np.isnan,
    "where": np.where,
    "min": lambda x, axis=None: np.min(x, axis=axis),
    "max": lambda x, axis=None: np.max(x, axis=axis),
    "minimum": np.minimum,
    "outer": np.outer,
    "sum": lambda x, axis=None: np.sum(x, axis=axis),
    "mean": lambda x, axis=0: np.mean(x, axis=axis),
    "stack": lambda xs, axis=0: np.stack(xs, axis=axis),
    "weighted_average": _numpy_weighted_average,
    "eye": _numpy_eye,
    "ones": _numpy_ones,
    "zeros": _numpy_zeros,
    "zeros_like": np.zeros_like,
    "diag_embed": _numpy_diag_embed,
    "diagonal": lambda x: np.diagonal(x, axis1=-2, axis2=-1),
    "diag_indices": _numpy_diag_indices,
    "tril_indices": _numpy_tril_indices,
    "triu_indices": _numpy_triu_indices,
    "swapaxes": np.swapaxes,
    "maximum": np.maximum,
    "norm_fro": lambda x: np.linalg.norm(x, ord="fro", axis=(-2, -1)),
}

_TORCH_OPS = {} if torch is None else {
    **_module_ops(torch, _SHARED_UNARY_OPS),
    **_module_ops(torch.linalg, _LINALG_OPS),
    "all": lambda x, axis=None: (
        bool(torch.all(x).item()) if axis is None else torch.all(x, dim=axis)
    ),
    "all_finite": lambda x: bool(torch.isfinite(x).all().item()),
    "any": lambda x: bool(torch.any(x).item()),
    "arctan2": torch.atan2,
    "concatenate": lambda xs, axis=0: torch.cat(xs, dim=axis),
    "isclose": torch.isclose,
    "isnan": torch.isnan,
    "where": torch.where,
    "min": lambda x, axis=None: torch.min(x) if axis is None else torch.min(
        x, dim=axis
    ).values,
    "max": lambda x, axis=None: torch.max(x) if axis is None else torch.max(
        x, dim=axis
    ).values,
    "minimum": lambda x, y: torch.minimum(
        x,
        torch.as_tensor(y, device=x.device, dtype=x.dtype),
    ),
    "outer": torch.outer,
    "sum": lambda x, axis=None: torch.sum(x) if axis is None else torch.sum(
        x, dim=axis
    ),
    "mean": lambda x, axis=0: torch.mean(x, dim=axis),
    "stack": lambda xs, axis=0: torch.stack(xs, dim=axis),
    "weighted_average": _torch_weighted_average,
    "eye": _torch_eye,
    "ones": _torch_ones,
    "zeros": _torch_zeros,
    "zeros_like": torch.zeros_like,
    "diag_embed": torch.diag_embed,
    "diagonal": lambda x: torch.diagonal(x, dim1=-2, dim2=-1),
    "diag_indices": _torch_diag_indices,
    "tril_indices": _torch_tril_indices,
    "triu_indices": _torch_triu_indices,
    "swapaxes": torch.swapaxes,
    "maximum": lambda x, y: torch.maximum(
        x, torch.as_tensor(y, device=x.device, dtype=x.dtype)
    ),
    "norm_fro": lambda x: torch.linalg.norm(x, ord="fro", dim=(-2, -1)),
}


@dataclass(frozen=True)
class _Backend:
    """Thin adapter around backend-specific operation tables."""

    name: str
    array_type: object
    ops: dict
    asarray_fn: object
    real_dtype_fn: object
    is_floating_dtype_fn: object

    def __post_init__(self):
        for name, fn in self.ops.items():
            object.__setattr__(self, name, fn)

    def is_array(self, x):
        return isinstance(x, self.array_type)

    def asarray(self, x, *, like=None, dtype=None):
        return self.asarray_fn(x, like=like, dtype=dtype)

    def real_dtype(self, x=None):
        return self.real_dtype_fn(x)

    def is_floating_dtype(self, x):
        return self.is_floating_dtype_fn(x)

    @staticmethod
    def as_float(x):
        """Convert a scalar array/tensor to a Python float."""
        return x.item() if hasattr(x, "item") else float(x)

    def __getattr__(self, name):
        try:
            return self.ops[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


numpy_backend = _Backend(
    name="numpy",
    array_type=np.ndarray,
    ops=_NUMPY_OPS,
    asarray_fn=_numpy_asarray,
    real_dtype_fn=_numpy_real_dtype,
    is_floating_dtype_fn=lambda x: np.issubdtype(
        np.asarray(x).real.dtype,
        np.floating,
    ),
)

torch_backend = _Backend(
    name="torch",
    array_type=() if torch is None else torch.Tensor,
    ops=_TORCH_OPS,
    asarray_fn=_torch_asarray,
    real_dtype_fn=_torch_real_dtype,
    is_floating_dtype_fn=lambda x: x.dtype.is_floating_point,
)


def resolve_backend(*xs, backend=None):
    """Resolve the backend from array inputs and an optional backend override.
    """
    if isinstance(backend, _Backend):
        return backend
    if backend not in (None, "numpy", "torch"):
        raise ValueError(
            f"Unknown backend '{backend}'. Must be one of: numpy, torch."
        )

    has_numpy = any(is_numpy_array(x) for x in xs if x is not None)
    has_torch = any(is_torch_tensor(x) for x in xs if x is not None)

    if has_numpy and has_torch:
        raise ValueError(
            "Mixed NumPy arrays and torch tensors are not supported."
        )

    if backend == "numpy":
        if has_torch:
            raise ValueError(
                "backend='numpy' is incompatible with torch tensor inputs."
            )
        return numpy_backend

    if backend == "torch":
        if torch is None:
            raise ImportError(
                "backend='torch' requested but torch is not installed."
            )
        if has_numpy:
            raise ValueError(
                "backend='torch' is incompatible with NumPy array inputs."
            )
        return torch_backend

    if has_torch:
        return torch_backend

    return numpy_backend


def check_matrix_pair(A, B, *, require_square=False, backend=None):
    """Check two arrays/tensors for pairwise matrix operations."""
    backend = resolve_backend(A, B, backend=backend)
    if not backend.is_array(A) or not backend.is_array(B):
        raise ValueError("Inputs must be ndarrays or tensors")
    if A.ndim < 2 or B.ndim < 2:
        raise ValueError("Inputs must be at least 2D arrays")
    if A.shape[-2:] != B.shape[-2:]:
        raise ValueError("Inputs must have equal matrix dimensions")
    if require_square and A.shape[-2] != A.shape[-1]:
        raise ValueError("Inputs must contain square matrices")
    if A.shape != B.shape:
        _broadcast_batch_shapes(A, B)
    return backend
