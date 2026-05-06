"""Shared helper functions for pyriemann.geometry.

This module contains validation, type-check, and decorator utilities used by
``pyriemann.geometry`` submodules. It has no internal pyriemann dependencies,
making it a safe leaf in the import graph.
"""

import warnings

import array_api_compat.numpy as _xpnp
from array_api_compat import (
    array_namespace as get_namespace,
    device as xpd,
    is_torch_namespace,
)
import numpy as np


# ---------------------------------------------------------------------------
# Array-API helpers
# ---------------------------------------------------------------------------

def check_like(like):
    """Resolve array-API namespace and device from a reference array.

    Parameters
    ----------
    like : None | ndarray
        Reference array. If None, returns the array-API NumPy namespace
        and a ``None`` device.

    Returns
    -------
    xp : module
        The array-API namespace (NumPy or PyTorch).
    dev : object | None
        The device of ``like`` if provided, else ``None``.

    Notes
    -----
    .. versionadded:: 0.12
    """
    if like is None:
        return _xpnp, None
    return get_namespace(like), xpd(like)


def check_matrix_pair(A, B, *, require_square=False):
    """Validate two matrix arrays share compatible matrix dimensions.

    Parameters
    ----------
    A, B : ndarray, shape (..., n, m)
        Input matrix arrays.
    require_square : bool, default=False
        If True, also require the matrix dimensions to be square.

    Returns
    -------
    xp : module
        The shared array-API namespace of ``A`` and ``B``.

    Notes
    -----
    .. versionadded:: 0.12
    """
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
        except (ValueError, RuntimeError) as exc:
            raise ValueError("Inputs have incompatible dimensions.") from exc
    return xp


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def check_weights(weights, n_weights, *, check_positivity=False, like=None):
    """Check weights.

    If input is None, output weights are equal.
    Strict positivity of weights can be checked.
    In any case, weights are normalized (sum equal to 1).

    Parameters
    ----------
    weights : None | ndarray, shape (n_weights,), default=None
        Input weights. If None, it provides equal weights.
    n_weights : int
        Number of weights to provide if None, or to check.
    check_positivity : bool, default=False
        Choose if strict positivity of weights is checked.
    like : None | ndarray, default=None
        Reference array used to infer the array-API namespace and device of
        the returned weights. If None, NumPy is used.

        .. versionadded:: 0.12

    Returns
    -------
    weights : ndarray, shape (n_weights,)
        Output checked weights.

    Notes
    -----
    .. versionadded:: 0.4
    .. versionchanged:: 0.12
        Add support for NumPy and PyTorch.
    """
    xp, dev = check_like(like)

    if weights is None:
        weights = xp.ones(n_weights, dtype=float, device=dev)

    else:
        weights = xp.asarray(weights, device=dev)
        if weights.shape != (n_weights,):
            raise ValueError(
                "Weights do not have the good shape. Should be (%d,) but got "
                "%s." % (n_weights, weights.shape,)
            )
        if check_positivity and bool(xp.any(weights <= 0)):
            raise ValueError("Weights must be strictly positive.")

    weights = weights / xp.sum(weights)
    return weights


def check_function(fun, functions):
    """Check the function to use.

    Parameters
    ----------
    fun : string | callable
        Function to check.
        If string, it must be one of the keys of ``functions``.
        If callable, it can be a function defined in API or by the user.
        In the latter case, the signature of the function as to match the ones
        defined in ``functions``. This is the user responsibility to ensure
        this, and will not be checked.
    functions : dict
        Functions available in API, used only when ``fun`` is a string.

    Returns
    -------
    fun : callable
        Function to use.

    Notes
    -----
    .. versionadded:: 0.6
    """
    if isinstance(fun, str):
        if fun not in functions.keys():
            raise ValueError(f"Unknown function name '{fun}'. Must be one of "
                             f"{' '.join(functions.keys())}")
        else:
            fun = functions[fun]
    elif not hasattr(fun, '__call__'):
        raise ValueError("Argument must be a string or a callable "
                         f"(Got {type(fun)}).")
    return fun


def check_init(init, n, *, like=None):
    """Check the initial matrix.

    Parameters
    ----------
    init : ndarray, shape (n, n)
        A square matrix used to initialize the algorithm.
    n : int
        Expected dimension of the matrix.
    like : None | ndarray, default=None
        Reference array used to infer the array-API namespace and device of
        the returned matrix. If None, NumPy is used.

        .. versionadded:: 0.12

    Returns
    -------
    init : ndarray, shape (n, n)
        The checked square matrix used to initialize the algorithm.

    Notes
    -----
    .. versionadded:: 0.8
    .. versionchanged:: 0.12
        Add support for NumPy and PyTorch.
    """
    xp, dev = check_like(like)
    init = xp.asarray(init, dtype=init.dtype, device=dev)
    if init.shape != (n, n):
        raise ValueError(
            "Init matrix does not have the good shape. "
            f"Should be ({n},{n}) but got {init.shape}."
        )
    return init


# ---------------------------------------------------------------------------
# Matrix property predicates
# ---------------------------------------------------------------------------

def is_square(X):
    """Check if matrices are square.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : bool
        True if matrices are square.
    """
    return X.ndim >= 2 and X.shape[-2] == X.shape[-1]


def is_real_type(X):
    """Check if matrices are real type.

    Parameters
    ----------
    X : ndarray, shape (..., n, m)
        The set of matrices.

    Returns
    -------
    ret : bool
        True if matrices are real type.

    Notes
    -----
    .. versionadded:: 0.6
    """
    xp = get_namespace(X)
    if is_torch_namespace(xp):
        return not X.dtype.is_complex
    return np.isrealobj(X)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

class deprecated(object):
    """Mark a function or class as deprecated (decorator).

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses::

        >>> from pyriemann.utils import deprecated
        >>> deprecated()
        <pyriemann.geometry._helpers.deprecated object at ...>
        >>> @deprecated()
        ... def some_function(): pass


    Parameters
    ----------
    extra: string
        To be added to the deprecation messages.
    """

    # Borrowed from MNE:
    # https://mne.tools/stable/generated/mne.utils.deprecated.html

    def __init__(self, extra=""):
        self.extra = extra

    def __call__(self, obj):
        """Call.
        Parameters
        ----------
        obj : object
            Object to call.
        """
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def deprecation_wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)

        cls.__init__ = deprecation_wrapped

        deprecation_wrapped.__name__ = "__init__"
        deprecation_wrapped.__doc__ = self._update_doc(init.__doc__)
        deprecation_wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun."""
        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        def deprecation_wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        deprecation_wrapped.__name__ = fun.__name__
        deprecation_wrapped.__dict__ = fun.__dict__
        deprecation_wrapped.__doc__ = self._update_doc(fun.__doc__)

        return deprecation_wrapped

    def _update_doc(self, olddoc):
        newdoc = ".. warning:: DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            # Get the spacing right to avoid sphinx warnings
            n_space = 4
            for li, line in enumerate(olddoc.split("\n")):
                if li > 0 and len(line.strip()):
                    n_space = len(line) - len(line.lstrip())
                    break
            newdoc = "%s\n\n%s%s" % (newdoc, " " * n_space, olddoc)

        return newdoc
