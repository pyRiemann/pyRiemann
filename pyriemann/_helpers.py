"""Shared helper functions for pyriemann.

This module contains validation and utility functions used by both
``pyriemann.geometry`` and ``pyriemann.utils``. It has no internal
pyriemann dependencies (only numpy/warnings/inspect), making it a safe
leaf in the import graph.
"""

import inspect
import warnings

import numpy as np


def check_weights(weights, n_weights, *, check_positivity=False):
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

    Returns
    -------
    weights : ndarray, shape (n_weights,)
        Output checked weights.

    Notes
    -----
    .. versionadded:: 0.4
    """
    if weights is None:
        weights = np.ones(n_weights)

    else:
        weights = np.asarray(weights)
        if weights.shape != (n_weights,):
            raise ValueError(
                "Weights do not have the good shape. Should be (%d,) but got "
                "%s." % (n_weights, weights.shape,)
            )
        if check_positivity and any(weights <= 0):
            raise ValueError("Weights must be strictly positive.")

    weights /= np.sum(weights)
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
    elif not callable(fun):
        raise ValueError("Argument must be a string or a callable "
                         f"(Got {type(fun)}).")
    return fun


def check_init(init, n):
    """Check the initial matrix.

    Parameters
    ----------
    init : ndarray, shape (n, n)
        A square matrix used to initialize the algorithm.
    n : int
        Expected dimension of the matrix.

    Returns
    -------
    init : ndarray, shape (n, n)
        The checked square matrix used to initialize the algorithm.

    Notes
    -----
    .. versionadded:: 0.8
    """
    init = np.asarray(init)
    if init.shape != (n, n):
        raise ValueError(
            "Init matrix does not have the good shape. "
            f"Should be ({n},{n}) but got {init.shape}."
        )
    return init


def check_param_in_func(param, func):
    """Check if a parameter is an argument of a function.

    Parameters
    ----------
    param : str
        Name of the parameter to check.
    func : callable
        Function to check.

    Returns
    -------
    ret : bool
        True if param is an argument of a function, else False.

    Notes
    -----
    .. versionadded:: 0.11
    """
    sig = inspect.signature(func).parameters
    return (param in sig)


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
    return np.isrealobj(X)


class deprecated(object):
    """Mark a function or class as deprecated (decorator).

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses::

        >>> from pyriemann._helpers import deprecated
        >>> deprecated()
        <pyriemann._helpers.deprecated object at ...>
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
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
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
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
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
