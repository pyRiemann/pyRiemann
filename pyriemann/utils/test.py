
import numpy as np


def _get_eigenvals(X):
    """ Private function to compute eigen values. """
    n = X.shape[-1]
    return np.linalg.eigvals(X.reshape((-1, n, n)))


def is_square(X):
    """ Check if matrices are square.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : boolean
        True if matrices are square.
    """
    return X.ndim >= 2 and X.shape[-2] == X.shape[-1]


def is_sym(X):
    """ Check if all matrices are symmetric.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : boolean
        True if all matrices are symmetric.
    """
    return is_square(X) and np.allclose(X, np.swapaxes(X, -2, -1))


def is_pos_def(X):
    """ Check if all matrices are positive definite.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : boolean
        True if all matrices are positive definite.
    """
    return is_square(X) and np.all(_get_eigenvals(X) > 0.0)


def is_pos_semi_def(X):
    """ Check if all matrices are positive semi-definite.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : boolean
        True if all matrices are positive semi-definite.
    """
    return is_square(X) and np.all(_get_eigenvals(X) >= 0.0)


def is_sym_pos_def(X):
    """ Check if all matrices are symmetric positive-definite.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : boolean
        True if all matrices are symmetric positive-definite.
    """
    return is_sym(X) and is_pos_def(X)


def is_sym_pos_semi_def(X):
    """ Check if all matrices are symmetric positive semi-definite.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : boolean
        True if all matrices are symmetric positive semi-definite.
    """
    return is_sym(X) and is_pos_semi_def(X)
