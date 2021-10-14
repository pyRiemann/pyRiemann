
import numpy as np


def generate_cov(n_trials, n_channels, rs, return_params=False):
    """Generate a set of covariances matrices for test purpose"""
    diags = 2.0 + 0.1 * rs.randn(n_trials, n_channels)
    A = 2 * rs.rand(n_channels, n_channels) - 1
    A /= np.linalg.norm(A, axis=1)[:, np.newaxis]
    covmats = np.empty((n_trials, n_channels, n_channels))
    for i in range(n_trials):
        covmats[i] = A @ np.diag(diags[i]) @ A.T
    if return_params:
        return covmats, diags, A
    else:
        return covmats


def is_square(X):
    """Check if matrices are square.

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


def is_symmetric(X):
    """Check if all matrices are symmetric.

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


def is_positive_definite(X):
    """Check if all matrices are positive definite.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : boolean
        True if all matrices are positive definite.
    """
    cs = X.shape[-1]
    return is_square(X) and \
        np.all(np.linalg.eigvals(X.reshape((-1, cs, cs))) > 0.0)


def is_positive_semi_definite(X):
    """Check if all matrices are positive semi-definite.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : boolean
        True if all matrices are positive semi-definite.
    """
    cs = X.shape[-1]
    return is_square(X) and \
        np.all(np.linalg.eigvals(X.reshape((-1, cs, cs))) >= 0.0)


def is_spd(X):
    """Check if all matrices are symmetric positive-definite.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : boolean
        True if all matrices are symmetric positive-definite.
    """

    return is_symmetric(X) and is_positive_definite(X)


def is_spsd(X):
    """Check if all matrices are symmetric positive semi-definite.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : boolean
        True if all matrices are symmetric positive semi-definite.
    """

    return is_symmetric(X) and is_positive_semi_definite(X)
