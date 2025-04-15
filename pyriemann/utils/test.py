import numpy as np


def _get_eigenvals(X):
    """Private function to compute all eigen values."""
    n = X.shape[-1]
    return np.linalg.eigvals(X.reshape((-1, n, n)))


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


def is_sym(X):
    """Check if all matrices are symmetric.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : bool
        True if all matrices are symmetric.
    """
    return is_square(X) and np.allclose(X, np.swapaxes(X, -2, -1))


def is_skew_sym(X):
    """Check if all matrices are skew-symmetric.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : bool
        True if all matrices are skew-symmetric.
    """
    return is_square(X) and np.allclose(X, -np.swapaxes(X, -2, -1))


def is_hankel(X):
    """Check if matrix is an Hankel matrix.

    Parameters
    ----------
    X : ndarray, shape (n, n)
        Square matrix.

    Returns
    -------
    ret : bool
        True if Hankel matrix.
    """
    if not is_square(X) or X.ndim != 2:
        return False
    n, _ = X.shape

    for i in range(n):
        for j in range(n):
            if (i + j < n):
                if (X[i, j] != X[i + j, 0]):
                    return False
            else:
                if (X[i, j] != X[i + j - n + 1, n - 1]):
                    return False

    return True


def is_real(X):
    """Check if all matrices are strictly real.

    Better management of numerical imprecisions than np.all(np.isreal()).

    Parameters
    ----------
    X : ndarray, shape (..., n, m)
        The set of matrices.

    Returns
    -------
    ret : bool
        True if all matrices are strictly real.
    """
    return np.allclose(X.imag, np.zeros_like(X.imag))


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


def is_hermitian(X):
    """Check if all matrices are Hermitian.

    Check if all matrices are Hermitian, ie with a symmetric real part and
    a skew-symmetric imaginary part.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : bool
        True if all matrices are Hermitian.
    """
    return is_sym(X.real) and is_skew_sym(X.imag)


def is_pos_def(X, tol=0.0, fast_mode=False):
    """Check if all matrices are positive definite (PD).

    Check if all matrices are positive definite, fast verification is done
    with Cholesky decomposition, while full check compute all eigenvalues
    to verify that they are positive.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.
    tol : float, default=0.0
        Threshold below which eigen values are considered zero.
    fast_mode : bool, default=False
        Use Cholesky decomposition to avoid computing all eigenvalues.

    Returns
    -------
    ret : bool
        True if all matrices are positive definite.
    """
    if fast_mode:
        try:
            np.linalg.cholesky(X)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return is_square(X) and np.all(_get_eigenvals(X) > tol)


def is_pos_semi_def(X):
    """Check if all matrices are positive semi-definite (PSD).

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : bool
        True if all matrices are positive semi-definite.
    """
    return is_square(X) and np.all(_get_eigenvals(X) >= 0.0)


def is_sym_pos_def(X, tol=0.0):
    """Check if all matrices are symmetric positive-definite (SPD).

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.
    tol : float, default=0.0
        Threshold below which eigen values are considered zero.

    Returns
    -------
    ret : bool
        True if all matrices are symmetric positive-definite.
    """
    return is_sym(X) and is_pos_def(X, tol=tol)


def is_sym_pos_semi_def(X):
    """Check if all matrices are symmetric positive semi-definite (SPSD).

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : bool
        True if all matrices are symmetric positive semi-definite.
    """
    return is_sym(X) and is_pos_semi_def(X)


def is_herm_pos_def(X, tol=0.0):
    """Check if all matrices are Hermitian positive-definite (HPD).

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.
    tol : float, default=0.0
        Threshold below which eigen values are considered zero.

    Returns
    -------
    ret : bool
        True if all matrices are Hermitian positive-definite.
    """
    return is_hermitian(X) and is_pos_def(X, tol=tol)


def is_herm_pos_semi_def(X):
    """Check if all matrices are Hermitian positive semi-definite (HPSD).

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : bool
        True if all matrices are Hermitian positive semi-definite.
    """
    return is_hermitian(X) and is_pos_semi_def(X)
