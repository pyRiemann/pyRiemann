from array_api_compat import array_namespace as get_namespace
import numpy as np

from ._helpers import is_real_type, is_square  # noqa: F401


def _allclose(A, B):
    """Array-API equivalent of ``numpy.allclose``."""
    xp = get_namespace(A, B)
    return bool(xp.all(xp.isclose(A, B)))


def _get_eigenvals(X):
    """Real part of eigenvalues for the trailing matrix dimension.

    ``xp.linalg.eigvals`` always returns complex dtype on torch (even for
    real inputs), and complex tensors cannot be compared against a float
    tolerance — so the real part is taken here once for all callers.
    """
    xp = get_namespace(X)
    n = X.shape[-1]
    return xp.real(xp.linalg.eigvals(X.reshape((-1, n, n))))


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
    return is_square(X) and _allclose(X, X.mT)


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
    return is_square(X) and _allclose(X, -X.mT)


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
                if bool((X[i, j] != X[i + j, 0]).item()):
                    return False
            else:
                if bool((X[i, j] != X[i + j - n + 1, n - 1]).item()):
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
    if is_real_type(X):
        return True
    xp = get_namespace(X)
    X_imag = xp.imag(X)
    return _allclose(X_imag, xp.zeros_like(X_imag))


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
    if is_real_type(X):
        return is_sym(X)
    xp = get_namespace(X)
    return is_sym(xp.real(X)) and is_skew_sym(xp.imag(X))


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
    xp = get_namespace(X)
    if fast_mode:
        try:
            xp.linalg.cholesky(X)
            return True
        except (np.linalg.LinAlgError, RuntimeError):
            return False
    else:
        if not is_square(X):
            return False
        return bool(xp.all(_get_eigenvals(X) > tol))


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
    xp = get_namespace(X)
    if not is_square(X):
        return False
    return bool(xp.all(_get_eigenvals(X) >= 0.0))


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
