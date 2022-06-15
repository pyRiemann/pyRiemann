import numpy as np
from scipy.linalg import eigh

from numpy.core.numerictypes import typecodes


def _matrix_operator(C, operator):
    """Matrix equivalent of an operator."""
    if C.dtype.char in typecodes['AllFloat'] and not np.isfinite(C).all():
        raise ValueError(
            "Covariance matrices must be positive definite. Add "
            "regularization to avoid this error.")
    eigvals, eigvects = eigh(C, check_finite=False)
    eigvals = np.diag(operator(eigvals))
    D = eigvects @ eigvals @ eigvects.T
    return D


def sqrtm(C):
    r""" Square root of SPD matrix.

    Return the matrix square root of a SPD matrix defined by:

    .. math::
        \mathbf{D} =
        \mathbf{V} \left( \mathbf{\Lambda} \right)^{1/2} \mathbf{V}^\top

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (n, n)
        SPD matrix.

    Returns
    -------
    D : ndarray, shape (n, n)
        Matrix square root of C.
    """
    return _matrix_operator(C, np.sqrt)


def logm(C):
    r""" Logarithm of SPD matrix.

    Return the matrix logarithm of a SPD matrix defined by:

    .. math::
        \mathbf{D} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^\top

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (n, n)
        SPD matrix.

    Returns
    -------
    D : ndarray, shape (n, n)
        Matrix logarithm of C.
    """
    return _matrix_operator(C, np.log)


def expm(C):
    r""" Exponential of SPD matrix.

    Return the matrix exponential of a SPD matrix defined by:

    .. math::
        \mathbf{D} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^\top

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (n, n)
        SPD matrix.

    Returns
    -------
    D : ndarray, shape (n, n)
        Matrix exponential of C.
    """
    return _matrix_operator(C, np.exp)


def invsqrtm(C):
    r""" Inverse square root of SPD matrix.

    Return the inverse matrix square root of a SPD matrix defined by:

    .. math::
        \mathbf{D} =
        \mathbf{V} \left( \mathbf{\Lambda} \right)^{-1/2} \mathbf{V}^\top

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (n, n)
        SPD matrix.

    Returns
    -------
    D : ndarray, shape (n, n)
        Inverse matrix square root of C.
    """
    def isqrt(x): return 1. / np.sqrt(x)
    return _matrix_operator(C, isqrt)


def powm(C, alpha):
    r""" Power of SPD matrix.

    Return the matrix power :math:`\alpha` of a SPD matrix defined by:

    .. math::
        \mathbf{D} =
        \mathbf{V} \left( \mathbf{\Lambda} \right)^{\alpha} \mathbf{V}^\top

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (n, n)
        SPD matrix.
    alpha : float
        The power to apply.

    Returns
    -------
    D : ndarray, shape (n, n)
        Matrix power of C.
    """
    def power(x): return x**alpha
    return _matrix_operator(C, power)
