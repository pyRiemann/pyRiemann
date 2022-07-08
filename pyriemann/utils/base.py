"""Base functions for SPD matrices."""

import numpy as np
from scipy.linalg import eigh

from numpy.core.numerictypes import typecodes


def _matrix_operator(C, operator):
    """Matrix function."""
    if C.dtype.char in typecodes['AllFloat'] and not np.isfinite(C).all():
        raise ValueError(
            "Matrices must be positive definite. Add "
            "regularization to avoid this error.")
    eigvals, eigvects = eigh(C, check_finite=False)
    D = (eigvects * operator(eigvals)[np.newaxis, :]) @ eigvects.T
    return D


def _matrices_operator(C, operator):
    """Recursive matrix function."""
    if not isinstance(C, np.ndarray) or C.ndim < 2:
        raise ValueError('Input must be at least a 2D ndarray')
    elif C.ndim == 2:
        return _matrix_operator(C, operator)
    else:
        return np.asarray([_matrices_operator(c, operator) for c in C])


def sqrtm(C):
    r"""Square root of SPD matrices.

    The matrix square root of a SPD matrix is defined by:

    .. math::
        \mathbf{D} =
        \mathbf{V} \left( \mathbf{\Lambda} \right)^{1/2} \mathbf{V}^\top

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        SPD matrices, at least 2D ndarray.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix square root of C.
    """
    return _matrices_operator(C, np.sqrt)


def logm(C):
    r"""Logarithm of SPD matrices.

    The matrix logarithm of a SPD matrix is defined by:

    .. math::
        \mathbf{D} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^\top

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        SPD matrices, at least 2D ndarray.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix logarithm of C.
    """
    return _matrices_operator(C, np.log)


def expm(C):
    r"""Exponential of SPD matrices.

    The matrix exponential of a SPD matrix is defined by:

    .. math::
        \mathbf{D} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^\top

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        SPD matrices, at least 2D ndarray.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix exponential of C.
    """
    return _matrices_operator(C, np.exp)


def invsqrtm(C):
    r"""Inverse square root of SPD matrices.

    The matrix inverse square root of a SPD matrix is defined by:

    .. math::
        \mathbf{D} =
        \mathbf{V} \left( \mathbf{\Lambda} \right)^{-1/2} \mathbf{V}^\top

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        SPD matrices, at least 2D ndarray.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix inverse square root of C.
    """
    def isqrt(x): return 1. / np.sqrt(x)
    return _matrices_operator(C, isqrt)


def powm(C, alpha):
    r"""Power of SPD matrices.

    The matrix power :math:`\alpha` of a SPD matrix is defined by:

    .. math::
        \mathbf{D} =
        \mathbf{V} \left( \mathbf{\Lambda} \right)^{\alpha} \mathbf{V}^\top

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        SPD matrices, at least 2D ndarray.
    alpha : float
        The power to apply.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix power of C.
    """
    def power(x): return x**alpha
    return _matrices_operator(C, power)
