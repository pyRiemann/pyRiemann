"""Base functions for SPD matrices."""

import numpy as np
from numpy.core.numerictypes import typecodes
from .test import is_pos_def


def _matrix_operator(C, operator):
    """Matrix function."""
    if not isinstance(C, np.ndarray) or C.ndim < 2:
        raise ValueError('Input must be at least a 2D ndarray')
    if C.dtype.char in typecodes['AllFloat'] and (
            np.isinf(C).any() or np.isnan(C).any()):
        raise ValueError(
            "Matrices must be positive definite. Add "
            "regularization to avoid this error.")
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = operator(eigvals)
    if C.ndim >= 3:
        eigvals = np.expand_dims(eigvals, -2)
    D = (eigvecs * eigvals) @ np.swapaxes(eigvecs, -2, -1)
    return D


def sqrtm(C):
    r"""Square root of SPD matrices.

    The matrix square root of a SPD matrix C is defined by:

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
    return _matrix_operator(C, np.sqrt)


def logm(C):
    r"""Logarithm of SPD matrices.

    The matrix logarithm of a SPD matrix C is defined by:

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
    return _matrix_operator(C, np.log)


def expm(C):
    r"""Exponential of SPD matrices.

    The matrix exponential of a SPD matrix C is defined by:

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
    return _matrix_operator(C, np.exp)


def invsqrtm(C):
    r"""Inverse square root of SPD matrices.

    The matrix inverse square root of a SPD matrix C is defined by:

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
    return _matrix_operator(C, isqrt)


def powm(C, alpha):
    r"""Power of SPD matrices.

    The matrix power :math:`\alpha` of a SPD matrix C is defined by:

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
    return _matrix_operator(C, power)


def _nearest_pos_def(S, reg=1e-6):
    """Find the nearest positive-definite matrix to input matrix

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1]_, which
    credits [2]_.

    Parameters
    ----------
    S : ndarray, shape (n, n)
        Square matrix.
    reg : float
        Regularization parameter

    Returns
    -------
    P : ndarray, shape (n, n)
        Nearest SPD matrix power of S

    References
    ----------
    .. [1] John D'Errico, `nearestSPD
       <https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd>`_
    .. [2]  N.J. Higham, `Computing a nearest symmetric positive semidefinite
       matrix <htttps://doi.org/10.1016/0024-3795(88)90223-6>`_ , 1988
    """
    A = (S + S.T) / 2
    _, s, V = np.linalg.svd(A)

    H = V.T @ np.diag(s) @ V

    B = (A + H) / 2

    P = (B + B.T) / 2

    if is_pos_def(P):
        # Regularize if already PD
        ei, ev = np.linalg.eigh(P)
        if np.min(ei) / np.max(ei) < reg:
            P = ev @ np.diag(ei + reg) @ ev.T
        return P

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(S.shape[0])  # noqa
    k = 1
    while not is_pos_def(P, fast_mode=False):
        mineig = np.min(np.real(np.linalg.eigvals(P)))
        P += I * (-mineig * k ** 2 + spacing)
        k += 1

    # Regularize
    ei, ev = np.linalg.eigh(P)
    if np.min(ei) / np.max(ei) < reg:
        P = ev @ np.diag(ei + reg) @ ev.T
    return P


def nearest_pos_def(X, reg=1e-6):
    """Find the nearest positive-definite matrices to input matrices.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Square matrices, at least 2D ndarray.
    reg : float
        Regularization parameter.

    Returns
    -------
    P : ndarray, shape (..., n, n)
        Nearest SPD matrices power of X.
    """
    return np.array([_nearest_pos_def(x, reg) for x in X])
