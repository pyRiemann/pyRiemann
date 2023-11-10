"""Base functions for SPD/HPD matrices."""

import numpy as np
from numpy.core.numerictypes import typecodes
from .test import is_pos_def


def _matrix_operator(C, operator):
    """Matrix function."""
    if not isinstance(C, np.ndarray) or C.ndim < 2:
        raise ValueError("Input must be at least a 2D ndarray")
    if C.dtype.char in typecodes['AllFloat'] and (
            np.isinf(C).any() or np.isnan(C).any()):
        raise ValueError(
            "Matrices must be positive definite. Add "
            "regularization to avoid this error.")
    if type(C) is BlockMatrix:
        blocks = C._extract_blocks()
        D = _apply_operator(blocks, operator)
        del blocks
        res = BlockMatrix(np.zeros(C.shape), block_size=C.block_size)
        res._insert_blocks(D)
        return res

    else:
        return _apply_operator(C, operator)


def _apply_operator(C, operator):
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = operator(eigvals)
    if C.ndim >= 3:
        eigvals = np.expand_dims(eigvals, -2)
    D = (eigvecs * eigvals) @ np.swapaxes(eigvecs.conj(), -2, -1)
    return D


def expm(C):
    r"""Exponential of SPD/HPD matrices.

    The symmetric matrix exponential of a SPD/HPD matrix
    :math:`\mathbf{C}` is defined by:

    .. math::
        \mathbf{D} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^H

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        SPD/HPD matrices, at least 2D ndarray.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix exponential of C.
    """
    return _matrix_operator(C, np.exp)


def invsqrtm(C):
    r"""Inverse square root of SPD/HPD matrices.

    The symmetric matrix inverse square root of a SPD/HPD matrix
    :math:`\mathbf{C}` is defined by:

    .. math::
        \mathbf{D} =
        \mathbf{V} \left( \mathbf{\Lambda} \right)^{-1/2} \mathbf{V}^H

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        SPD/HPD matrices, at least 2D ndarray.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix inverse square root of C.
    """
    def isqrt(x): return 1. / np.sqrt(x)
    return _matrix_operator(C, isqrt)


def logm(C):
    r"""Logarithm of SPD/HPD matrices.

    The symmetric matrix logarithm of a SPD/HPD matrix
    :math:`\mathbf{C}` is defined by:

    .. math::
        \mathbf{D} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^H

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        SPD/HPD matrices, at least 2D ndarray.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix logarithm of C.
    """
    return _matrix_operator(C, np.log)


def powm(C, alpha):
    r"""Power of SPD/HPD matrices.

    The symmetric matrix power :math:`\alpha` of a SPD/HPD matrix
    :math:`\mathbf{C}` is defined by:

    .. math::
        \mathbf{D} =
        \mathbf{V} \left( \mathbf{\Lambda} \right)^{\alpha} \mathbf{V}^H

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        SPD/HPD matrices, at least 2D ndarray.
    alpha : float
        The power to apply.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix power of C.
    """
    def power(x): return x**alpha
    return _matrix_operator(C, power)


def sqrtm(C):
    r"""Square root of SPD/HPD matrices.

    The symmetric matrix square root of a SPD/HPD matrix
    :math:`\mathbf{C}` is defined by:

    .. math::
        \mathbf{D} =
        \mathbf{V} \left( \mathbf{\Lambda} \right)^{1/2} \mathbf{V}^H

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        SPD/HPD matrices, at least 2D ndarray.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix square root of C.
    """
    return _matrix_operator(C, np.sqrt)


###############################################################################


def _nearest_sym_pos_def(S, reg=1e-6):
    """Find the nearest SPD matrix.

    Parameters
    ----------
    S : ndarray, shape (n, n)
        Square matrix.
    reg : float
        Regularization parameter.

    Returns
    -------
    P : ndarray, shape (n, n)
        Nearest SPD matrix.
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


def nearest_sym_pos_def(X, reg=1e-6):
    """Find the nearest SPD matrices.

    A NumPy port of John D'Errico's `nearestSPD` MATLAB code [1]_,
    which credits [2]_.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Square matrices, at least 2D ndarray.
    reg : float
        Regularization parameter.

    Returns
    -------
    P : ndarray, shape (..., n, n)
        Nearest SPD matrices.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] `nearestSPD
        <https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd>`_
        J. D'Errico, MATLAB Central File Exchange
    .. [2] `Computing a nearest symmetric positive semidefinite matrix
        <https://www.sciencedirect.com/science/article/pii/0024379588902236>`_
        N.J. Higham, Linear Algebra and its Applications, vol 103, 1988
    """
    return np.array([_nearest_sym_pos_def(x, reg) for x in X])


class BlockMatrix(np.ndarray):
    def __new__(cls, input_array, block_size=-1):
        obj = np.asarray(input_array).view(cls)
        if block_size == -1:
            block_size = obj.shape[-1]
        obj.block_size = block_size
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("Parameter block_size must be a positive "
                             "integer or -1.")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        if len(obj.shape) == 0:
            return
        self.block_size = getattr(obj, 'block_size', obj.shape[-1])

    def _extract_blocks(self):

        if self.shape[-1] % self.block_size != 0:
            raise ValueError(
                f"Number of channels ({self.shape[-1]}) "
                f"must be a multiple of n_blocks ({self.block_size}).")

        n_blocks = self.shape[-1] // self.block_size
        new_shape = (*(self.shape[:-2]),
                     n_blocks,
                     self.block_size,
                     self.block_size)

        new_strides = (*(self.strides[:-2]),
                       self.block_size * self.strides[-2] + self.block_size *
                       self.strides[-1],
                       self.strides[-2], self.strides[-1])

        view = np.lib.stride_tricks.as_strided(self, new_shape,
                                               new_strides)
        return view

    def _insert_blocks(self, blocks):
        block_view = self._extract_blocks()
        block_view[:] = blocks
