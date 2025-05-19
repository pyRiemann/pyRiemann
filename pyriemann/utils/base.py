"""Base functions for SPD/HPD matrices."""

import numpy as np

from .test import is_pos_def


def ctranspose(X):
    """Conjugate transpose operator.

    Conjugate transpose operator for complex-valued array,
    giving transpose operator for real-valued array.

    Parameters
    ----------
    X : ndarray, shape (..., n, m)
        Matrices, at least 2D ndarray.

    Returns
    -------
    X_new : ndarray, shape (..., m, n)
        Conjugate transpose of X.

    Notes
    -----
    .. versionadded:: 0.9

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Conjugate_transpose
    """
    return np.swapaxes(X.conj(), -2, -1)


###############################################################################


def _matrix_operator(X, operator):
    """Matrix function for SPD/HPD matrices."""
    if not isinstance(X, np.ndarray) or X.ndim < 2:
        raise ValueError("Input must be at least a 2D ndarray")
    if X.dtype.char in np.typecodes['AllFloat'] and (
            np.isinf(X).any() or np.isnan(X).any()):
        raise ValueError(
            "Matrices must be positive definite. "
            "You should add regularization to avoid this error."
        )

    eigvals, eigvecs = np.linalg.eigh(X)
    eigvals = operator(eigvals)
    X_new = eigvecs @ (np.expand_dims(eigvals, -1) * ctranspose(eigvecs))
    return X_new


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
    reg : float, default=1e-6
        Regularization parameter.

    Returns
    -------
    P : ndarray, shape (n, n)
        Nearest SPD matrix.
    """
    def regularize(X, reg):
        ei, ev = np.linalg.eigh(X)
        if np.min(ei) / np.max(ei) < reg:
            X = ev @ np.diag(ei + reg) @ ev.T
        return X

    A = (S + S.T) / 2
    _, s, V = np.linalg.svd(A)
    H = V.T @ (s[:, np.newaxis] * V)
    B = (A + H) / 2
    P = (B + B.T) / 2

    if is_pos_def(P):
        # Regularize if already PD
        return regularize(P, reg)

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(S.shape[0])  # noqa
    k = 1
    while not is_pos_def(P, fast_mode=False):
        mineig = np.min(np.real(np.linalg.eigvals(P)))
        P += I * (-mineig * k ** 2 + spacing)
        k += 1

    # Regularize
    return regularize(P, reg)


def nearest_sym_pos_def(X, reg=1e-6):
    """Find the nearest SPD matrices.

    A NumPy port of John D'Errico's ``nearestSPD`` MATLAB code [1]_,
    which credits [2]_.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Square matrices.
    reg : float, default=1e-6
        Regularization parameter.

    Returns
    -------
    P : ndarray, shape (n_matrices, n, n)
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


###############################################################################


def _first_divided_difference(d, fct, fctder, atol=1e-12, rtol=1e-12):
    r"""First divided difference of a matrix function.

    First divided difference of a matrix function applied to the eigenvalues
    of a symmetric matrix. The first divided difference is defined as [1]_:

    .. math::
       [FDD(d)]_{i,j} =
           \begin{cases}
           \frac{fct(d_i)-fct(d_j)}{d_i-d_j},
           & d_i \neq d_j\\
           fctder(d_i),
           & d_i = d_j
           \end{cases}


    Parameters
    ----------
    d : ndarray, shape (n,)
        Eigenvalues of a symmetric matrix.
    fct : callable
        Function to apply to eigenvalues of d. Has to be defined for all
        possible eigenvalues of d.
    fctder : callable
        Derivative of the function to apply. Has to be defined for all
        possible eigenvalues of d.
    atol : float, default=1e-12
        Absolute tolerance for equality of eigenvalues.
    rtol : float, default=1e-12
        Relative tolerance for equality of eigenvalues.

    Returns
    -------
    FDD : ndarray, shape (n, n)
        First divided difference of the function applied to the eigenvalues.

    Notes
    -----
    .. versionadded:: 0.8

    References
    ----------
    .. [1] `Matrix  Analysis <https://doi.org/10.1007/978-1-4612-0653-8>`_
        R. Bhatia, Springer, 1997
    """
    dif = np.repeat(d[None, :], len(d), axis=0)

    close_ = np.isclose(dif, dif.T, atol=atol, rtol=rtol)
    dif[close_] = fctder(dif[close_])
    dif[~close_] = (fct(dif[~close_]) - fct(dif.T[~close_])) / \
                   (dif[~close_] - dif.T[~close_])
    return dif


def ddexpm(X, Cref):
    r"""Directional derivative of the matrix exponential.

    The directional derivative of the matrix exponential at a SPD/HPD matrix
    :math:`\mathbf{C}_{\text{ref}}` in the direction of a SPD/HPD matrix
    :math:`\mathbf{X}` is defined as Eq. (V.13) in [1]_:

    .. math::
        \text{ddexpm}(\mathbf{X}, \mathbf{C}_{\text{ref}}) =
        \mathbf{V} \left( \text{fddexpm}(\mathbf{\Lambda}) \odot
        \mathbf{V}^H \mathbf{X} \mathbf{V} \right) \mathbf{V}^H

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues of
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}_{\text{ref}}`,
    and :math:`\text{fddexpm}` the first divided difference of the exponential
    function.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        SPD/HPD matrices.
    Cref : ndarray, shape (n, n)
        SPD/HPD matrix.

    Returns
    -------
    ddexpm : ndarray, shape (..., n, n)
        Directional derivative of the matrix exponential.

    Notes
    -----
    .. versionadded:: 0.8

    References
    ----------
    .. [1] `Matrix  Analysis <https://doi.org/10.1007/978-1-4612-0653-8>`_
        R. Bhatia, Springer, 1997
    """

    d, V = np.linalg.eigh(Cref)
    expfdd = _first_divided_difference(d, np.exp, np.exp)
    return V @ (expfdd * (V.conj().T @ X @ V)) @ V.conj().T


def ddlogm(X, Cref):
    r"""Directional derivative of the matrix logarithm.

    The directional derivative of the matrix logarithm at a SPD/HPD matrix
    :math:`\mathbf{C}_{\text{ref}}` in the direction of a SPD/HPD matrix
    :math:`\mathbf{X}` is defined as Eq. (V.13) in [1]_:

    .. math::
        \text{ddlogm}(\mathbf{X}, \mathbf{C}_{\text{ref}}) =
        \mathbf{V} \left( \text{fddlogm}(\mathbf{\Lambda}) \odot
        \mathbf{V}^H \mathbf{X} \mathbf{V} \right) \mathbf{V}^H

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues of
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}_{\text{ref}}`,
    and :math:`\text{fddlogm}` the first divided difference of the logarithm
    function.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        SPD/HPD matrices.
    Cref : ndarray, shape (n, n)
        SPD/HPD matrix.

    Returns
    -------
    ddlogm : ndarray, shape (..., n, n)
        Directional derivative of the matrix logarithm.

    Notes
    -----
    .. versionadded:: 0.8

    References
    ----------
    .. [1] `Matrix  Analysis <https://doi.org/10.1007/978-1-4612-0653-8>`_
        R. Bhatia, Springer, 1997
    """

    d, V = np.linalg.eigh(Cref)
    logfdd = _first_divided_difference(d, np.log, lambda x: 1 / x)
    return V @ (logfdd * (V.conj().T @ X @ V)) @ V.conj().T
