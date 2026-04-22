"""Base functions for SPD/HPD matrices."""

from functools import wraps

import numpy as np


def ctranspose(X):
    """Conjugate transpose operator.

    Conjugate transpose operator for complex-valued array,
    giving transpose operator for real-valued array.

    Parameters
    ----------
    X : ndarray, shape (..., n, m)
        Matrices.

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


def _vectorize_nd(n_axes=2):
    """Decorator to vectorize a function over leading batch dimensions.

    Parameters
    ----------
    n_axes : int, default=2
        Number of trailing axes that form the core dimensions:

        - n_axes=2: (..., n1, n2) -> func(n1, n2) -> (..., m1, m2)
        - n_axes=3: (..., n1, n2, n3) -> func(n1, n2, n3) -> (..., m1, m2)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(X, *args, **kwargs):
            batch_shape = X.shape[:-n_axes]
            if len(batch_shape) == 0:
                return func(X, *args, **kwargs)
            n_batch = np.prod(batch_shape, dtype=int)
            core_shape = X.shape[-n_axes:]
            X_flat = X.reshape(n_batch, *core_shape)
            X_new = []
            for b in range(n_batch):
                X_new.append(
                    np.atleast_2d(func(X_flat[b], *args, **kwargs))
                )
            X_new = np.asarray(X_new)
            return X_new.reshape(*batch_shape, *X_new.shape[1:])
        return wrapper
    return decorator


###############################################################################


def _recursive(fun, A, B, *args, **kwargs):
    """Recursive function with two inputs."""
    if A.ndim == 2:
        return fun(A, B, *args, **kwargs)
    else:
        return np.asarray(
            [_recursive(fun, a, b, *args, **kwargs) for a, b in zip(A, B)]
        )


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

    if X.shape[-1] == 2:
        # Fast computation for 2x2 matrices
        # Start by computing the eigenvalues using a
        # stable closed-form solution
        a, b, c = X[..., 0, 0], X[..., 0, 1], X[..., 1, 1]
        trace = a + c
        det = a * c - np.abs(b)**2

        disc = np.sqrt((a - c)**2 + 4 * np.abs(b)**2)
        lam1 = (trace + disc) / 2
        # Stable small eigenvalue: uses lam1*lam2 = det to avoid cancellation
        # in (trace - disc) when trace ≈ disc (near-singular matrices).
        with np.errstate(invalid="ignore", divide="ignore"):
            lam2 = np.where(disc > 0, det / lam1, lam1)
        eigvals = np.array([lam1, lam2])

        # Apply the operator to the eigenvalues,
        # handling degeneracy (lam1 ≈ lam2)
        diff = eigvals[0] - eigvals[1]
        degenerate = np.isclose(eigvals[0], eigvals[1])
        with np.errstate(invalid="ignore", divide="ignore"):
            alpha_1 = (operator(eigvals[0]) - operator(eigvals[1])) / diff
            alpha_2 = (
                eigvals[0] * operator(eigvals[1])
                - eigvals[1] * operator(eigvals[0])
            ) / diff
        alpha_1 = np.where(degenerate, 0, alpha_1)
        alpha_2 = np.where(degenerate, operator(eigvals[0]), alpha_2)
        alpha_1 = np.asarray(alpha_1)[..., np.newaxis, np.newaxis]
        alpha_2 = np.asarray(alpha_2)[..., np.newaxis, np.newaxis]
        X_new = alpha_1 * X + alpha_2 * np.eye(2)
    else:
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
        SPD/HPD matrices.

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
        SPD/HPD matrices.

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
        SPD/HPD matrices.

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
        SPD/HPD matrices.
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
        SPD/HPD matrices.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix square root of C.
    """
    return _matrix_operator(C, np.sqrt)


###############################################################################


def nearest_sym_pos_def(X, reg=1e-6):
    """Find the nearest SPD matrices.

    A NumPy port of John D'Errico's ``nearestSPD`` MATLAB code [1]_,
    which credits [2]_.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Square matrices.
    reg : float, default=1e-6
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
    n = X.shape[-1]

    # Symmetrize
    A = (X + np.swapaxes(X, -2, -1)) / 2

    _, s, Vh = np.linalg.svd(A)
    H = np.swapaxes(Vh, -2, -1) @ (s[..., :, np.newaxis] * Vh)
    B = (A + H) / 2
    P = (B + np.swapaxes(B, -2, -1)) / 2

    # PD fix: iteratively shift non-PD matrices
    eigvals = np.linalg.eigvalsh(P)
    neg_ev = np.any(eigvals <= 0, axis=-1)  # (...,)

    if np.any(neg_ev):
        spacing = np.spacing(np.linalg.norm(A, axis=(-2, -1)))
        I = np.eye(n)  # noqa
        k = 1
        while np.any(neg_ev) and k < 100:
            mineig = np.min(np.linalg.eigvalsh(P), axis=-1)
            shift = np.where(neg_ev, -mineig * k**2 + spacing, 0.0)
            P = P + shift[..., np.newaxis, np.newaxis] * I
            eigvals = np.linalg.eigvalsh(P)
            neg_ev = np.any(eigvals <= 0, axis=-1)
            k += 1

    # Regularize
    ei, ev = np.linalg.eigh(P)
    ratio = np.min(ei, axis=-1) / np.max(ei, axis=-1)
    needs_reg = ratio < reg  # (...,)
    if np.any(needs_reg):
        ei_reg = ei + reg
        P_reg = ev @ (ei_reg[..., :, np.newaxis] * np.swapaxes(ev, -2, -1))
        P = np.where(needs_reg[..., np.newaxis, np.newaxis], P_reg, P)

    return P


###############################################################################


def _first_divided_difference(d, fct, fctder, atol=1e-12, rtol=1e-12):
    r"""First divided difference of a matrix function.

    The first divided difference of a matrix function applied to the
    eigenvalues :math:`\mathbf{d}` of a symmetric matrix is [1]_:

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
    d : ndarray, shape (..., n)
        Eigenvalues of symmetric matrices.
    fct : callable
        Function to apply to eigenvalues d. Has to be defined for all
        possible eigenvalues d.
    fctder : callable
        Derivative of the function to apply. Has to be defined for all
        possible eigenvalues d.
    atol : float, default=1e-12
        Absolute tolerance for equality of eigenvalues.
    rtol : float, default=1e-12
        Relative tolerance for equality of eigenvalues.

    Returns
    -------
    FDD : ndarray, shape (..., n, n)
        First divided difference of the function applied to the eigenvalues.

    Notes
    -----
    .. versionadded:: 0.8

    References
    ----------
    .. [1] `Matrix  Analysis <https://doi.org/10.1007/978-1-4612-0653-8>`_
        R. Bhatia, Springer, 1997
    """
    di = d[..., :, np.newaxis]
    dj = d[..., np.newaxis, :]

    close_ = np.isclose(di, dj, atol=atol, rtol=rtol)
    safe_diff = np.where(close_, np.ones_like(di - dj), di - dj)
    return np.where(close_, fctder(di), (fct(di) - fct(dj)) / safe_diff)


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
    Vh = ctranspose(V)
    expfdd = _first_divided_difference(d, np.exp, np.exp)
    return V @ (expfdd * (Vh @ X @ V)) @ Vh


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
    Vh = ctranspose(V)
    logfdd = _first_divided_difference(d, np.log, lambda x: 1 / x)
    return V @ (logfdd * (Vh @ X @ V)) @ Vh
