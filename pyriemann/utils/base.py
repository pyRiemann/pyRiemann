"""Base functions for SPD/HPD matrices."""

from functools import wraps

import numpy as np

from ._backend import get_namespace, is_numpy_namespace, xpd


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
    xp = get_namespace(X)
    return xp.conj(X).mT


def _vectorize_nd(n_axes=2, batch_native=True):
    """Decorator to vectorize a function over leading batch dimensions.

    Parameters
    ----------
    n_axes : int, default=2
        Number of trailing axes that form the core dimensions:

        - n_axes=2: (..., n1, n2) -> func(n1, n2) -> (..., m1, m2)
        - n_axes=3: (..., n1, n2, n3) -> func(n1, n2, n3) -> (..., m1, m2)
    batch_native : bool, default=True
        If True, non-numpy backends (e.g. torch) call the function
        directly — it must handle batch dims natively for optimal GPU
        parallelism and autograd support.
        If False, always loops over batch elements (for functions that
        cannot be made batch-aware, e.g. those using Python lists).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(X, *args, **kwargs):
            batch_shape = X.shape[:-n_axes]
            if len(batch_shape) == 0:
                return func(X, *args, **kwargs)
            xp = get_namespace(X)
            if batch_native and not is_numpy_namespace(xp):
                # Non-numpy backends broadcast natively
                return func(X, *args, **kwargs)
            # Loop over batch elements
            n_batch = int(np.prod(batch_shape))
            core_shape = X.shape[-n_axes:]
            X_flat = xp.reshape(X, (n_batch, *core_shape))
            X_new = xp.stack(
                [func(X_flat[b], *args, **kwargs) for b in range(n_batch)],
                axis=0,
            )
            return xp.reshape(X_new, (*batch_shape, *X_new.shape[1:]))
        return wrapper
    return decorator


###############################################################################


def _recursive(fun, A, B, *args, **kwargs):
    """Recursive function with two inputs."""
    if A.ndim == 2:
        return fun(A, B, *args, **kwargs)
    else:
        xp = get_namespace(A)
        return xp.stack(
            [_recursive(fun, a, b, *args, **kwargs) for a, b in zip(A, B)],
            axis=0,
        )


def _matrix_operator(X, operator):
    """Matrix function for SPD/HPD matrices."""
    xp = get_namespace(X)
    if not hasattr(X, "ndim") or X.ndim < 2:
        raise ValueError("Input must be at least a 2D ndarray")
    if X.shape[-2] != X.shape[-1]:
        raise ValueError("Input must contain square matrices")
    if xp.isdtype(X.dtype, ("real floating", "complex floating")) \
            and not bool(xp.all(xp.isfinite(X))):
        raise ValueError(
            "Matrices must be positive definite. "
            "You should add regularization to avoid this error."
        )

    eigvals, eigvecs = xp.linalg.eigh(X)
    eigvals = operator(eigvals)
    X_new = eigvecs @ (eigvals[..., None] * ctranspose(eigvecs))
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
    xp = get_namespace(C)
    return _matrix_operator(C, xp.exp)


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
    xp = get_namespace(C)
    def isqrt(x): return 1. / xp.sqrt(x)
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
    xp = get_namespace(C)
    return _matrix_operator(C, xp.log)


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
    xp = get_namespace(C)
    return _matrix_operator(C, xp.sqrt)


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
    xp = get_namespace(X)
    n = X.shape[-1]
    eps = xp.finfo(X.dtype).eps

    # Symmetrize
    A = (X + X.mT) / 2

    _, s, Vh = xp.linalg.svd(A)
    H = Vh.mT @ (s[..., :, xp.newaxis] * Vh)
    B = (A + H) / 2
    P = (B + B.mT) / 2

    # PD fix: iteratively shift non-PD matrices
    eigvals = xp.linalg.eigvalsh(P)
    neg_ev = xp.any(eigvals <= 0, axis=-1)  # (...,)

    if bool(xp.any(neg_ev)):
        # we don't have spacing numpy in torch,
        # so we re-implement.
        spacing = xp.abs(xp.linalg.matrix_norm(A)) * eps
        eye_n = xp.eye(n, dtype=X.dtype, device=xpd(X))
        k = 1
        while bool(xp.any(neg_ev)) and k < 100:
            mineig = xp.min(xp.linalg.eigvalsh(P), axis=-1)
            shift = xp.where(
                neg_ev, -mineig * k**2 + spacing, 0.0,
            )
            P = P + shift[..., xp.newaxis, xp.newaxis] * eye_n
            eigvals = xp.linalg.eigvalsh(P)
            neg_ev = xp.any(eigvals <= 0, axis=-1)
            k += 1

    # Regularize
    ei, ev = xp.linalg.eigh(P)
    ratio = xp.min(ei, axis=-1) / xp.max(ei, axis=-1)
    needs_reg = ratio < reg  # (...,)
    if bool(xp.any(needs_reg)):
        ei_reg = ei + reg
        P_reg = ev @ (ei_reg[..., :, xp.newaxis] * ev.mT)
        P = xp.where(
            needs_reg[..., xp.newaxis, xp.newaxis], P_reg, P,
        )

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
    xp = get_namespace(d)
    di = d[..., :, None]
    dj = d[..., None, :]
    close_ = xp.isclose(di, dj, atol=atol, rtol=rtol)
    safe_diff = xp.where(close_, xp.ones_like(di - dj), di - dj)
    return xp.where(close_, fctder(di), (fct(di) - fct(dj)) / safe_diff)


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
        SPD/HPD matrices, at least 2D ndarray.
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
    xp = get_namespace(X, Cref)
    d, V = xp.linalg.eigh(Cref)
    
    Vh = ctranspose(V)
    expfdd = _first_divided_difference(d, xp.exp, xp.exp)
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
    xp = get_namespace(X, Cref)
    d, V = xp.linalg.eigh(Cref)

    Vh = ctranspose(V)
    logfdd = _first_divided_difference(d, xp.log, lambda x: 1 / x)
    return V @ (logfdd * (Vh @ X @ V)) @ Vh
