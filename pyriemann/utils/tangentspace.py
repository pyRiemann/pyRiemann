"""Tangent space for SPD/HPD matrices."""

import math

import numpy as np

from ._backend import (
    check_matrix_pair,
    create_diagonal,
    diag_indices,
    get_namespace,
    tril_indices,
    triu_indices,
    xpd,
)
from .base import ctranspose, expm, invsqrtm, logm, sqrtm, ddexpm, ddlogm
from .utils import check_function


def exp_map_euclid(X, Cref):
    r"""Project matrices back to manifold by Euclidean exponential map.

    The projection of a matrix :math:`\mathbf{X}` from tangent space
    to manifold with Euclidean exponential map
    according to a reference matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{original} = \mathbf{X} + \mathbf{C}_\text{ref}

    Parameters
    ----------
    X : ndarray, shape (..., n, m)
        Matrices in tangent space.
    Cref : ndarray, shape (n, m)
        Reference matrix.

    Returns
    -------
    X_original : ndarray, shape (..., n, m)
        Matrices in manifold.

    Notes
    -----
    .. versionadded:: 0.4
    """
    return X + Cref


def exp_map_logchol(X, Cref):
    r"""Project matrices back to manifold by log-Cholesky exponential map.

    The projection of a matrix :math:`\mathbf{X}` from tangent space
    to SPD/HPD manifold with log-Cholesky exponential map, see Table 2 of [1]_.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in tangent space.
    Cref : ndarray, shape (n, n)
        Reference SPD/HPD matrix.

    Returns
    -------
    X_original : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manifold.

    Notes
    -----
    .. versionadded:: 0.7

    References
    ----------
    .. [1] `Riemannian geometry of symmetric positive definite matrices via
        Cholesky decomposition
        <https://arxiv.org/pdf/1908.09326>`_
        Z. Lin. SIAM J Matrix Anal Appl, 2019, 40(4), pp. 1353-1370.
    """
    xp = check_matrix_pair(X, Cref, require_square=True)
    Cref_chol = xp.linalg.cholesky(Cref)
    eye_n = xp.eye(Cref.shape[-1], dtype=Cref.dtype, device=xpd(Cref))
    Cref_invchol = xp.linalg.solve(Cref_chol, eye_n)

    tri0, tri1 = tril_indices(X.shape[-1], -1, xp=xp, like=X)
    diag0, diag1 = diag_indices(X.shape[-1], xp=xp, like=X)

    diff_bracket = Cref_invchol @ X @ ctranspose(Cref_invchol)
    diff_bracket[..., tri1, tri0] = 0
    diff_bracket[..., diag0, diag1] /= 2
    diff = Cref_chol @ diff_bracket

    exp_map = xp.zeros(diff.shape, dtype=diff.dtype, device=xpd(diff))

    exp_map[..., tri0, tri1] = Cref_chol[..., tri0, tri1] + \
        diff[..., tri0, tri1]

    exp_map[..., diag0, diag1] = (
        xp.exp(diff_bracket[..., diag0, diag1]) *
        Cref_chol[..., diag0, diag1]
    )

    return exp_map @ ctranspose(exp_map)


def exp_map_logeuclid(X, Cref):
    r"""Project matrices back to manifold by log-Euclidean exponential map.

    The projection of a matrix :math:`\mathbf{X}` from tangent space
    to SPD/HPD manifold with log-Euclidean exponential map
    according to a reference SPD/HPD matrix :math:`\mathbf{C}_\text{ref}` as
    described in Eq.(3.4) of [1]_:

    .. math::
        \mathbf{X}_\text{original} = \exp \left( \log(\mathbf{C}_\text{ref})
        + [D_{\mathbf{C}_\text{ref}} \log] \left(\mathbf{X}\right) \right)

    where :math:`[D_{\mathbf{A}} \log] \left( \mathbf{B}\right)`
    indicates the differential of the matrix logarithm at point
    :math:`\mathbf{A}` applied to :math:`\mathbf{B}`.
    Calculation is performed according to Eq. (5) in [2]_.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in tangent space.
    Cref : ndarray, shape (n, n)
        Reference SPD/HPD matrix.

    Returns
    -------
    X_original : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manifold.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] `Geometric Means in a Novel Vector Space Structure on Symmetric
        Positive‐Definite Matrices
        <https://epubs.siam.org/doi/10.1137/050637996>`_
        V. Arsigny, P. Fillard, X. Pennec, N. Ayache. SIMAX, 2006, 29(1),
        pp. 328-347.
    .. [2] `A New Canonical Log-Euclidean Kernel for Symmetric Positive
        Definite Matrices for EEG Analysis
        <https://ieeexplore.ieee.org/document/10735221>`_
        G. Wagner vom Berg, V. Röhr, D. Platt, B. Blankertz. IEEE TBME, 2024.
    """
    xp = check_matrix_pair(X, Cref, require_square=True)
    return expm(
        logm(Cref) + ddlogm(X, Cref),
    )


def exp_map_riemann(X, Cref, Cm12=False):
    r"""Project matrices back to manifold by Riemannian exponential map.

    The projection of a matrix :math:`\mathbf{X}` from tangent space
    to SPD/HPD manifold with affine-invariant Riemannian exponential map
    according to a reference SPD/HPD matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{original} = \mathbf{C}_\text{ref}^{1/2}
        \exp(\mathbf{X}) \mathbf{C}_\text{ref}^{1/2}

    When Cm12=True, it returns the full affine-invariant Riemannian exponential
    map as in Section 3.4 of [1]_:

    .. math::
        \mathbf{X}_\text{original} = \mathbf{C}_\text{ref}^{1/2}
        \exp( \mathbf{C}_\text{ref}^{-1/2} \mathbf{X}
        \mathbf{C}_\text{ref}^{-1/2}) \mathbf{C}_\text{ref}^{1/2}

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in tangent space.
    Cref : ndarray, shape (n, n)
        Reference SPD/HPD matrix.
    Cm12 : bool, default=False
        If True, it returns the full Riemannian exponential map.

    Returns
    -------
    X_original : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manifold.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] `A Riemannian Framework for Tensor Computing
        <https://link.springer.com/article/10.1007/s11263-005-3222-z>`_
        X. Pennec, P. Fillard, N. Ayache. IJCV, 2006, 66(1), pp. 41-66.
    """
    xp = check_matrix_pair(X, Cref, require_square=True)
    if Cm12:
        Cm12 = invsqrtm(Cref)
        X = Cm12 @ X @ Cm12
    C12 = sqrtm(Cref)
    return C12 @ expm(X) @ C12


def exp_map_wasserstein(X, Cref):
    r"""Project matrices back to manifold by Wasserstein exponential map.

    The projection of a matrix :math:`\mathbf{X}` from tangent space
    to SPD/HPD manifold with Wasserstein exponential map according to a
    reference SPD/HPD matrix :math:`\mathbf{C}_\text{ref}` is given in
    Eq.(36) of [1]_.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in tangent space.
    Cref : ndarray, shape (n, n)
        Reference SPD/HPD matrix.

    Returns
    -------
    X_original : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manifold.

    Notes
    -----
    .. versionadded:: 0.8

    References
    ----------
    .. [1] `Wasserstein Riemannian geometry of Gaussian densities
        <https://link.springer.com/article/10.1007/s41884-018-0014-4>`_
        L. Malagò, L. Montrucchio, G. Pistone. Information Geometry, 2018, 1,
        pp. 137–179.
    """
    xp = check_matrix_pair(X, Cref, require_square=True)
    d, V = xp.linalg.eigh(Cref)
    C = 1 / (d[..., :, None] + d[..., None, :])
    d = xp.asarray(d, dtype=Cref.dtype, device=xpd(Cref))

    X_rotated = ctranspose(V) @ X @ V
    X_tmp = C * X_rotated
    X_tmp = X_tmp @ create_diagonal(d) @ X_tmp
    X_tmp = V @ X_tmp @ ctranspose(V)

    return Cref + X + X_tmp


exp_map_functions = {
    "euclid": exp_map_euclid,
    "logchol": exp_map_logchol,
    "logeuclid": exp_map_logeuclid,
    "riemann": exp_map_riemann,
    "wasserstein": exp_map_wasserstein,
}


def exp_map(X, Cref, *, metric="riemann"):
    """Project matrices back to manifold by exponential map.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in tangent space.
    Cref : ndarray, shape (n, n)
        Reference matrix.
    metric : string | callable, default="riemann"
        Metric used for exponential map, can be:
        "euclid", "logchol", "logeuclid", "riemann", "wasserstein",
        or a callable function.

    Returns
    -------
    X_original : ndarray, shape (..., n, n)
        Matrices in manifold.

    Notes
    -----
    .. versionadded:: 0.9

    See Also
    --------
    exp_map_euclid
    exp_map_logchol
    exp_map_logeuclid
    exp_map_riemann
    exp_map_wasserstein
    """
    exp_map_function = check_function(metric, exp_map_functions)
    return exp_map_function(X, Cref)


###############################################################################


def log_map_euclid(X, Cref):
    r"""Project matrices in tangent space by Euclidean logarithmic map.

    The projection of a matrix :math:`\mathbf{X}` from manifold
    to tangent space by Euclidean logarithmic map
    according to a reference matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{new} = \mathbf{X} - \mathbf{C}_\text{ref}

    Parameters
    ----------
    X : ndarray, shape (..., n, m)
        Matrices in manifold.
    Cref : ndarray, shape (n, m)
        Reference matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n, m)
        Matrices projected in tangent space.

    Notes
    -----
    .. versionadded:: 0.4
    """
    return X - Cref


def log_map_logchol(X, Cref):
    r"""Project matrices in tangent space by log-Cholesky logarithmic map.

    The projection of a matrix :math:`\mathbf{X}` from SPD/HPD manifold
    to tangent space by log-Cholesky logarithmic map, see Table 2 of [1]_ .

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manifold.
    Cref : ndarray, shape (n, n)
        Reference SPD matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Matrices projected in tangent space.

    Notes
    -----
    .. versionadded:: 0.7

    References
    ----------
    .. [1] `Riemannian geometry of symmetric positive definite matrices via
        Cholesky decomposition
        <https://arxiv.org/pdf/1908.09326>`_
        Z. Lin. SIAM J Matrix Anal Appl, 2019, 40(4), pp. 1353-1370.
    """
    xp = check_matrix_pair(X, Cref, require_square=True)
    X_chol = xp.linalg.cholesky(X)
    Cref_chol = xp.linalg.cholesky(Cref)

    batch_shape = np.broadcast_shapes(
        X_chol.shape[:-2],
        Cref_chol.shape[:-2],
    )
    res = xp.zeros(batch_shape + X_chol.shape[-2:], dtype=X_chol.dtype, device=xpd(X_chol))

    tri0, tri1 = tril_indices(X.shape[-1], -1, xp=xp, like=X)
    res[..., tri0, tri1] = X_chol[..., tri0, tri1] - Cref_chol[..., tri0, tri1]

    diag0, diag1 = diag_indices(X.shape[-1], xp=xp, like=X)
    res[..., diag0, diag1] = Cref_chol[..., diag0, diag1] * xp.log(
            X_chol[..., diag0, diag1] / Cref_chol[..., diag0, diag1]
    )

    X_new = (
        Cref_chol @ ctranspose(res) +
        res @ ctranspose(Cref_chol)
    )

    return X_new


def log_map_logeuclid(X, Cref):
    r"""Project matrices in tangent space by log-Euclidean logarithmic map.

    The projection of a matrix :math:`\mathbf{X}` from SPD/HPD manifold
    to tangent space by log-Euclidean logarithmic map according to a SPD/HPD
    reference matrix :math:`\mathbf{C}_\text{ref}` as described in Eq.(3.4)
    of [1]_:

    .. math::
        \mathbf{X}_\text{new} = [D_{\log(\mathbf{C}_\text{ref})} \exp] \left(
        \log(\mathbf{X}) - \log(\mathbf{C}_\text{ref}) \right)

    where :math:`[D_{\mathbf{A}} \exp]\left(\mathbf{B}\right)`
    indicates the differential of the matrix exponential at point
    :math:`\mathbf{A}` applied to :math:`\mathbf{B}`.
    Calculation is performed according to Eq. (7) in [2]_.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manifold.
    Cref : ndarray, shape (n, n)
        Reference SPD matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Matrices projected in tangent space.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] `Geometric Means in a Novel Vector Space Structure on Symmetric
        Positive‐Definite Matrices
        <https://epubs.siam.org/doi/10.1137/050637996>`_
        V. Arsigny, P. Fillard, X. Pennec, N. Ayache. SIMAX, 2006, 29(1),
        pp. 328-347.
    .. [2] `A New Canonical Log-Euclidean Kernel for Symmetric Positive
        Definite Matrices for EEG Analysis
        <https://ieeexplore.ieee.org/document/10735221>`_
        G. Wagner vom Berg, V. Röhr, D. Platt, B. Blankertz. IEEE TBME, 2024.
    """
    xp = check_matrix_pair(X, Cref, require_square=True)
    logCref = logm(Cref)
    X_new = ddexpm(
        logm(X) - logCref,
        logCref,
    )
    return X_new


def log_map_riemann(X, Cref, C12=False):
    r"""Project matrices in tangent space by Riemannian logarithmic map.

    The projection of a matrix :math:`\mathbf{X}` from SPD/HPD manifold
    to tangent space by affine-invariant Riemannian logarithmic map
    according to a SPD/HPD reference matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{new} = \log ( \mathbf{C}_\text{ref}^{-1/2}
        \mathbf{X} \mathbf{C}_\text{ref}^{-1/2})

    When C12=True, it returns the full affine-invariant Riemannian logarithmic
    map as in Section 3.4 of [1]_:

    .. math::
        \mathbf{X}_\text{new} = \mathbf{C}_\text{ref}^{1/2}
        \log( \mathbf{C}_\text{ref}^{-1/2} \mathbf{X}
        \mathbf{C}_\text{ref}^{-1/2}) \mathbf{C}_\text{ref}^{1/2}

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manifold.
    Cref : ndarray, shape (n, n)
        Reference SPD/HPD matrix.
    C12 : bool, default=False
        If True, it returns the full Riemannian logarithmic map.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Matrices projected in tangent space.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] `A Riemannian Framework for Tensor Computing
        <https://link.springer.com/article/10.1007/s11263-005-3222-z>`_
        X. Pennec, P. Fillard, N. Ayache. IJCV, 2006, 66(1), pp. 41-66.
    """
    xp = check_matrix_pair(X, Cref, require_square=True)
    Cm12 = invsqrtm(Cref)
    X_new = logm(Cm12 @ X @ Cm12)
    if C12:
        C12 = sqrtm(Cref)
        X_new = C12 @ X_new @ C12
    return X_new


def log_map_wasserstein(X, Cref):
    r"""Project matrices in tangent space by Wasserstein logarithmic map.

    The projection of a matrix :math:`\mathbf{X}` from SPD/HPD manifold
    to tangent space by the Wasserstein logarithmic map
    according to a SPD/HPD reference matrix :math:`\mathbf{C}_\text{ref}` is
    given in Proposition 9 of [1]_:

    .. math::
        \mathbf{X}_\text{new} = (\mathbf{X}\mathbf{C}_\text{ref})^{1/2} +
        (\mathbf{C}_\text{ref}\mathbf{X})^{1/2} - 2\mathbf{C}_\text{ref}

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manifold.
    Cref : ndarray, shape (n, n)
        Reference SPD/HPD matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Matrices projected in tangent space.

    Notes
    -----
    .. versionadded:: 0.8

    References
    ----------
    .. [1] `Wasserstein Riemannian geometry of Gaussian densities
        <https://link.springer.com/article/10.1007/s41884-018-0014-4>`_
        L. Malagò, L. Montrucchio, G. Pistone. Information Geometry, 2018, 1,
        pp. 137–179.
    """
    xp = check_matrix_pair(X, Cref, require_square=True)
    P12 = sqrtm(Cref)
    P12inv = invsqrtm(Cref)
    sqrt_bracket = sqrtm(P12 @ X @ P12)
    tmp = P12inv @ sqrt_bracket @ P12
    return tmp + ctranspose(tmp) - 2 * Cref


log_map_functions = {
    "euclid": log_map_euclid,
    "logchol": log_map_logchol,
    "logeuclid": log_map_logeuclid,
    "riemann": log_map_riemann,
    "wasserstein": log_map_wasserstein,
}


def log_map(X, Cref, *, metric="riemann"):
    """Project matrices in tangent space by logarithmic map.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in manifold.
    Cref : ndarray, shape (n, n)
        Reference matrix.
    metric : string | callable, default="riemann"
        Metric used for logarithmic map, can be:
        "euclid", "logchol", "logeuclid", "riemann", "wasserstein",
        or a callable function.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Matrices projected in tangent space.

    Notes
    -----
    .. versionadded:: 0.9

    See Also
    --------
    log_map_euclid
    log_map_logchol
    log_map_logeuclid
    log_map_riemann
    log_map_wasserstein
    """
    log_map_function = check_function(metric, log_map_functions)
    return log_map_function(X, Cref)


###############################################################################


def upper(X):
    r"""Return the weighted upper triangular part of matrices.

    This function computes the minimal representation of a matrix in tangent
    space [1]_: it keeps the upper triangular part of the symmetric/Hermitian
    matrix and vectorizes it by applying unity weight for diagonal elements and
    :math:`\sqrt{2}` weight for out-of-diagonal elements.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Symmetric/Hermitian matrices.

    Returns
    -------
    T : ndarray, shape (..., n * (n + 1) / 2)
        Weighted upper triangular parts of symmetric/Hermitian matrices.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] `Pedestrian detection via classification on Riemannian manifolds
        <https://ieeexplore.ieee.org/document/4479482>`_
        O. Tuzel, F. Porikli, and P. Meer. IEEE Transactions on Pattern
        Analysis and Machine Intelligence, Volume 30, Issue 10, October 2008.
    """
    xp = get_namespace(X)
    n = X.shape[-1]
    if X.shape[-2] != n:
        raise ValueError("Matrices must be square")
    idx = triu_indices(n, xp=xp, like=X)
    idx_no_diag = triu_indices(n, k=1, xp=xp, like=X)
    coeffs = xp.ones((n, n), dtype=X.real.dtype, device=xpd(X))
    coeffs[idx_no_diag[0], idx_no_diag[1]] = math.sqrt(2.0)
    coeffs = coeffs[idx[0], idx[1]]
    T = coeffs * X[..., idx[0], idx[1]]
    return T


def unupper(T):
    """Inverse upper function.

    This function is the inverse of upper function: it reconstructs symmetric/
    Hermitian matrices from their weighted upper triangular parts.

    Parameters
    ----------
    T : ndarray, shape (..., n * (n + 1) / 2)
        Weighted upper triangular parts of symmetric/Hermitian matrices.

    Returns
    -------
    X : ndarray, shape (..., n, n)
        Symmetric/Hermitian matrices.

    See Also
    --------
    upper

    Notes
    -----
    .. versionadded:: 0.4
    """
    xp = get_namespace(T)
    dims = T.shape
    n = int((math.sqrt(1 + 8 * dims[-1]) - 1) / 2)
    X = xp.zeros((*dims[:-1], n, n), dtype=T.dtype, device=xpd(T))
    idx = triu_indices(n, xp=xp, like=T)
    X[..., idx[0], idx[1]] = T
    idx = triu_indices(n, k=1, xp=xp, like=T)
    X[..., idx[0], idx[1]] /= math.sqrt(2.0)
    X[..., idx[1], idx[0]] = xp.conj(X[..., idx[0], idx[1]])
    return X


def tangent_space(X, Cref, *, metric="riemann"):
    """Transform matrices into tangent vectors.

    Transform matrices into tangent vectors, according to a reference
    matrix Cref and to a specific logarithmic map.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in manifold.
    Cref : ndarray, shape (n, n)
        Reference matrix.
    metric : string | callable, default="riemann"
        Metric used for logarithmic map, can be:
        "euclid", "logchol", "logeuclid", "riemann", "wasserstein",
        or a callable function.

    Returns
    -------
    T : ndarray, shape (..., n * (n + 1) / 2)
        Tangent vectors.

    See Also
    --------
    log_map
    upper
    """
    X_ = log_map(X, Cref, metric=metric)
    T = upper(X_)
    return T


def untangent_space(T, Cref, *, metric="riemann"):
    """Transform tangent vectors back to matrices.

    Transform tangent vectors back to matrices, according to a reference
    matrix Cref and to a specific exponential map.

    Parameters
    ----------
    T : ndarray, shape (..., n * (n + 1) / 2)
        Tangent vectors.
    Cref : ndarray, shape (n, n)
        Reference matrix.
    metric : string | callable, default="riemann"
        Metric used for exponential map, can be:
        "euclid", "logchol", "logeuclid", "riemann", "wasserstein",
        or a callable function.

    Returns
    -------
    X : ndarray, shape (..., n, n)
        Matrices in manifold.

    See Also
    --------
    unupper
    exp_map
    """
    X_ = unupper(T)
    X = exp_map(X_, Cref, metric=metric)
    return X


###############################################################################


def innerproduct_euclid(X, Y, *args):
    r"""Euclidean inner product.

    Euclidean inner product :math:`\mathbf{g}`
    between matrices :math:`\mathbf{X}` and :math:`\mathbf{Y}` is:

    .. math::
        \mathbf{g}(\mathbf{X}, \mathbf{Y}) = \text{tr}(\mathbf{X}^H \mathbf{Y})

    Parameters
    ----------
    X : ndarray, shape (..., n, m)
        First  matrices.
    Y : ndarray, shape (..., n, m) | None
        Second matrices.
        If None, Y is set to X, giving the squared norm of X.

    Returns
    -------
    G : float or ndarray, shape (...,)
        Euclidean inner product between X and Y.

    Notes
    -----
    .. versionadded:: 0.11

    See Also
    --------
    innerproduct
    """
    if Y is None:
        Y = X
    G = _apply_inner_product(X, Y)
    return G


def innerproduct_logeuclid(X, Y, Cref):
    r"""Log-Euclidean inner product.

    Log-Euclidean inner product :math:`\mathbf{g}` between
    symmetric/Hermitian matrices in tangent space :math:`\mathbf{X}`
    and :math:`\mathbf{Y}` at :math:`\mathbf{C}_\text{ref}` is [1]_:

    .. math::
        \mathbf{g}_{\mathbf{C}_\text{ref}}(\mathbf{X}, \mathbf{Y}) =
            \text{tr} \left(
            [D_{\mathbf{C}_\text{ref}} \log](X)^*
            [D_{\mathbf{C}_\text{ref}} \log](Y)
            \right)

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        First symmetric/Hermitian matrices in tangent space at Cref.
    Y : ndarray, shape (..., n, n) | None
        Second symmetric/Hermitian matrices in tangent space at Cref.
        If None, Y is set to X, giving the squared norm of X.
    Cref : ndarray, shape (n, n)
        Reference SPD/HPD matrix.

    Returns
    -------
    G : float or ndarray, shape (...,)
        Log-Euclidean inner product between X and Y.

    Notes
    -----
    .. versionadded:: 0.11

    See Also
    --------
    innerproduct

    References
    ----------
    .. [1] `Geometric means in a novel vector space structure on symmetric
        positive-definite matrices
        <https://epubs.siam.org/doi/abs/10.1137/050637996>`_
        V. Arsigny, P. Fillard, X. Pennec, N. Ayache.
        SIAM J Matrix Anal Appl, 2007, 29 (1), pp. 328-347
    """
    if Y is None:
        Y = X
    G = _apply_inner_product(ddlogm(X, Cref), ddlogm(Y, Cref))
    return G


def innerproduct_riemann(X, Y, Cref):
    r"""Affine-invariant Riemannian inner product.

    Affine-invariant Riemannian inner product :math:`\mathbf{g}` between
    symmetric/Hermitian matrices in tangent space :math:`\mathbf{X}`
    and :math:`\mathbf{Y}` at :math:`\mathbf{C}_\text{ref}` is [1]_:

    .. math::
        \mathbf{g}_{\mathbf{C}_\text{ref}}(\mathbf{X}, \mathbf{Y}) =
            \text{tr} \left(
            (\mathbf{C}_\text{ref}^{-1/2} \mathbf{X}
            \mathbf{C}_\text{ref}^{-1/2})^*
            \mathbf{C}_\text{ref}^{-1/2} \mathbf{Y}
            \mathbf{C}_\text{ref}^{-1/2}
            \right)

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        First symmetric/Hermitian matrices in tangent space at Cref.
    Y : ndarray, shape (..., n, n) | None
        Second symmetric/Hermitian matrices in tangent space at Cref.
        If None, Y is set to X, giving the squared norm of X.
    Cref : ndarray, shape (n, n)
        Reference SPD/HPD matrix.

    Returns
    -------
    G : float or ndarray, shape (...,)
        Affine-invariant Riemannian inner product between X and Y.

    Notes
    -----
    .. versionadded:: 0.11

    See Also
    --------
    innerproduct

    References
    ----------
    .. [1] `A metric for covariance matrices
        <https://www.ipb.uni-bonn.de/pdfs/Forstner1999Metric.pdf>`_
        W. Förstner & B. Moonen.
        Geodesy-the Challenge of the 3rd Millennium, 2003
    """
    if Y is None:
        Y = X
    Cm12 = invsqrtm(Cref)
    G = _apply_inner_product(Cm12 @ X @ Cm12, Cm12 @ Y @ Cm12)
    return G


def _apply_inner_product(X, Y):
    # product G = trace(X^H @ Y)
    G = np.einsum("...nm,...nm->...", X.conj(), Y, optimize=True).real

    if G.size == 1:
        return G.item()
    else:
        return G


innerproduct_functions = {
    "euclid": innerproduct_euclid,
    "logeuclid": innerproduct_logeuclid,
    "riemann": innerproduct_riemann,
}


def innerproduct(X, Y, Cref, metric="riemann"):
    r"""Inner product according to a specified metric.

    It calculates the inner product between matrices in the tangent space
    :math:`\mathbf{X}` and :math:`\mathbf{Y}` at :math:`\mathbf{C}_\text{ref}`,
    according to a specified metric.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        First matrices in tangent space at Cref.
    Y : ndarray, shape (..., n, n) | None
        Second matrices in tangent space at Cref.
        If None, Y is set to X, giving the squared norm of X.
    Cref : ndarray, shape (n, n) | None
        Reference matrix.
    metric : string | callable, default="riemann"
        Metric used for inner product, can be:
        "euclid", "logeuclid", "riemann", or a callable function.

    Returns
    -------
    G : float or ndarray, shape (...,)
        Inner product between X and Y.

    Notes
    -----
    .. versionadded:: 0.11

    See Also
    --------
    innerproduct_euclid
    innerproduct_logeuclid
    innerproduct_riemann
    """
    innerproduct_function = check_function(metric, innerproduct_functions)
    G = innerproduct_function(X, Y, Cref)
    return G


def norm(X, Cref, metric="riemann"):
    r"""Norm according to a specified metric.

    It calculates the norm of the matrix :math:`\mathbf{X}`
    in the tangent space at :math:`\mathbf{C}_\text{ref}`,
    according to a specified metric.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in the tangent space at Cref.
    Cref : ndarray, shape (n, n) | None
        Reference matrix.
    metric : string | callable, default="riemann"
        Metric used for norm, can be:
        "euclid", "logeuclid", "riemann", or a callable function.

    Returns
    -------
    N : float or ndarray, shape (...,)
        Norm of X.

    Notes
    -----
    .. versionadded:: 0.11

    See Also
    --------
    innerproduct
    """
    N2 = innerproduct(X, None, Cref, metric=metric)
    return np.sqrt(N2)


###############################################################################


def transport_euclid(X, A=None, B=None):
    """Parallel transport for Euclidean metric.

    Parameters
    ----------
    X : ndarray, shape (..., n, m)
        Matrices.
    A : None | ndarray, shape (n, m), default=None
        Initial matrix, unused.
    B : None | ndarray, shape (n, m), default=None
        Final matrix, unused.

    Returns
    -------
    X_new : ndarray, shape (..., n, m)
        Matrices, equal to X.

    Notes
    -----
    .. versionadded:: 0.10

    See Also
    --------
    transport
    """
    return X


def transport_logchol(X, A, B):
    r"""Parallel transport for log-Cholesky metric.

    The parallel transport of matrices :math:`\mathbf{X}` in tangent space
    from an initial SPD/HPD matrix :math:`\mathbf{A}` to a final SPD/HPD
    matrix :math:`\mathbf{B}` for log-Cholesky metric is given in Proposition 7
    of [1]_.

    Warning: this function must be applied to matrices :math:`\mathbf{X}`
    already projected in tangent space with a logarithmic map at
    :math:`\mathbf{A}`, not to SPD/HPD matrices in manifold.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Symmetric/Hermitian matrices in tangent space at A.
    A : ndarray, shape (n, n)
        Initial SPD/HPD matrix.
    B : ndarray, shape (n, n)
        Final SPD/HPD matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Matrices in tangent space transported from A to B.

    Notes
    -----
    .. versionadded:: 0.10
    .. versionchanged:: 0.11
        Correct formula for HPD matrices.

    See Also
    --------
    transport

    References
    ----------
    .. [1] `Riemannian geometry of symmetric positive definite matrices via
        Cholesky decomposition
        <https://arxiv.org/pdf/1908.09326>`_
        Z. Lin. SIAM J Matrix Anal Appl, 2019, 40(4), pp. 1353-1370.
    """
    xp = get_namespace(X, A, B)
    A_chol = xp.linalg.cholesky(A)
    B_chol = xp.linalg.cholesky(B)
    eye_n = xp.eye(A.shape[-1], dtype=A.dtype, device=xpd(A))
    A_invchol = xp.linalg.solve(A_chol, eye_n)

    tri0, tri1 = tril_indices(X.shape[-1], -1, xp=xp, like=X)
    diag0, diag1 = diag_indices(X.shape[-1], xp=xp, like=X)

    P = A_invchol @ X @ ctranspose(A_invchol)
    P12 = xp.zeros_like(P)
    P12[..., tri0, tri1] = P[..., tri0, tri1]
    P12[..., diag0, diag1] = P[..., diag0, diag1] / 2
    X_ = A_chol @ P12

    T = xp.zeros_like(X)
    T[..., tri0, tri1] = X_[..., tri0, tri1]
    T[..., diag0, diag1] = B_chol[..., diag0, diag1] \
        / A_chol[..., diag0, diag1] * X_[..., diag0, diag1]

    X_new = B_chol @ ctranspose(T) + \
        T @ ctranspose(B_chol)
    return X_new


def transport_logeuclid(X, A, B):
    r"""Parallel transport for log-Euclidean metric.

    The parallel transport of matrices :math:`\mathbf{X}` in tangent space
    from an initial SPD/HPD matrix :math:`\mathbf{A}` to a final SPD/HPD
    matrix :math:`\mathbf{B}` for log-Euclidean metric is given in Table 4 of
    [1]_:

    .. math::
        \mathbf{X}_\text{new} = [D_{\log \mathbf{B}} \exp] \left(
        [D_{\mathbf{A}} \log]\left(\mathbf{X}\right)
        \right)

    Warning: this function must be applied to matrices :math:`\mathbf{X}`
    already projected in tangent space with a logarithmic map at
    :math:`\mathbf{A}`, not to SPD/HPD matrices in manifold.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Symmetric/Hermitian matrices in tangent space at A.
    A : ndarray, shape (n, n)
        Initial SPD/HPD matrix.
    B : ndarray, shape (n, n)
        Final SPD/HPD matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Matrices in tangent space transported from A to B.

    Notes
    -----
    .. versionadded:: 0.10
    .. versionchanged:: 0.11
        Correct formula.

    See Also
    --------
    transport

    References
    ----------
    .. [1] `O(n)-invariant Riemannian metrics on SPD matrices
        <https://www.sciencedirect.com/science/article/pii/S0024379522004360>`_
        Y. Thanwerdas & X. Pennec. Linear Algebra and its Applications, 2023.
    """
    return ddexpm(
        ddlogm(X, A),
        logm(B),
    )


def transport_riemann(X, A, B):
    r"""Parallel transport for affine-invariant Riemannian metric.

    The parallel transport of matrices :math:`\mathbf{X}` in tangent space
    from an initial SPD/HPD matrix :math:`\mathbf{A}` to a final SPD/HPD
    matrix :math:`\mathbf{B}` according to the Levi-Civita connection along
    the geodesic under the affine-invariant Riemannian metric is given by
    Eq.(3.4) of [1]_:

    .. math::
        \mathbf{X}_\text{new} = \mathbf{E} \mathbf{X} \mathbf{E}^H

    where :math:`\mathbf{E} = (\mathbf{B} \mathbf{A}^{-1})^{1/2}`.

    Warning: this function must be applied to matrices :math:`\mathbf{X}`
    already projected in tangent space with a logarithmic map at
    :math:`\mathbf{A}`, not to SPD/HPD matrices in manifold.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Symmetric/Hermitian matrices in tangent space at A.
    A : ndarray, shape (n, n)
        Initial SPD/HPD matrix.
    B : ndarray, shape (n, n)
        Final SPD/HPD matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Matrices in tangent space transported from A to B.

    Notes
    -----
    .. versionchanged:: 0.8
        Change input arguments and calculation of the function.
    .. versionchanged:: 0.10
        Rename function and add to API.

    See Also
    --------
    transport

    References
    ----------
    .. [1] `Conic geometric optimisation on the manifold of positive definite
        matrices
        <https://optml.mit.edu/papers/sra_hosseini_gopt.pdf>`_
        S. Sra and R. Hosseini. SIAM Journal on Optimization, 2015.
    """
    # BA^{-1} is not sym => use sqrtm from scipy:
    # E = scipy.linalg.sqrtm(B @ np.linalg.inv(A))
    # (BA^{-1})^{1/2} = A^{1/2} (A^{-1/2}BA^{-1/2})^{1/2} A^{-1/2}
    xp = get_namespace(X, A, B)
    A12 = sqrtm(A)
    A12inv = invsqrtm(A)
    E = A12 @ sqrtm(A12inv @ B @ A12inv) @ A12inv
    X_new = E @ X @ ctranspose(E)
    return X_new


transport_functions = {
    "euclid": transport_euclid,
    "logchol": transport_logchol,
    "logeuclid": transport_logeuclid,
    "riemann": transport_riemann,
}


def transport(X, A, B, metric="riemann"):
    r"""Parallel transport according to a specified metric.

    Parallel transport of matrices :math:`\mathbf{X}` in tangent space
    from an initial SPD/HPD matrix :math:`\mathbf{A}` to a final SPD/HPD
    matrix :math:`\mathbf{B}`, according to a metric.

    Warning: this function must be applied to matrices :math:`\mathbf{X}`
    already projected in tangent space with a logarithmic map at
    :math:`\mathbf{A}`, not to SPD/HPD matrices in manifold.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in tangent space at A.
    A : ndarray, shape (n, n)
        Initial SPD/HPD matrix.
    B : ndarray, shape (n, n)
        Final SPD/HPD matrix.
    metric : string | callable, default="riemann"
        Metric used for parallel transport, can be:
        "euclid", "logchol", "logeuclid", "riemann",
        or a callable function.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Matrices in tangent space transported from A to B.

    Notes
    -----
    .. versionadded:: 0.10

    See Also
    --------
    transport_euclid
    transport_logchol
    transport_logeuclid
    transport_riemann
    """
    transport_function = check_function(metric, transport_functions)
    return transport_function(X, A, B)
