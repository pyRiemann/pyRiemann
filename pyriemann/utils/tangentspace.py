"""Tangent space for SPD/HPD matrices."""

import numpy as np

from .base import expm, invsqrtm, logm, sqrtm, ddexpm, ddlogm
from .utils import check_function


def _check_dimensions(X, Cref):
    n_1, n_2 = X.shape[-2:]
    n_3, n_4 = Cref.shape
    if not (n_1 == n_2 == n_3 == n_4):
        raise ValueError("Inputs have incompatible dimensions.")


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
    Cref_chol = np.linalg.cholesky(Cref)
    Cref_invchol = np.linalg.inv(Cref_chol)

    tri0, tri1 = np.tril_indices(X.shape[-1], -1)
    diag0, diag1 = np.diag_indices(X.shape[-1])

    diff_bracket = Cref_invchol @ X @ Cref_invchol.conj().T
    diff_bracket[..., tri1, tri0] = 0
    diff_bracket[..., diag0, diag1] /= 2
    diff = Cref_chol @ diff_bracket

    exp_map = np.zeros_like(X)

    exp_map[..., tri0, tri1] = Cref_chol[..., tri0, tri1] + \
        diff[..., tri0, tri1]

    exp_map[..., diag0, diag1] = np.exp(diff_bracket[..., diag0, diag1]) \
        * Cref_chol[..., diag0, diag1]

    return exp_map @ exp_map.conj().swapaxes(-1, -2)


def exp_map_logeuclid(X, Cref):
    r"""Project matrices back to manifold by log-Euclidean exponential map.

    The projection of a matrix :math:`\mathbf{X}` from tangent space
    to SPD/HPD manifold with log-Euclidean exponential map
    according to a reference SPD/HPD matrix :math:`\mathbf{C}_\text{ref}` as
    described in Eq.(3.4) of [1]_:

    .. math::
        \mathbf{X}_\text{original} = \exp \left( \log(\mathbf{C}_\text{ref})
        + [D_{\mathbf{C}_\text{ref}} \log] \left(\mathbf{X}\right) \right)

    where :math:`[D_{\mathbf{C}_\text{ref}} \log] \left( \mathbf{X}\right)`
    indicates the differential of the matrix logarithm at point
    :math:`\mathbf{C}_\text{ref}` applied to :math:`\mathbf{X}`.
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
    return expm(logm(Cref) + ddlogm(X, Cref))


def exp_map_riemann(X, Cref, Cm12=False):
    r"""Project matrices back to manifold by Riemannian exponential map.

    The projection of a matrix :math:`\mathbf{X}` from tangent space
    to SPD/HPD manifold with Riemannian exponential map
    according to a reference SPD/HPD matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{original} = \mathbf{C}_\text{ref}^{1/2}
        \exp(\mathbf{X}) \mathbf{C}_\text{ref}^{1/2}

    When Cm12=True, it returns the full Riemannian exponential map as in
    Section 3.4 of [1]_:

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

    d, V = np.linalg.eigh(Cref)
    C = 1 / (d[:, None] + d[None, :])

    X_rotated = V.conj().T @ X @ V
    X_tmp = C * X_rotated
    X_tmp = X_tmp @ np.diag(d) @ X_tmp
    X_tmp = V @ X_tmp @ V.conj().T

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
    to tangent space by log-Cholesky logarithmic map, see [1]_ Table 2.

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
    X_chol = np.linalg.cholesky(X)
    Cref_chol = np.linalg.cholesky(Cref)

    res = np.zeros_like(X)

    tri0, tri1 = np.tril_indices(X.shape[-1], -1)
    res[..., tri0, tri1] = X_chol[..., tri0, tri1] - Cref_chol[..., tri0, tri1]

    diag0, diag1 = np.diag_indices(X.shape[-1])
    res[..., diag0, diag1] = Cref_chol[..., diag0, diag1] * \
        np.log(X_chol[..., diag0, diag1] / Cref_chol[..., diag0, diag1])

    X_new = Cref_chol @ res.conj().swapaxes(-1, -2) + \
        res @ Cref_chol.conj().swapaxes(-1, -2)

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

    where :math:`[D_{\log(\mathbf{C}_\text{ref})} \exp]\left(\mathbf{X}\right)`
    indicates the differential of the matrix exponential at point
    :math:`\log(\mathbf{C}_\text{ref})` applied to :math:`\mathbf{X}`.
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
    _check_dimensions(X, Cref)
    logCref = logm(Cref)
    return ddexpm(logm(X) - logCref, logCref)


def log_map_riemann(X, Cref, C12=False):
    r"""Project matrices in tangent space by Riemannian logarithmic map.

    The projection of a matrix :math:`\mathbf{X}` from SPD/HPD manifold
    to tangent space by Riemannian logarithmic map
    according to a SPD/HPD reference matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{new} = \log ( \mathbf{C}_\text{ref}^{-1/2}
        \mathbf{X} \mathbf{C}_\text{ref}^{-1/2})

    When C12=True, it returns the full Riemannian logarithmic map as in
    Section 3.4 of [1]_:

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
    _check_dimensions(X, Cref)
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
    _check_dimensions(X, Cref)
    P12 = sqrtm(Cref)
    P12inv = invsqrtm(Cref)
    sqrt_bracket = sqrtm(P12 @ X @ P12)
    tmp = P12inv @ sqrt_bracket @ P12
    return tmp + tmp.conj().swapaxes(-2, -1) - 2 * Cref


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
    n = X.shape[-1]
    if X.shape[-2] != n:
        raise ValueError("Matrices must be square")
    idx = np.triu_indices_from(np.empty((n, n)))
    coeffs = (np.sqrt(2) * np.triu(np.ones((n, n)), 1) + np.eye(n))[idx]
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
    dims = T.shape
    n = int((np.sqrt(1 + 8 * dims[-1]) - 1) / 2)
    X = np.empty((*dims[:-1], n, n), dtype=T.dtype)
    idx = np.triu_indices_from(np.empty((n, n)))
    X[..., idx[0], idx[1]] = T
    idx = np.triu_indices_from(np.empty((n, n)), k=1)
    X[..., idx[0], idx[1]] /= np.sqrt(2)
    X[..., idx[1], idx[0]] = X[..., idx[0], idx[1]].conj()
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

# NOT IN API
def transport(X, A, B):
    r"""Parallel transport of matrices in tangent space.

    The parallel transport of matrices :math:`\mathbf{X}` in tangent space
    from an initial SPD/HPD matrix :math:`\mathbf{A}` to a final SPD/HPD
    matrix :math:`\mathbf{B}` according to the Levi-Civita connection along
    the geodesic under the affine-invariant metric is [1]_:

    .. math::
        \mathbf{X}_\text{new} = \mathbf{E} \mathbf{X} \mathbf{E}^H

    where :math:`\mathbf{E} = (\mathbf{B} \mathbf{A}^{-1})^{1/2}`.

    Warning: this function must be applied to matrices already projected in
    tangent space with a logarithmic map, not to SPD/HPD matrices in manifold.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Symmetric/Hermitian matrices in tangent space.
    A : ndarray, shape (n, n)
        Initial SPD/HPD matrix.
    B : ndarray, shape (n, n)
        Final SPD/HPD matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Transported matrices in tangent space.

    Notes
    -----
    .. versionchanged:: 0.8
        Change input arguments.

    References
    ----------
    .. [1] `Conic geometric optimisation on the manifold of positive definite
        matrices
        <https://arxiv.org/pdf/1312.1039>`_
        S. Sra and R. Hosseini. SIAM Journal on Optimization, 2015.
    """
    # (BA^{-1})^{1/2} = A^{1/2}(A^{-1/2}BA^{-1/2})^{1/2}A^{-1/2}
    A12inv = invsqrtm(A)
    A12 = sqrtm(A)
    E = A12 @ sqrtm(A12inv @ B @ A12inv) @ A12inv
    X_new = E @ X @ E.conj().T
    return X_new
