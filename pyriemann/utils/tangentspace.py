"""Tangent space for SPD/HPD matrices."""

import numpy as np

from .base import sqrtm, invsqrtm, logm, expm
from .mean import mean_covariance


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
        The reference matrix.

    Returns
    -------
    X_original : ndarray, shape (..., n, m)
        Matrices in manifold.

    Notes
    -----
    .. versionadded:: 0.4
    """
    return X + Cref


def exp_map_logeuclid(X, Cref):
    r"""Project matrices back to manifold by Log-Euclidean exponential map.

    The projection of a matrix :math:`\mathbf{X}` from tangent space
    to SPD/HPD manifold with Log-Euclidean exponential map
    according to a reference SPD/HPD matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{original} =
        \exp(\mathbf{X} + \log(\mathbf{C}_\text{ref}))

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in tangent space.
    Cref : ndarray, shape (n, n)
        The reference SPD/HPD matrix.

    Returns
    -------
    X_original : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manifold.

    Notes
    -----
    .. versionadded:: 0.4
    """
    return expm(X + logm(Cref))


def exp_map_riemann(X, Cref, Cm12=False):
    r"""Project matrices back to manifold by Riemannian exponential map.

    The projection of a matrix :math:`\mathbf{X}` from tangent space
    to SPD/HPD manifold with Riemannian exponential map
    according to a reference SPD/HPD matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{original} = \mathbf{C}_\text{ref}^{1/2}
        \exp(\mathbf{X}) \mathbf{C}_\text{ref}^{1/2}

    When Cm12=True, it returns the full Riemannian exponential map:

    .. math::
        \mathbf{X}_\text{original} = \mathbf{C}_\text{ref}^{1/2}
        \exp( \mathbf{C}_\text{ref}^{-1/2} \mathbf{X}
        \mathbf{C}_\text{ref}^{-1/2}) \mathbf{C}_\text{ref}^{1/2}

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in tangent space.
    Cref : ndarray, shape (n, n)
        The reference SPD/HPD matrix.
    Cm12 : bool, default=False
        If True, it returns the full Riemannian exponential map.

    Returns
    -------
    X_original : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manifold.

    Notes
    -----
    .. versionadded:: 0.4
    """
    if Cm12:
        Cm12 = invsqrtm(Cref)
        X = Cm12 @ X @ Cm12
    C12 = sqrtm(Cref)
    return C12 @ expm(X) @ C12


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
        Matrices in manidold.
    Cref : ndarray, shape (n, m)
        The reference matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n, m)
        Matrices projected in tangent space.

    Notes
    -----
    .. versionadded:: 0.4
    """
    return X - Cref


def log_map_logeuclid(X, Cref):
    r"""Project matrices in tangent space by Log-Euclidean logarithmic map.

    The projection of a matrix :math:`\mathbf{X}` from SPD/HPD manifold
    to tangent space by Log-Euclidean logarithmic map
    according to a SPD/HPD reference matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{new} = \log(\mathbf{X}) - \log(\mathbf{C}_\text{ref})

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manidold.
    Cref : ndarray, shape (n, n)
        The reference SPD matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Matrices projected in tangent space.

    Notes
    -----
    .. versionadded:: 0.4
    """
    _check_dimensions(X, Cref)
    return logm(X) - logm(Cref)


def log_map_riemann(X, Cref, C12=False):
    r"""Project matrices in tangent space by Riemannian logarithmic map.

    The projection of a matrix :math:`\mathbf{X}` from SPD/HPD manifold
    to tangent space by Riemannian logarithmic map
    according to a SPD/HPD reference matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{new} = \log ( \mathbf{C}_\text{ref}^{-1/2}
        \mathbf{X} \mathbf{C}_\text{ref}^{-1/2})

    When C12=True, it returns the full Riemannian logarithmic map:

    .. math::
        \mathbf{X}_\text{new} = \mathbf{C}_\text{ref}^{1/2}
        \log( \mathbf{C}_\text{ref}^{-1/2} \mathbf{X}
        \mathbf{C}_\text{ref}^{-1/2}) \mathbf{C}_\text{ref}^{1/2}

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manidold.
    Cref : ndarray, shape (n, n)
        The reference SPD/HPD matrix.
    C12 : bool, default=False
        If True, it returns the full Riemannian logarithmic map.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Matrices projected in tangent space.

    Notes
    -----
    .. versionadded:: 0.4
    """
    _check_dimensions(X, Cref)
    Cm12 = invsqrtm(Cref)
    X_new = logm(Cm12 @ X @ Cm12)
    if C12:
        C12 = sqrtm(Cref)
        X_new = C12 @ X_new @ C12
    return X_new


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


def tangent_space(X, Cref, *, metric='riemann'):
    """Transform matrices into tangent vectors.

    Transform matrices into tangent vectors, according to a reference
    matrix Cref and to a specific logarithmic map.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in manidold.
    Cref : ndarray, shape (n, n)
        The reference matrix.
    metric : string, default='riemann'
        The metric used for logarithmic map, can be: 'euclid', 'logeuclid',
        'riemann'.

    Returns
    -------
    T : ndarray, shape (..., n * (n + 1) / 2)
        Tangent vectors.

    See Also
    --------
    log_map_euclid
    log_map_logeuclid
    log_map_riemann
    upper
    """
    log_map_functions = {
        'euclid': log_map_euclid,
        'logeuclid': log_map_logeuclid,
        'riemann': log_map_riemann,
    }
    X_ = log_map_functions[metric](X, Cref)
    T = upper(X_)

    return T


def untangent_space(T, Cref, *, metric='riemann'):
    """Transform tangent vectors back to matrices.

    Transform tangent vectors back to matrices, according to a reference
    matrix Cref and to a specific exponential map.

    Parameters
    ----------
    T : ndarray, shape (..., n * (n + 1) / 2)
        Tangent vectors.
    Cref : ndarray, shape (n, n)
        The reference matrix.
    metric : string, default='riemann'
        The metric used for exponential map, can be: 'euclid', 'logeuclid',
        'riemann'.

    Returns
    -------
    X : ndarray, shape (..., n, n)
        Matrices in manidold.

    See Also
    --------
    unupper
    exp_map_euclid
    exp_map_logeuclid
    exp_map_riemann
    """
    X_ = unupper(T)
    exp_map_functions = {
        'euclid': exp_map_euclid,
        'logeuclid': exp_map_logeuclid,
        'riemann': exp_map_riemann,
    }
    X = exp_map_functions[metric](X_, Cref)

    return X


###############################################################################


# NOT IN API
def transport(Covs, Cref, metric='riemann'):
    """Parallel transport of a set of SPD matrices towards a reference matrix.

    Parameters
    ----------
    Covs : ndarray, shape (n_matrices, n, n)
        Set of SPD matrices.
    Cref : ndarray, shape (n, n)
        The reference SPD matrix.
    metric : string, default='riemann'
        The metric used for mean, can be: 'euclid', 'logeuclid', 'riemann'.

    Returns
    -------
    out : ndarray, shape (n_matrices, n, n)
        Set of transported SPD matrices.
    """
    C = mean_covariance(Covs, metric=metric)
    iC = invsqrtm(C)
    E = sqrtm(iC @ Cref @ iC)
    out = E @ Covs @ E.T
    return out
