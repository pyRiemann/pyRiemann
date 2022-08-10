""" Tangent space for SPD matrices. """

import numpy as np

from .base import sqrtm, invsqrtm, logm, expm
from .mean import mean_covariance


def _check_dimensions(X, Cref):
    n_channels_1, n_channels_2 = X.shape[-2:]
    n_channels_3, n_channels_4 = Cref.shape
    if not (n_channels_1 == n_channels_2 == n_channels_3 == n_channels_4):
        raise ValueError("Inputs have incompatible dimensions.")


def exp_map_euclid(X, Cref):
    r"""Project matrices back to SPD manifold by Euclidean exponential map.

    The projection of a matrix X back to the SPD manifold with Euclidean
    exponential map according to a reference matrix
    :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{new} = \mathbf{X} + \mathbf{C}_\text{ref}

    Parameters
    ----------
    X : ndarray, shape (..., n_channels, n_channels)
        Matrices in tangent space.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.

    Returns
    -------
    X_original : ndarray, shape (..., n_channels, n_channels)
        SPD matrices.

    Notes
    -----
    .. versionadded:: 0.3.1
    """
    return X + Cref


def exp_map_logeuclid(X, Cref):
    r"""Project matrices back to SPD manifold by Log-Euclidean exponential map.

    The projection of a matrix X back to the SPD manifold with Log-Euclidean
    exponential map according to a reference matrix
    :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{new} = \exp(\mathbf{X} + \log(\mathbf{C}_\text{ref}))

    Parameters
    ----------
    X : ndarray, shape (..., n_channels, n_channels)
        Matrices in tangent space.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.

    Returns
    -------
    X_original : ndarray, shape (..., n_channels, n_channels)
        SPD matrices.

    Notes
    -----
    .. versionadded:: 0.3.1
    """
    return expm(X + logm(Cref))


def exp_map_riemann(X, Cref, Cm12=False):
    r"""Project matrices back to SPD manifold by Riemannian exponential map.

    The projection of a matrix X back to the SPD manifold with Riemannian
    exponential map according to a reference matrix
    :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{new} = \mathbf{C}_\text{ref}^{1/2} \exp(\mathbf{X})
        \mathbf{C}_\text{ref}^{1/2}

    When Cm12=True, it returns the full Riemannian exponential map:

    .. math::
        \mathbf{X}_\text{new} = \mathbf{C}_\text{ref}^{1/2}
        \exp( \mathbf{C}_\text{ref}^{-1/2} \mathbf{X}
        \mathbf{C}_\text{ref}^{-1/2}) \mathbf{C}_\text{ref}^{1/2}

    Parameters
    ----------
    X : ndarray, shape (..., n_channels, n_channels)
        Matrices in tangent space.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.
    Cm12 : bool, default=False
        If True, it returns the full Riemannian exponential map.

    Returns
    -------
    X_original : ndarray, shape (..., n_channels, n_channels)
        SPD matrices.

    Notes
    -----
    .. versionadded:: 0.3.1
    """
    if Cm12:
        Cm12 = invsqrtm(Cref)
        X = Cm12 @ X @ Cm12
    C12 = sqrtm(Cref)
    return C12 @ expm(X) @ C12


def log_map_euclid(X, Cref):
    r"""Project SPD matrices in tangent space by Euclidean logarithmic map.

    The projection of a SPD matrix X in the tangent space by Euclidean
    logarithmic map according to a reference matrix
    :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{new} = \mathbf{X} - \mathbf{C}_\text{ref}

    Parameters
    ----------
    X : ndarray, shape (..., n_channels, n_channels)
        SPD matrices.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n_channels, n_channels)
        SPD matrices projected in tangent space.

    Notes
    -----
    .. versionadded:: 0.3.1
    """
    _check_dimensions(X, Cref)
    return X - Cref


def log_map_logeuclid(X, Cref):
    r"""Project SPD matrices in tangent space by Log-Euclidean logarithmic map.

    The projection of a SPD matrix X in the tangent space by Log-Euclidean
    logarithmic map according to a reference matrix
    :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{new} = \log(\mathbf{X}) - \log(\mathbf{C}_\text{ref})

    Parameters
    ----------
    X : ndarray, shape (..., n_channels, n_channels)
        SPD matrices.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n_channels, n_channels)
        SPD matrices projected in tangent space.

    Notes
    -----
    .. versionadded:: 0.3.1
    """
    _check_dimensions(X, Cref)
    return logm(X) - logm(Cref)


def log_map_riemann(X, Cref, C12=False):
    r"""Project SPD matrices in tangent space by Riemannian logarithmic map.

    The projection of a SPD matrix X in the tangent space by Riemannian
    logarithmic map according to a reference matrix
    :math:`\mathbf{C}_\text{ref}` is:

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
    X : ndarray, shape (..., n_channels, n_channels)
        SPD matrices.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.
    C12 : bool, default=False
        If True, it returns the full Riemannian logarithmic map.

    Returns
    -------
    X_new : ndarray, shape (..., n_channels, n_channels)
        SPD matrices projected in tangent space.

    Notes
    -----
    .. versionadded:: 0.3.1
    """
    _check_dimensions(X, Cref)
    Cm12 = invsqrtm(Cref)
    X_new = logm(Cm12 @ X @ Cm12)
    if C12:
        C12 = sqrtm(Cref)
        X_new = C12 @ X_new @ C12
    return X_new


def upper(X):
    r"""Return the weighted upper triangular part of symmetric matrices.

    This function computes the minimal representation of a matrix in tangent
    space [1]_: it keeps the upper triangular part of the symmetric matrix and
    vectorizes it by applying unity weight for diagonal elements and
    :math:`\sqrt(2)` weight for out-of-diagonal elements.

    Parameters
    ----------
    X : ndarray, shape (..., n_channels, n_channels)
        Symmetric matrices.

    Returns
    -------
    T : ndarray, shape (..., n_channels * (n_channels + 1) / 2)
        Weighted upper triangular parts of symmetric matrices.

    Notes
    -----
    .. versionadded:: 0.3.1

    References
    ----------
    .. [1] O. Tuzel, F. Porikli, P. Meer, "Pedestrian detection via
        classification on Riemannian manifolds", IEEE Trans Pattern Anal Mach
        Intell, 2008
    """
    n_channels = X.shape[-1]
    idx = np.triu_indices_from(np.empty((n_channels, n_channels)))
    coeffs = (np.sqrt(2) * np.triu(np.ones((n_channels, n_channels)), 1) +
              np.eye(n_channels))[idx]
    T = coeffs * X[..., idx[0], idx[1]]
    return T


def unupper(T):
    """Inverse upper function.

    This function is the inverse of upper function: it computes symmetric
    matrices from their weighted upper triangular parts.

    Parameters
    ----------
    T : ndarray, shape (..., n_channels * (n_channels + 1) / 2)
        Weighted upper triangular parts of symmetric matrices.

    Returns
    -------
    X : ndarray, shape (..., n_channels, n_channels)
        Symmetric matrices.

    See Also
    --------
    upper

    Notes
    -----
    .. versionadded:: 0.3.1
    """
    dims = T.shape
    n_channels = int((np.sqrt(1 + 8 * dims[-1]) - 1) / 2)
    X = np.empty((*dims[:-1], n_channels, n_channels))
    idx = np.triu_indices_from(np.empty((n_channels, n_channels)))
    X[..., idx[0], idx[1]] = T
    idx = np.triu_indices_from(np.empty((n_channels, n_channels)), k=1)
    X[..., idx[0], idx[1]] /= np.sqrt(2)
    X[..., idx[1], idx[0]] = X[..., idx[0], idx[1]]
    return X


def tangent_space(X, Cref, *, metric='riemann'):
    """Transform SPD matrices into tangent vectors.

    Transform SPD matrices into tangent vectors, according to a reference
    matrix Cref and to a specific logarithmic map.

    Parameters
    ----------
    X : ndarray, shape (..., n_channels, n_channels)
        SPD matrices.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.
    metric : string, default='riemann'
        The metric used for logarithmic map, can be: 'euclid', 'logeuclid',
        'riemann'.

    Returns
    -------
    T : ndarray, shape (..., n_channels * (n_channels + 1) / 2)
        Tangent vectors.

    See Also
    --------
    log_map_euclid
    log_map_logeuclid
    log_map_riemann
    upper
    """
    options = {
        'euclid': log_map_euclid,
        'logeuclid': log_map_logeuclid,
        'riemann': log_map_riemann,
    }
    X_ = options[metric](X, Cref)
    T = upper(X_)

    return T


def untangent_space(T, Cref, *, metric='riemann'):
    """Transform tangent vectors back to SPD matrices.

    Transform tangent vectors back to SPD matrices, according to a reference
    matrix Cref and to a specific exponential map.

    Parameters
    ----------
    T : ndarray, shape (..., n_channels * (n_channels + 1) / 2)
        Tangent vectors.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.
    metric : string, default='riemann'
        The metric used for exponential map, can be: 'euclid', 'logeuclid',
        'riemann'.

    Returns
    -------
    X : ndarray, shape (..., n_channels, n_channels)
        SPD matrices.

    See Also
    --------
    unupper
    exp_map_euclid
    exp_map_logeuclid
    exp_map_riemann
    """
    X_ = unupper(T)
    options = {
        'euclid': exp_map_euclid,
        'logeuclid': exp_map_logeuclid,
        'riemann': exp_map_riemann,
    }
    X = options[metric](X_, Cref)

    return X


###############################################################################


def transport(Covs, Cref, metric='riemann'):
    """Parallel transport of a set of SPD matrices towards a reference matrix.

    Parameters
    ----------
    Covs : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.
    metric : string, default='riemann'
        The metric used for mean, can be: 'euclid', 'logeuclid', 'riemann'.

    Returns
    -------
    out : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of transported SPD matrices.
    """
    C = mean_covariance(Covs, metric=metric)
    iC = invsqrtm(C)
    E = sqrtm(iC @ Cref @ iC)
    out = E @ Covs @ E.T
    return out
