""" Tangent space for SPD matrices. """

import numpy as np

from .base import sqrtm, invsqrtm, logm, expm
from .mean import mean_covariance


def tangent_space(X, Cref):
    """Project SPD matrices in the tangent space.

    Project SPD matrices in the tangent space, according to the reference
    matrix Cref.

    Parameters
    ----------
    X : ndarray, shape (..., n_channels, n_channels)
        SPD matrices.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.

    Returns
    -------
    T : ndarray, shape (..., n_channels * (n_channels + 1) / 2)
        Tangent vectors.

    See Also
    --------
    untangent_space
    """
    Cm12 = invsqrtm(Cref)
    tmp = logm(Cm12 @ X @ Cm12)

    n_channels = X.shape[-1]
    idx = np.triu_indices_from(Cref)
    coeffs = (np.sqrt(2) * np.triu(np.ones((n_channels, n_channels)), 1) +
              np.eye(n_channels))[idx]
    T = coeffs * tmp[..., idx[0], idx[1]]
    return T


def untangent_space(T, Cref):
    """Project tangent vectors back to the manifold.

    Project tangent vectors back to the matrix manifold, according to the
    reference matrix Cref.

    Parameters
    ----------
    T : ndarray, shape (..., n_channels * (n_channels + 1) / 2)
        Tangent vectors.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.

    Returns
    -------
    X : ndarray, shape (..., n_channels, n_channels)
        SPD matrices.

    See Also
    --------
    tangent_space
    """
    dims = T.shape
    n_channels = int((np.sqrt(1 + 8 * dims[-1]) - 1) / 2)
    X = np.empty((*dims[:-1], n_channels, n_channels))
    idx = np.triu_indices_from(Cref)
    X[..., idx[0], idx[1]] = T
    idx = np.triu_indices_from(Cref, k=1)
    X[..., idx[0], idx[1]] /= np.sqrt(2)
    X[..., idx[1], idx[0]] = X[..., idx[0], idx[1]]

    C12 = sqrtm(Cref)
    X = C12 @ expm(X) @ C12
    return X


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
