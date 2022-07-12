""" Tangent space for SPD matrices. """

import numpy as np

from .base import sqrtm, invsqrtm, logm, expm
from .mean import mean_covariance


def tangent_space(X, Cref):
    """Project a set of SPD matrices in the tangent space.

    Project a set of SPD matrices in the tangent space, according to the
    reference matrix Cref.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.

    Returns
    -------
    X_new : ndarray, shape (n_matrices, n_channels * (n_channels + 1) / 2)
        Set of tangent space vectors.

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
    X_new = coeffs * tmp[..., idx[0], idx[1]]
    return X_new


def untangent_space(X, Cref):
    """Project a set of tangent space vectors back to the manifold.

    Project a set of tangent space vectors back to the manifold, according to
    the reference matrix Cref.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels * (n_channels + 1) / 2)
        Set of tangent space vectors.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.

    Returns
    -------
    X_original : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.

    See Also
    --------
    tangent_space
    """
    n_matrices, n_ts = X.shape
    n_channels = int((np.sqrt(1 + 8 * n_ts) - 1) / 2)
    idx = np.triu_indices_from(Cref)
    X_original = np.empty((n_matrices, n_channels, n_channels))
    X_original[..., idx[0], idx[1]] = X
    idx = np.triu_indices_from(Cref, k=1)
    X_original[..., idx[0], idx[1]] /= np.sqrt(2)
    X_original[..., idx[1], idx[0]] = X_original[..., idx[0], idx[1]]

    C12 = sqrtm(Cref)
    X_original = C12 @ expm(X_original) @ C12
    return X_original


def transport(X, Cref, metric='riemann'):
    """Parallel transport of a set of SPD matrices towards a reference matrix.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.
    metric : string, default='riemann'
        The metric used for mean, can be: 'euclid', 'logeuclid', 'riemann'.

    Returns
    -------
    X_new : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of transported SPD matrices.
    """
    C = mean_covariance(X, metric=metric)
    iC = invsqrtm(C)
    E = sqrtm(iC @ Cref @ iC)
    X_new = E @ X @ E.T
    return X_new
