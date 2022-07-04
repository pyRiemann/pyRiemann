""" Tangent Space """
import numpy as np

from .base import sqrtm, invsqrtm, logm, expm
from .mean import mean_covariance


def tangent_space(covmats, Cref):
    """Project a set of SPD matrices in the tangent space.

    Project a set of SPD matrices in the tangent space, according to the
    reference matrix Cref.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.

    Returns
    -------
    T : ndarray, shape (n_matrices, n_channels * (n_channels + 1) / 2)
        Set of tangent space vectors.

    See Also
    --------
    untangent_space
    """
    n_matrices, n_channels, _ = covmats.shape
    Cm12 = invsqrtm(Cref)
    idx = np.triu_indices_from(Cref)
    n_ts = int(n_channels * (n_channels + 1) / 2)
    T = np.empty((n_matrices, n_ts))
    coeffs = (np.sqrt(2) * np.triu(np.ones((n_channels, n_channels)), 1) +
              np.eye(n_channels))[idx]
    for i in range(n_matrices):
        tmp = logm(Cm12 @ covmats[i] @ Cm12)
        T[i] = np.multiply(coeffs, tmp[idx])
    return T


def untangent_space(T, Cref):
    """Project a set of tangent space vectors back to the manifold.

    Project a set of tangent space vectors back to the manifold, according to
    the reference matrix Cref.

    Parameters
    ----------
    T : ndarray, shape (n_matrices, n_channels * (n_channels + 1) / 2)
        Set of tangent space vectors.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference SPD matrix.

    Returns
    -------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.

    See Also
    --------
    tangent_space
    """
    n_matrices, n_ts = T.shape
    n_channels = int((np.sqrt(1 + 8 * n_ts) - 1) / 2)
    C12 = sqrtm(Cref)

    idx = np.triu_indices_from(Cref)
    covmats = np.empty((n_matrices, n_channels, n_channels))
    covmats[:, idx[0], idx[1]] = T
    for i in range(n_matrices):
        triuc = np.triu(covmats[i], 1) / np.sqrt(2)
        covmats[i] = (np.diag(np.diag(covmats[i])) + triuc + triuc.T)
        covmats[i] = C12 @ expm(covmats[i]) @ C12

    return covmats


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
    Covs : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of transported SPD matrices.
    """
    C = mean_covariance(Covs, metric=metric)
    iC = invsqrtm(C)
    E = sqrtm(iC @ Cref @ iC)
    out = np.array([E @ c @ E.T for c in Covs])
    return out
