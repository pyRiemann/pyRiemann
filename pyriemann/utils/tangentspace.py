import numpy as np

from .base import sqrtm, invsqrtm, logm, expm
from .mean import mean_covariance
###############################################################
# Tangent Space
###############################################################


def tangent_space(covmats, Cref):
    """Project a set of covariance matrices in the tangent space. according to
    the reference point Cref

    :param covmats: ndarray, shape (n_trials, n_channels, n_channels)
        Covariance matrices set
    :param Cref: ndarray, shape (n_channels, n_channels)
        The reference covariance matrix
    :returns: ndarray, shape (n_trials, n_channels * (n_channels + 1) / 2)
        the Tangent space

    """
    Nt, Ne, Ne = covmats.shape
    Cm12 = invsqrtm(Cref)
    idx = np.triu_indices_from(Cref)
    Nf = int(Ne * (Ne + 1) / 2)
    T = np.empty((Nt, Nf))
    coeffs = (np.sqrt(2) * np.triu(np.ones((Ne, Ne)), 1) +
              np.eye(Ne))[idx]
    for index in range(Nt):
        tmp = np.dot(np.dot(Cm12, covmats[index, :, :]), Cm12)
        tmp = logm(tmp)
        T[index, :] = np.multiply(coeffs, tmp[idx])
    return T


def untangent_space(T, Cref):
    """Project a set of Tangent space vectors back to the manifold.

    :param T: ndarray, shape (n_trials, n_channels * (n_channels + 1) / 2)
        The Tangent space
    :param Cref: ndarray, shape (n_channels, n_channels)
        The reference covariance matrix

    :returns: ndarray, shape (n_trials, n_channels, n_channels)
        A set of Covariance matrix
    """
    Nt, Nd = T.shape
    Ne = int((np.sqrt(1 + 8 * Nd) - 1) / 2)
    C12 = sqrtm(Cref)

    idx = np.triu_indices_from(Cref)
    covmats = np.empty((Nt, Ne, Ne))
    covmats[:, idx[0], idx[1]] = T
    for i in range(Nt):
        triuc = np.triu(covmats[i], 1) / np.sqrt(2)
        covmats[i] = (np.diag(np.diag(covmats[i])) + triuc + triuc.T)
        covmats[i] = expm(covmats[i])
        covmats[i] = np.dot(np.dot(C12, covmats[i]), C12)

    return covmats


def transport(Covs, Cref, metric='riemann'):
    """Parallel transport of two set of covariance matrix.

    """
    C = mean_covariance(Covs, metric=metric)
    iC = invsqrtm(C)
    E = sqrtm(np.dot(np.dot(iC, Cref), iC))
    out = np.array([np.dot(np.dot(E, c), E.T) for c in Covs])
    return out
