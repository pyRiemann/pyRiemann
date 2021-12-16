import numpy as np

from .mean import mean_riemann
from .base import sqrtm, invsqrtm, powm, logm, expm


def inner_product(A, B, Cref):
    """Inner product of symmetric matrices A and B on the tangent space of Cref.

    Parameters
    ----------
    A : ndarray, shape (n_channels, n_channels)
        First symmetric matrix to calculate inner product.
    B : ndarray, shape (n_channels, n_channels)
        Second symmetric matrix to calculate inner product.
    Cref : ndarray, shape (n_channels, n_channels)
        Reference point for the tangent space and inner product calculation.

    Notes
    -----
    .. versionadded:: 0.2.8
    """
    Cref_inv = powm(Cref, -1)
    return np.trace(Cref_inv@A@Cref_inv@B)


def kernel_riemann(X, Y=None, Cref=None):
    r""" Calculates the Kernel matrix K of inner products of two sets
     X and Y of SPD matrices on tangent space of C by calculating pairwise

    .. math::
        K_{i,j} =
        {tr}(\log(C^{-1/2}X_i C^{-1/2})\log(C^{-1/2}Y_j C^{-1/2}))

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        First set of SPD matrices.
    Y : ndarray, shape (n_matrices, n_channels, n_channels), optional
        Second set of SPD matrices. If None, Y is set to X.
    Cref : ndarray, shape (n_channels, n_channels), optional
        Reference point for the tangent space and inner product calculation.
        If None, Cref is calculated as the Riemannian mean of X.

    Returns
    ----------
    K : ndarray, shape (n_matrices, n_matrices)
        The kernel matrix of X and Y.

    Notes
    -----
    .. versionadded:: 0.2.8
    """
    if Cref is None:
        G = mean_riemann(X)
        G_invsq = invsqrtm(G)

    else:
        G_invsq = invsqrtm(Cref)

    n_matrices_X, n_channels, n_channels = X.shape

    # tangent space projection
    X_ = np.matmul(G_invsq, np.matmul(X, G_invsq))
    X_ = np.array([logm(X_[index]) for index in range(n_matrices_X)])

    if isinstance(Y, type(None)):
        Y_ = X_

    else:
        n_matrices_Y, n_channels, n_channels = Y.shape

        Y = np.matmul(G_invsq, np.matmul(Y, G_invsq))
        Y_ = np.array([logm(Y[index]) for index in range(n_matrices_Y)])

    # calculate scalar products
    # for i in range(n_matrices_X):
    #     for j in range(n_matrices_Y):
    #         res[i][j] = np.trace(X_[i] @ Y_[j])
    # einsum does same as that just a looooooot faster

    res = np.einsum('abc,dbc->ad', X_, Y_)

    return res
