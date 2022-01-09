import numpy as np

from .mean import mean_riemann
from .base import invsqrtm, logm


def kernel(X, Y=None, Cref=None, metric='riemann', reg=1e-10):
    r""" Calculates the Kernel matrix K of inner products of two sets
         X and Y of SPD matrices on tangent space of C according to the
         specified metric.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            First set of SPD matrices.
        Y : None | ndarray, shape (n_matrices, n_channels, n_channels)
            Second set of SPD matrices. If None, Y is set to X.
        Cref : None | ndarray, shape (n_channels, n_channels)
            Reference point for the tangent space and inner product
            calculation. If None, Cref is calculated as the Riemannian mean of
            X according to the specified metric.
        metric : {'riemann'}
            The type of metric used for tangent space and mean estimation. Can
            be 'riemann'.
        reg : float, (default : 1e-10)
            Regularization parameter to mitigate numerical errors in kernel
            matrix estimation, to provide a positive-definite kernel matrix.

        Returns
        ----------
        K : ndarray, shape (n_matrices, n_matrices)
            The kernel matrix of X and Y.

        Notes
        -----
        .. versionadded:: 0.2.8
        """
    if metric == 'riemann':
        return kernel_riemann(X, Y, Cref, reg=reg)
    else:
        raise ValueError("Kernel metric must be 'riemann'.")


def kernel_riemann(X, Y=None, Cref=None, reg=1e-10):
    r""" Calculates the Kernel matrix K of inner products of two sets
     X and Y of SPD matrices on tangent space of C by calculating pairwise

    .. math::
        K_{i,j} =
        {tr}(\log(C^{-1/2}X_i C^{-1/2})\log(C^{-1/2}Y_j C^{-1/2}))

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices, n_channels, n_channels)
        Second set of SPD matrices. If None, Y is set to X.
    Cref : None | ndarray, shape (n_channels, n_channels)
        Reference point for the tangent space and inner product calculation.
        If None, Cref is calculated as the Riemannian mean of X.
    reg : float, (default : 1e-10)
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    ----------
    K : ndarray, shape (n_matrices, n_matrices)
        The kernel matrix of X and Y.

    Notes
    -----
    .. versionadded:: 0.2.8
    """
    if Cref is None:
        Cref = mean_riemann(X)

    G_invsq = invsqrtm(Cref)

    n_matrices_X, n_channels, n_channels = X.shape

    X_ = np.matmul(G_invsq, np.matmul(X, G_invsq))
    X_ = np.array([logm(x_) for x_ in X_])

    if isinstance(Y, type(None)) or np.array_equal(X, Y):
        Y_ = X_

    else:
        n_matrices_Y, n_channels, n_channels = Y.shape

        Y_ = G_invsq @ Y @ G_invsq
        Y_ = np.array([logm(y_) for y_ in Y_])

    # calculate scalar products
    # for i in range(n_matrices_X):
    #     for j in range(n_matrices_Y):
    #         K[i][j] = np.trace(X_[i] @ Y_[j])
    # einsum does same as that just a looooooot faster

    K = np.einsum('abc,dbc->ad', X_, Y_)

    # regularization due to numerical errors
    if np.array_equal(X_, Y_):
        K.flat[:: n_matrices_X + 1] += reg

    return K
