"""Kernels for SPD matrices."""

import numpy as np

from .base import invsqrtm, logm
from .mean import mean_riemann


def kernel_euclid(X, Y=None, *, reg=1e-10, **kwargs):
    r"""Euclidean kernel between two sets of SPD matrices.

    Calculates the Euclidean kernel matrix K of inner products of two sets
    X and Y of SPD matrices by calculating pairwise:

    .. math::
        K_{i,j} = \text{tr}(X_i Y_j)

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n_channels, n_channels)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n_channels, n_channels), default=None
        Second set of SPD matrices. If None, Y is set to X.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The Euclidean kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    kernel
    """  # noqa
    def kernelfct(X, Cref):
        return X

    return _apply_matrix_kernel(kernelfct, X, Y, reg=reg)


def kernel_logeuclid(X, Y=None, *, reg=1e-10, **kwargs):
    r"""Log-Euclidean kernel between two sets of SPD matrices.

    Calculates the Log-Euclidean kernel matrix K of inner products of two sets
    X and Y of SPD matrices by calculating pairwise [1]_:

    .. math::
        K_{i,j} = \text{tr}(\log(X_i) \log(Y_j))

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n_channels, n_channels)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n_channels, n_channels), default=None
        Second set of SPD matrices. If None, Y is set to X.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The Log-Euclidean kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    kernel

    References
    ----------
    .. [1] `Classification of covariance matrices using a Riemannian-based
        kernel for BCI applications
        <https://hal.archives-ouvertes.fr/hal-00820475/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. Neurocomputing,
        Elsevier, 2013, 112, pp.172-178.
    """  # noqa
    def kernelfct(X, Cref):
        return logm(X)

    return _apply_matrix_kernel(kernelfct, X, Y, reg=reg)


def kernel_riemann(X, Y=None, *, Cref=None, reg=1e-10):
    r"""Affine-invariant Riemannian kernel between two sets of SPD matrices.

    Calculates the affine-invariant Riemannian kernel matrix K of inner
    products of two sets X and Y of SPD matrices on tangent space of Cref by
    calculating pairwise [1]_:

    .. math::
        K_{i,j} = \text{tr}(\log(C_\text{ref}^{-1/2} X_i C_\text{ref}^{-1/2})
        \log(C_\text{ref}^{-1/2} Y_j C_\text{ref}^{-1/2}) )

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n_channels, n_channels)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n_channels, n_channels), default=None
        Second set of SPD matrices. If None, Y is set to X.
    Cref : None | ndarray, shape (n_channels, n_channels), default=None
        Reference point for the tangent space and inner product calculation.
        If None, Cref is calculated as the Riemannian mean of X.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The affine-invariant Riemannian kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    kernel

    References
    ----------
    .. [1] `Classification of covariance matrices using a Riemannian-based
        kernel for BCI applications
        <https://hal.archives-ouvertes.fr/hal-00820475/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. Neurocomputing,
        Elsevier, 2013, 112, pp.172-178.
    """  # noqa
    def kernelfct(X, Cref):
        if Cref is None:
            Cref = mean_riemann(X)

        C_invsq = invsqrtm(Cref)
        X_ = logm(C_invsq @ X @ C_invsq)
        return X_

    return _apply_matrix_kernel(kernelfct, X, Y, Cref=Cref, reg=reg)


###############################################################################


def _check_dimensions(X, Y, Cref):
    """Check for matching dimensions in X, Y and Cref."""
    if not isinstance(Y, type(None)):
        assert Y.shape[1:] == X.shape[1:], f"Dimension of matrices in Y must "\
                                           f"match dimension of matrices in " \
                                           f"X. Expected {X.shape[1:]}, got " \
                                           f"{Y.shape[1:]}."

    if not isinstance(Cref, type(None)):
        assert Cref.shape == X.shape[1:], f"Dimension of Cref must match " \
                                          f"dimension of matrices in X. " \
                                          f"Expected {X.shape[1:]}, got " \
                                          f"{Cref.shape}."


def _apply_matrix_kernel(kernel_fct, X, Y=None, *, Cref=None, reg=1e-10):
    """Apply a matrix kernel function."""
    _check_dimensions(X, Y, Cref)
    n_matrices_X, n_channels, n_channels = X.shape

    X_ = kernel_fct(X, Cref)

    if isinstance(Y, type(None)) or np.array_equal(X, Y):
        Y_ = X_

    else:
        Y_ = kernel_fct(Y, Cref)

    # calculate scalar products
    # for i in range(n_matrices_X):
    #     for j in range(n_matrices_Y):
    #         K[i][j] = np.trace(X_[i] @ Y_[j])
    # einsum does same as that just a looooooot faster

    K = np.einsum('abc,dbc->ad', X_, Y_, optimize=True)

    # regularization due to numerical errors
    if np.array_equal(X_, Y_):
        K.flat[:: n_matrices_X + 1] += reg

    return K


def kernel(X, Y=None, *, Cref=None, metric='riemann', reg=1e-10):
    """Kernel matrix between SPD matrices according to a specified metric.

    Calculates the kernel matrix K of inner products of two sets X and Y of
    SPD matrices on the tangent space of Cref according to a specified metric.

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n_channels, n_channels)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n_channels, n_channels), \
            default=None
        Second set of SPD matrices. If None, Y is set to X.
    Cref : None | ndarray, shape (n_channels, n_channels), default=None
        Reference point for the tangent space and inner product
        calculation. Only used if metric='riemann'.
    metric : {'euclid', 'logeuclid', 'riemann'}, default='riemann'
        The type of metric used for tangent space and mean estimation.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation, to provide a positive-definite kernel matrix.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    kernel_euclid
    kernel_logeuclid
    kernel_riemann
    """
    try:
        return globals()[f'kernel_{metric}'](X, Y, Cref=Cref, reg=reg)
    except KeyError:
        raise ValueError("Kernel metric must be 'euclid', 'logeuclid', or "
                         "'riemann'.")
