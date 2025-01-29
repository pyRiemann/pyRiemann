"""Kernels for SPD matrices."""

import numpy as np

from .base import invsqrtm, logm
from .mean import mean_riemann
from .utils import check_function


def kernel_euclid(X, Y=None, *, Cref=None, reg=1e-10):
    r"""Euclidean kernel between two sets of matrices.

    Euclidean kernel matrix :math:`\mathbf{K}` of two sets
    :math:`\mathbf{X}` and :math:`\mathbf{Y}` of matrices in
    :math:`\mathbb{R}^{n \times m}` at :math:`\mathbf{C}_\text{ref}`
    is calculated with pairwise inner products:

    .. math::
        \mathbf{K}_{i,j} = \text{tr}(
        (\mathbf{X}_i - \mathbf{C}_\text{ref})^T
        (\mathbf{Y}_j - \mathbf{C}_\text{ref})
        )

    If :math:`\mathbf{C}_\text{ref}` is None [1]_:

    .. math::
        \mathbf{K}_{i,j} = \text{tr}(\mathbf{X}_i^T \mathbf{Y}_j)

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, m)
        First set of matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, m), default=None
        Second set of matrices. If None, Y is set to X.
    Cref : None | ndarray, shape (n, m), default=None
        Reference matrix.
        If None, Cref is defined as null matrix.

        .. versionadded:: 0.8
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        Euclidean kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    kernel

    References
    ----------
    .. [1] `A linear feature space for simultaneous learning of spatio-spectral
        filters in BCI
        <https://doi.org/10.1016/j.neunet.2009.06.035>`_
        J. Farquhar. Neural Networks, 2009
    """
    def kernelfct(X, Cref):
        if Cref is None:
            return X, Cref
        else:
            return X - Cref, Cref

    return _apply_matrix_kernel(kernelfct, X, Y, Cref=Cref, reg=reg)


def kernel_logeuclid(X, Y=None, *, Cref=None, reg=1e-10):
    r"""Log-Euclidean kernel between two sets of SPD matrices.

    Log-Euclidean kernel matrix :math:`\mathbf{K}` of two sets
    :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD matrices in
    :math:`\mathbb{R}^{n \times n}` on tangent space at
    :math:`\mathbf{C}_\text{ref}` is calculated with pairwise inner products
    [1]_:

    .. math::
        \mathbf{K}_{i,j} = \text{tr}(
        (\log(\mathbf{X}_i) - \log(\mathbf{C}_\text{ref}))
        (\log(\mathbf{Y}_j) - \log(\mathbf{C}_\text{ref}))
        )

    If :math:`\mathbf{C}_\text{ref}` is None [2]_:

    .. math::
        \mathbf{K}_{i,j} = \text{tr}(\log(\mathbf{X}_i) \log(\mathbf{Y}_j))

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    Cref : None | ndarray, shape (n, n), default=None
        Reference SPD matrix.
        If None, Cref is defined as identity matrix.

        .. versionadded:: 0.8
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        Log-Euclidean kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.3
    .. versionchanged:: 0.8
        Add parameter Cref to use a reference matrix.

    See Also
    --------
    kernel

    References
    ----------
    .. [1] `A New Canonical Log-Euclidean Kernel for Symmetric Positive
        Definite Matrices for EEG Analysis
        <https://ieeexplore.ieee.org/iel8/10/4359967/10735221.pdf>`_
        G. L. W. vom Berg, V. Rohr, D. Platt and B. Blankertz.
        IEEE Transactions on Biomedical Engineering, 2024
    .. [2] `Factor analysis based spatial correlation modeling for speaker
        verification
        <https://ieeexplore.ieee.org/abstract/document/5684490>`_
        E. Wang, W. Guo, L. Dai, K. Lee, B. Ma and H. Li. IEEE ISCSLP, 2010
    """
    def kernelfct(X, Cref):
        if Cref is None:
            return logm(X), Cref
        else:
            return logm(X) - logm(Cref), Cref

    return _apply_matrix_kernel(kernelfct, X, Y, Cref=Cref, reg=reg)


def kernel_riemann(X, Y=None, *, Cref=None, reg=1e-10):
    r"""Affine-invariant Riemannian kernel between two sets of SPD matrices.

    Affine-invariant Riemannian kernel matrix :math:`\mathbf{K}` of two sets
    :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD matrices in
    :math:`\mathbb{R}^{n \times n}` on tangent space at
    :math:`\mathbf{C}_\text{ref}` is calculated with pairwise inner products
    [1]_:

    .. math::
        \mathbf{K}_{i,j} = \text{tr}(
        \log( \mathbf{C}_\text{ref}^{-1/2}
        \mathbf{X}_i \mathbf{C}_\text{ref}^{-1/2} )
        \log( \mathbf{C}_\text{ref}^{-1/2} \mathbf{Y}_j
        \mathbf{C}_\text{ref}^{-1/2}) )

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    Cref : None | ndarray, shape (n, n), default=None
        Reference SPD matrix.
        If None, Cref is calculated as the Riemannian mean of X, see
        :func:`pyriemann.utils.mean.mean_riemann`.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        Affine-invariant Riemannian kernel matrix between X and Y.

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
    """
    def kernelfct(X, Cref):
        if Cref is None:
            Cref = mean_riemann(X)

        C_invsq = invsqrtm(Cref)
        X_ = logm(C_invsq @ X @ C_invsq)
        return X_, Cref

    return _apply_matrix_kernel(kernelfct, X, Y, Cref=Cref, reg=reg)


###############################################################################


def _check_dimensions(X, Y, Cref):
    """Check for matching dimensions in X, Y and Cref."""
    if not isinstance(Y, type(None)):
        assert Y.shape[1:] == X.shape[1:], "Dimension of matrices in Y must "\
                                           "match dimension of matrices in " \
                                           f"X. Expected {X.shape[1:]}, got " \
                                           f"{Y.shape[1:]}."

    if not isinstance(Cref, type(None)):
        assert Cref.shape == X.shape[1:], "Dimension of Cref must match " \
                                          "dimension of matrices in X. " \
                                          f"Expected {X.shape[1:]}, got " \
                                          f"{Cref.shape}."


def _apply_matrix_kernel(kernel_fct, X, Y=None, *, Cref=None, reg=1e-10):
    """Apply a matrix kernel function."""
    _check_dimensions(X, Y, Cref)
    n_matrices_X, n, n = X.shape

    X_, Cref = kernel_fct(X, Cref)

    if Y is None or np.array_equal(X, Y):
        Y_ = X_
    else:
        Y_, _ = kernel_fct(Y, Cref)

    # calculate scalar products: K[i,j] = np.trace(X_[i]^T @ Y_[j])
    X_T = X_.transpose((0, 2, 1))
    K = np.einsum('acb,dbc->ad', X_T, Y_, optimize=True)

    # regularization due to numerical errors
    if np.array_equal(X_, Y_):
        K.flat[:: n_matrices_X + 1] += reg

    return K


kernel_functions = {
    "euclid": kernel_euclid,
    "logeuclid": kernel_logeuclid,
    "riemann": kernel_riemann,
}


def kernel(X, Y=None, *, Cref=None, metric="riemann", reg=1e-10):
    r"""Kernel matrix between matrices according to a specified metric.

    It calculates the kernel matrix :math:`\mathbf{K}` of pairwise inner
    products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}`
    of matrices on the tangent space at :math:`\mathbf{C}_\text{ref}`,
    according to a specified metric.

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of matrices. If None, Y is set to X.
    Cref : None | ndarray, shape (n, n), default=None
        Reference matrix.
    metric : string | callable, default="riemann"
        Metric used for tangent space and mean estimation, can be:
        "euclid", "logeuclid", "riemann", or a callable function.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation, to provide a positive-definite kernel matrix.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        Kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    kernel_euclid
    kernel_logeuclid
    kernel_riemann
    """
    kernel_function = check_function(metric, kernel_functions)
    K = kernel_function(X, Y, Cref=Cref, reg=reg)
    return K
