"""Medians of SPD matrices."""

import warnings
import numpy as np

from .base import sqrtm, invsqrtm, logm, expm
from .distance import pairwise_distance
from .mean import _get_sample_weight, mean_euclid


def median_euclid(X, *, tol=10e-6, maxiter=50, init=None, weights=None):
    r"""Euclidean geometric median of matrices.

    The Euclidean geometric median minimizes the sum of Euclidean distances
    :math:`d_E` to all matrices [1]_ [2]_:

    .. math::
        \arg \min_{\mathbf{M}} \sum_i w_i d_E (\mathbf{M}, \mathbf{X}_i)

    It is different from the marginal median provided by NumPy [3]_.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of matrices.
    tol : float, default=10e-6
        The tolerance to stop the iterative algorithm.
    maxiter : int, default=50
        The maximum number of iterations.
    init : None | ndarray, shape (n_channels, n_channels), default=None
        A matrix used to initialize the iterative algorithm.
        If None, the Euclidean mean is used.
    weights : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (n_channels, n_channels)
        Euclidean geometric median.

    Notes
    -----
    .. versionadded:: 0.3.1

    References
    ----------
    .. [1] Weiszfeld E. "Sur le point pour lequel la somme des distances de n
        points donnés est minimum", Tohoku Math J, 1937
    .. [2] Vardi Y and Zhan C-H. "The multivariate L1-median and associated
        data depth", PNAS, 2000
    .. [3] https://numpy.org/doc/stable/reference/generated/numpy.median.html
    """
    weights = _get_sample_weight(weights, X)
    if init is None:
        M = mean_euclid(X, sample_weight=weights)
    else:
        M = init

    for _ in range(maxiter):
        dists = pairwise_distance(
            X,
            M[np.newaxis, ...],
            metric='euclid'
        )[:, 0]
        is_zero = (dists == 0)

        w = weights[~is_zero] / dists[~is_zero]
        Mnew = mean_euclid(X[~is_zero], sample_weight=w)  # Eq(2.4) of [2]

        n_zeros = np.sum(is_zero)
        if n_zeros > 0:
            R = np.einsum('a,abc->bc', w, X[~is_zero] - M)  # Eq(2.7)
            r = np.linalg.norm(R, ord='fro')
            rinv = 0 if r == 0 else np.mean(weights[is_zero]) / r
            Mnew = max(0, 1 - rinv) * Mnew + min(1, rinv) * M  # Eq(2.6)

        crit = np.linalg.norm(Mnew - M, ord='fro')
        M = Mnew
        if crit <= tol:
            break
    else:
        warnings.warn('Convergence not reached')

    return M


def median_riemann(X, *, tol=10e-6, maxiter=50, init=None, weights=None,
                   step_size=1):
    r"""Affine-invariant Riemannian geometric median of SPD matrices.

    The affine-invariant Riemannian geometric median minimizes the sum of
    affine-invariant Riemannian distances :math:`d_R` to all SPD matrices [1]_:

    .. math::
        \arg \min_{\mathbf{M}} \sum_i w_i d_R (\mathbf{M}, \mathbf{X}_i)

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    tol : float, default=10e-6
        The tolerance to stop the gradient descent.
    maxiter : int, default=50
        The maximum number of iterations.
    init : None | ndarray, shape (n_channels, n_channels), default=None
        A SPD matrix used to initialize the gradient descent.
        If None, the Euclidean mean is used.
    weights : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.
    step_size : float, default=1.0
        The step size of the gradient descent, in (0,2].

    Returns
    -------
    M : ndarray, shape (n_channels, n_channels)
        Affine-invariant Riemannian geometric median.

    Notes
    -----
    .. versionadded:: 0.3.1

    References
    ----------
    .. [1] Fletcher PT, Venkatasubramanian S and Joshi S. "The geometric median
        on Riemannian manifolds with application to robust atlas estimation",
        NeuroImage, 2009
    .. [2] Yang L, Arnaudon M and Barbaresco F. "Riemannian median, geometry of
        covariance matrices and radar target detection", EURAD, 2010
    """
    weights = _get_sample_weight(weights, X)
    if not 0 < step_size <= 2:
        raise ValueError(
            'Value step_size must be included in (0, 2] (Got %d)' % step_size)
    if init is None:
        M = mean_euclid(X, sample_weight=weights)
    else:
        M = init

    for _ in range(maxiter):
        dists = pairwise_distance(
            X,
            M[np.newaxis, ...],
            metric='riemann'
        )[:, 0]
        is_zero = (dists == 0)
        w = weights[~is_zero] / dists[~is_zero]

        # Eq(11) of [1]
        M12, Mm12 = sqrtm(M), invsqrtm(M)
        tangvecs = logm(Mm12 @ X[~is_zero] @ Mm12)
        J = np.einsum('a,abc->bc', w / np.sum(w), tangvecs)
        M = M12 @ expm(step_size * J) @ M12

        crit = np.linalg.norm(J, ord='fro')
        if crit <= tol:
            break
    else:
        warnings.warn('Convergence not reached')

    return M
