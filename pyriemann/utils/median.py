"""Medians of SPD/HPD matrices."""

import warnings

import numpy as np

from .base import sqrtm, invsqrtm, logm, expm
from .distance import distance
from .mean import mean_euclid
from .utils import check_weights


def median_euclid(X, *, tol=10e-6, maxiter=50, init=None, weights=None):
    r"""Euclidean geometric median of matrices.

    The Euclidean geometric median minimizes the sum of Euclidean distances
    :math:`d_E` to all matrices [1]_ [2]_:

    .. math::
        \arg \min_{\mathbf{M}} \sum_i w_i \ d_E (\mathbf{M}, \mathbf{X}_i)

    Geometric median is different from the marginal median provided by NumPy
    [3]_.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, m)
        Set of matrices.
    tol : float, default=10e-6
        The tolerance to stop the iterative algorithm.
    maxiter : int, default=50
        The maximum number of iterations.
    init : None | ndarray, shape (n, m), default=None
        A matrix used to initialize the iterative algorithm.
        If None, the weighted Euclidean mean is used.
    weights : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (n, m)
        Euclidean geometric median.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] `Sur le point pour lequel la somme des distances de n points donnés
        est minimum
        <https://www.jstage.jst.go.jp/article/tmj1911/43/0/43_0_355/_pdf>`_
        E Weiszfeld. Tohoku Mathematical Journal, 1937, 43, pp. 355-386.
    .. [2] `The multivariate L1-median and associated data depth
        <https://www.pnas.org/doi/pdf/10.1073/pnas.97.4.1423>`_
        Y Vardi and C-H Zhan. Proceedings of the National Academy of Sciences,
        2000, vol. 97, no 4, p. 1423-1426
    .. [3] https://numpy.org/doc/stable/reference/generated/numpy.median.html
    """
    n_matrices, _, _ = X.shape
    weights = check_weights(weights, n_matrices)
    if init is None:
        M = mean_euclid(X, sample_weight=weights)
    else:
        M = init

    for _ in range(maxiter):
        dists = distance(X, M, metric="euclid")[:, 0]
        is_zero = (dists == 0)

        w = weights[~is_zero] / dists[~is_zero]
        Mnew = mean_euclid(X[~is_zero], sample_weight=w)  # Eq(2.4) of [2]

        n_zeros = np.sum(is_zero)
        if n_zeros > 0:
            R = np.einsum("a,abc->bc", w, X[~is_zero] - M)  # Eq(2.7)
            r = np.linalg.norm(R, ord="fro")
            rinv = 0 if r == 0 else np.mean(weights[is_zero]) / r
            Mnew = max(0, 1 - rinv) * Mnew + min(1, rinv) * M  # Eq(2.6)

        crit = np.linalg.norm(Mnew - M, ord="fro")
        M = Mnew
        if crit <= tol:
            break
    else:
        warnings.warn("Convergence not reached")

    return M


def median_riemann(
    X,
    *,
    tol=10e-6,
    maxiter=50,
    init=None,
    weights=None,
    step_size=1,
):
    r"""Affine-invariant Riemannian geometric median of SPD/HPD matrices.

    The affine-invariant Riemannian geometric median minimizes the sum of
    affine-invariant Riemannian distances :math:`d_R` to all SPD/HPD matrices
    [1]_:

    .. math::
        \arg \min_{\mathbf{M}} \sum_i w_i \ d_R (\mathbf{M}, \mathbf{X}_i)

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices.
    tol : float, default=10e-6
        The tolerance to stop the gradient descent.
    maxiter : int, default=50
        The maximum number of iterations.
    init : None | ndarray, shape (n, n), default=None
        A SPD/HPD matrix used to initialize the gradient descent.
        If None, the weighted Euclidean mean is used.
    weights : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.
    step_size : float, default=1.0
        The step size of the gradient descent, in (0,2].

    Returns
    -------
    M : ndarray, shape (n, n)
        Affine-invariant Riemannian geometric median.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] `The geometric median on Riemannian manifolds with application to
        robust atlas estimation
        <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2735114/>`_
        PT. Fletcher, S. Venkatasubramanian S and S. Joshi.
        NeuroImage, 2009, 45(1), S143-S152
    .. [2] `Riemannian median, geometry of covariance matrices and radar target
        detection
        <https://ieeexplore.ieee.org/abstract/document/5615027>`_
        L Yang, M Arnaudon and F Barbaresco. 7th European Radar Conference,
        2010, pp. 415-418
    """
    if not 0 < step_size <= 2:
        raise ValueError(
            f"Value step_size must be included in (0, 2] (Got {step_size})"
        )
    n_matrices, _, _ = X.shape
    weights = check_weights(weights, n_matrices)
    if init is None:
        M = mean_euclid(X, sample_weight=weights)
    else:
        M = init

    for _ in range(maxiter):
        dists = distance(X, M, metric="riemann")[:, 0]
        is_zero = (dists == 0)
        w = weights[~is_zero] / dists[~is_zero]

        # Eq(11) of [1]
        M12, Mm12 = sqrtm(M), invsqrtm(M)
        tangvecs = logm(Mm12 @ X[~is_zero] @ Mm12)
        J = np.einsum("a,abc->bc", w / np.sum(w), tangvecs)
        M = M12 @ expm(step_size * J) @ M12

        crit = np.linalg.norm(J, ord="fro")
        if crit <= tol:
            break
    else:
        warnings.warn("Convergence not reached")

    return M
