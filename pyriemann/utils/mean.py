"""Means of SPD/HPD matrices."""

from copy import deepcopy
import numpy as np
import warnings

from .ajd import ajd_pham
from .base import sqrtm, invsqrtm, logm, expm, powm
from .distance import distance_riemann
from .geodesic import geodesic_riemann
from .utils import check_weights, check_function


def _deprecate_covmats(covmats, X):
    if covmats is not None:
        print("DeprecationWarning: input covmats has been renamed into X and "
              "will be removed in 0.8.0.")
        X = covmats
    return X


def mean_ale(X=None, tol=10e-7, maxiter=50, sample_weight=None, covmats=None):
    """AJD-based log-Euclidean (ALE) mean of SPD matrices.

    Return the mean of a set of SPD matrices using the approximate joint
    diagonalization (AJD) based log-Euclidean (ALE) mean [1]_.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of SPD matrices.
    tol : float, default=10e-7
        The tolerance to stop the gradient descent.
    maxiter : int, default=50
        The maximum number of iterations.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (n, n)
        ALE mean.

    Notes
    -----
    .. versionadded:: 0.2.4

    See Also
    --------
    mean_covariance

    References
    ----------
    .. [1] `Approximate Joint Diagonalization and Geometric Mean of Symmetric
        Positive Definite Matrices
        <https://arxiv.org/abs/1505.07343>`_
        M. Congedo, B. Afsari, A. Barachant, M. Moakher. PLOS ONE, 2015
    """
    X = _deprecate_covmats(covmats, X)
    n_matrices, n, _ = X.shape
    sample_weight = check_weights(sample_weight, n_matrices)

    # init with AJD
    B = ajd_pham(X)[0]

    eye_n = np.eye(n)
    crit = np.inf
    for _ in range(maxiter):
        J = np.einsum('a,abc->bc', sample_weight, logm(B @ X @ B.conj().T))
        delta = np.real(np.diag(expm(J)))
        B = (np.abs(delta) ** -.5)[:, np.newaxis] * B

        crit = distance_riemann(eye_n, np.diag(delta))
        if crit <= tol:
            break
    else:
        warnings.warn("Convergence not reached")

    J = np.einsum('a,abc->bc', sample_weight, logm(B @ X @ B.conj().T))
    A = np.linalg.inv(B)
    M = A @ expm(J) @ A.conj().T
    return M


def mean_alm(X=None, tol=1e-14, maxiter=100, sample_weight=None, covmats=None):
    r"""Ando-Li-Mathias (ALM) mean of SPD/HPD matrices.

    Return the geometric mean recursively [1]_, generalizing from:

    .. math::
        \mathbf{M} = X_1^{\frac{1}{2}} (X_1^{-\frac{1}{2}}X_2^{\frac{1}{2}}
                     X_1^{-\frac{1}{2}})^{\frac{1}{2}} X_1^{\frac{1}{2}}

    and requiring a high number of iterations.
    Extremely slow, due to the recursive formulation.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices.
    tol : float, default=10e-14
        The tolerance to stop the gradient descent.
    maxiter : int, default=100
        The maximum number of iterations.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (n, n)
        ALM mean.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    mean_covariance

    References
    ----------
    .. [1] `Geometric Means
        <https://www.sciencedirect.com/science/article/pii/S0024379503008693>`_
        T. Ando, C.-K. Li, and R. Mathias. Linear Algebra and its Applications.
        Volume 385, July 2004, Pages 305-334.
    """
    X = _deprecate_covmats(covmats, X)
    n_matrices, _, _ = X.shape
    sample_weight = check_weights(sample_weight, n_matrices)

    if n_matrices == 2:
        alpha = sample_weight[1] / sample_weight[0] / 2
        M = geodesic_riemann(X[0], X[1], alpha=alpha)
        return M

    M = X
    M_iter = np.zeros_like(M)
    for _ in range(maxiter):
        for h in range(n_matrices):
            s = np.mod(np.arange(h, h + n_matrices - 1) + 1, n_matrices)
            M_iter[h] = mean_alm(M[s], sample_weight=sample_weight[s])

        norm_iter = np.linalg.norm(M_iter[0] - M[0], 2)
        norm_c = np.linalg.norm(M[0], 2)
        if (norm_iter / norm_c) < tol:
            break
        M = deepcopy(M_iter)
    else:
        warnings.warn("Convergence not reached")

    return M_iter.mean(axis=0)


def mean_euclid(X=None, sample_weight=None, covmats=None):
    r"""Mean of matrices according to the Euclidean metric.

    .. math::
        \mathbf{M} = \sum_i w_i \ \mathbf{X}_i

    This mean is also called arithmetic.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, m)
        Set of matrices.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (n, m)
        Euclidean mean.

    See Also
    --------
    mean_covariance
    """
    X = _deprecate_covmats(covmats, X)
    return np.average(X, axis=0, weights=sample_weight)


def mean_harmonic(X=None, sample_weight=None, covmats=None):
    r"""Harmonic mean of invertible matrices.

    .. math::
        \mathbf{M} = \left( \sum_i w_i \ {\mathbf{X}_i}^{-1} \right)^{-1}

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of invertible matrices.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (n, n)
        Harmonic mean.

    See Also
    --------
    mean_covariance
    """
    X = _deprecate_covmats(covmats, X)
    T = mean_euclid(np.linalg.inv(X), sample_weight=sample_weight)
    M = np.linalg.inv(T)
    return M


def mean_identity(X=None, sample_weight=None, covmats=None):
    r"""Identity matrix corresponding to the matrices dimension.

    .. math::
        \mathbf{M} = \mathbf{I}_n

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of square matrices.
    sample_weight : None
        Not used, here for compatibility with other means.

    Returns
    -------
    M : ndarray, shape (n, n)
        Identity matrix.

    See Also
    --------
    mean_covariance
    """
    X = _deprecate_covmats(covmats, X)
    M = np.eye(X.shape[-1])
    return M


def mean_kullback_sym(X=None, sample_weight=None, covmats=None):
    """Mean of SPD/HPD matrices according to Kullback-Leibler divergence.

    Symmetrized Kullback-Leibler mean is the geometric mean between the
    Euclidean and the harmonic means [1]_.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (n, n)
        Symmetrized Kullback-Leibler mean.

    See Also
    --------
    mean_covariance

    References
    ----------
    .. [1] `Symmetric positive-definite matrices: From geometry to applications
        and visualization
        <https://link.springer.com/chapter/10.1007/3-540-31272-2_17>`_
        M. Moakher and P. Batchelor. Visualization and Processing of Tensor
        Fields, pp. 285-298, 2006
    """
    X = _deprecate_covmats(covmats, X)
    M_euclid = mean_euclid(X, sample_weight=sample_weight)
    M_harmonic = mean_harmonic(X, sample_weight=sample_weight)
    M = geodesic_riemann(M_euclid, M_harmonic, 0.5)
    return M


def mean_logdet(X=None, tol=10e-5, maxiter=50, init=None, sample_weight=None,
                covmats=None):
    r"""Mean of SPD/HPD matrices according to the log-det metric.

    Log-det mean is obtained by an iterative procedure where the update is:

    .. math::
        \mathbf{M} = \left( \sum_i w_i \ \left( 0.5 \mathbf{M}
                     + 0.5 \mathbf{X}_i \right)^{-1} \right)^{-1}

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices.
    tol : float, default=10e-5
        The tolerance to stop the gradient descent.
    maxiter : int, default=50
        The maximum number of iterations.
    init : None | ndarray, shape (n, n), default=None
        A SPD/HPD matrix used to initialize the gradient descent.
        If None, the weighted Euclidean mean is used.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (n, n)
        Log-det mean.

    See Also
    --------
    mean_covariance
    """
    X = _deprecate_covmats(covmats, X)
    n_matrices, _, _ = X.shape
    sample_weight = check_weights(sample_weight, n_matrices)
    if init is None:
        M = mean_euclid(X, sample_weight=sample_weight)
    else:
        M = init

    crit = np.finfo(np.float64).max
    for _ in range(maxiter):
        invX = np.linalg.inv(0.5 * X + 0.5 * M)
        J = np.einsum('a,abc->bc', sample_weight, invX)
        Mnew = np.linalg.inv(J)

        crit = np.linalg.norm(Mnew - M, ord='fro')
        M = Mnew
        if crit <= tol:
            break
    else:
        warnings.warn("Convergence not reached")

    return M


def mean_logeuclid(X=None, sample_weight=None, covmats=None):
    r"""Mean of SPD/HPD matrices according to the log-Euclidean metric.

    Log-Euclidean mean is [1]_:

    .. math::
        \mathbf{M} = \exp{ \left( \sum_i w_i \ \log{\mathbf{X}_i} \right) }

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (n, n)
        Log-Euclidean mean.

    See Also
    --------
    mean_covariance

    References
    ----------
    .. [1] `Geometric means in a novel vector space structure on symmetric
        positive-definite matrices
        <https://epubs.siam.org/doi/abs/10.1137/050637996?journalCode=sjmael>`_
        V. Arsigny, P. Fillard, X. Pennec, and N. Ayache. SIAM Journal on
        Matrix Analysis and Applications. Volume 29, Issue 1 (2007).
    """
    X = _deprecate_covmats(covmats, X)
    M = expm(mean_euclid(logm(X), sample_weight=sample_weight))
    return M


def mean_power(X=None, p=None, *, sample_weight=None, zeta=10e-10, maxiter=100,
               covmats=None):
    r"""Power mean of SPD/HPD matrices.

    Power mean of order p is the solution of [1]_ [2]_:

    .. math::
        \mathbf{M} = \sum_i w_i \ \mathbf{M} \sharp_p \mathbf{X}_i

    where :math:`\mathbf{A} \sharp_p \mathbf{B}` is the geodesic between
    matrices :math:`\mathbf{A}` and :math:`\mathbf{B}`.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices.
    p : float
        Exponent, in [-1,+1]. For p=0, it returns
        :func:`pyriemann.utils.mean.mean_riemann`.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.
    zeta : float, default=10e-10
        Stopping criterion.
    maxiter : int, default=100
        The maximum number of iterations.

    Returns
    -------
    M : ndarray, shape (n, n)
        Power mean.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    mean_covariance

    References
    ----------
    .. [1] `Matrix Power means and the Karcher mean
        <https://www.sciencedirect.com/science/article/pii/S0022123611004101>`_
        Y. Lim and M. Palfia. Journal of Functional Analysis, Volume 262,
        Issue 4, 15 February 2012, Pages 1498-1514.
    .. [2] `Fixed Point Algorithms for Estimating Power Means of Positive
        Definite Matrices
        <https://hal.archives-ouvertes.fr/hal-01500514>`_
        M. Congedo, A. Barachant, and R. Bhatia. IEEE Transactions on Signal
        Processing, Volume 65, Issue 9, pp.2211-2220, May 2017
    """
    X = _deprecate_covmats(covmats, X)
    if p is None:
        raise ValueError("Input p can not be None")
    if not isinstance(p, (int, float)):
        raise ValueError("Power mean only defined for a scalar exponent")
    if p < -1 or 1 < p:
        raise ValueError("Exponent must be in [-1,+1]")

    if p == 1:
        return mean_euclid(X, sample_weight=sample_weight)
    elif p == 0:
        return mean_riemann(X, sample_weight=sample_weight)
    elif p == -1:
        return mean_harmonic(X, sample_weight=sample_weight)

    n_matrices, n, _ = X.shape
    sample_weight = check_weights(sample_weight, n_matrices)
    phi = 0.375 / np.abs(p)

    G = np.einsum('a,abc->bc', sample_weight, powm(X, p))
    if p > 0:
        K = invsqrtm(G)
    else:
        K = sqrtm(G)

    eye_n, sqrt_n = np.eye(n), np.sqrt(n)
    crit = 10 * zeta
    for _ in range(maxiter):
        H = np.einsum(
            'a,abc->bc',
            sample_weight,
            powm(K @ powm(X, np.sign(p)) @ K.conj().T, np.abs(p))
        )
        K = powm(H, -phi) @ K

        crit = np.linalg.norm(H - eye_n) / sqrt_n
        if crit <= zeta:
            break
    else:
        warnings.warn("Convergence not reached")

    M = K.conj().T @ K
    if p > 0:
        M = np.linalg.inv(M)

    return M


def mean_riemann(X=None, tol=10e-9, maxiter=50, init=None, sample_weight=None,
                 covmats=None):
    r"""Mean of SPD/HPD matrices according to the Riemannian metric.

    The affine-invariant Riemannian mean minimizes the sum of squared
    affine-invariant Riemannian distances :math:`d_R` to all SPD/HPD matrices
    [1]_ [2]_:

    .. math::
         \arg \min_{\mathbf{M}} \sum_i w_i \ d_R (\mathbf{M}, \mathbf{X}_i)^2

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices.
    tol : float, default=10e-9
        The tolerance to stop the gradient descent.
    maxiter : int, default=50
        The maximum number of iterations.
    init : None | ndarray, shape (n, n), default=None
        A SPD/HPD matrix used to initialize the gradient descent.
        If None, the weighted Euclidean mean is used.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (n, n)
        Affine-invariant Riemannian mean.

    See Also
    --------
    mean_covariance

    References
    ----------
    .. [1] `Principal geodesic analysis for the study of nonlinear statistics
        of shape
        <https://ieeexplore.ieee.org/document/1318725>`_
        P.T. Fletcher, C. Lu, S. M. Pizer, S. Joshi.
        IEEE Trans Med Imaging, 2004, 23(8), pp. 995-1005
    .. [2] `A differential geometric approach to the geometric mean of
        symmetric positive-definite matrices
        <https://epubs.siam.org/doi/10.1137/S0895479803436937>`_
        M. Moakher. SIAM J Matrix Anal Appl, 2005, 26 (3), pp. 735-747
    """
    X = _deprecate_covmats(covmats, X)
    n_matrices, _, _ = X.shape
    sample_weight = check_weights(sample_weight, n_matrices)
    if init is None:
        M = mean_euclid(X, sample_weight=sample_weight)
    else:
        M = init

    nu = 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max
    for _ in range(maxiter):
        M12, Mm12 = sqrtm(M), invsqrtm(M)
        J = np.einsum('a,abc->bc', sample_weight, logm(Mm12 @ X @ Mm12))
        M = M12 @ expm(nu * J) @ M12

        crit = np.linalg.norm(J, ord='fro')
        h = nu * crit
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu
        if crit <= tol or nu <= tol:
            break
    else:
        warnings.warn("Convergence not reached")

    return M


def mean_wasserstein(X=None, tol=10e-4, maxiter=50, init=None,
                     sample_weight=None, covmats=None):
    r"""Mean of SPD/HPD matrices according to the Wasserstein metric.

    Wasserstein mean is obtained by an iterative procedure where the update is
    [1]_:

    .. math::
        \mathbf{K} = \left( \sum_i w_i \ \left( \mathbf{K} \mathbf{X}_i
                     \mathbf{K} \right)^{1/2} \right)^{1/2}

    with :math:`\mathbf{K} = \mathbf{M}^{1/2}`.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices.
    tol : float, default=10e-4
        The tolerance to stop the gradient descent.
    maxiter : int, default=50
        The maximum number of iterations.
    init : None | ndarray, shape (n, n), default=None
        A SPD/HPD matrix used to initialize the gradient descent.
        If None the Euclidean mean is used.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (n, n)
        Wasserstein mean.

    See Also
    --------
    mean_covariance

    References
    ----------
    .. [1] `Geometric Radar Processing based on Frechet distance: Information
        geometry versus Optimal Transport Theory
        <https://ieeexplore.ieee.org/document/6042179>`_
        F. Barbaresco. 12th International Radar Symposium (IRS), October 2011
    """
    X = _deprecate_covmats(covmats, X)
    n_matrices, _, _ = X.shape
    sample_weight = check_weights(sample_weight, n_matrices)
    if init is None:
        M = mean_euclid(X, sample_weight=sample_weight)
    else:
        M = init
    K = sqrtm(M)

    crit = np.finfo(np.float64).max
    for _ in range(maxiter):
        J = np.einsum('a,abc->bc', sample_weight, sqrtm(K @ X @ K))
        Knew = sqrtm(J)

        crit = np.linalg.norm(Knew - K, ord='fro')
        K = Knew
        if crit <= tol:
            break
    else:
        warnings.warn("Convergence not reached")

    M = K @ K
    return M


###############################################################################


mean_functions = {
    "ale": mean_ale,
    "alm": mean_alm,
    "euclid": mean_euclid,
    "harmonic": mean_harmonic,
    "identity": mean_identity,
    "kullback_sym": mean_kullback_sym,
    "logdet": mean_logdet,
    "logeuclid": mean_logeuclid,
    "riemann": mean_riemann,
    "wasserstein": mean_wasserstein,
}


def mean_covariance(X=None, metric="riemann", sample_weight=None, covmats=None,
                    **kwargs):
    """Mean of matrices according to a metric.

    Compute the mean of a set of matrices according to a metric [1]_.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of matrices.
    metric : string | callable, default="riemann"
        Metric for mean estimation, can be: "ale", "alm", "euclid", "harmonic",
        "identity", "kullback_sym", "logdet", "logeuclid", "riemann",
        "wasserstein", or a callable function.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.
    **kwargs : dict
        The keyword arguments passed to the sub function.

    Returns
    -------
    M : ndarray, shape (n, n)
        Mean of matrices.

    References
    ----------
    .. [1] `Review of Riemannian distances and divergences, applied to
        SSVEP-based BCI
        <https://hal.archives-ouvertes.fr/LISV/hal-03015762v1>`_
        S. Chevallier, E. K. Kalunga, Q. Barthélemy, E. Monacelli.
        Neuroinformatics, Springer, 2021, 19 (1), pp.93-106
    """
    X = _deprecate_covmats(covmats, X)
    mean_function = check_function(metric, mean_functions)
    M = mean_function(
        X,
        sample_weight=sample_weight,
        **kwargs,
    )
    return M


###############################################################################


def _get_mask_from_nan(X):
    nan_col = np.all(np.isnan(X), axis=0)
    nan_row = np.all(np.isnan(X), axis=1)
    if not np.array_equal(nan_col, nan_row):
        raise ValueError("NaN values are not symmetric.")
    nan_inds = np.where(nan_col)
    subX_ = np.delete(X, nan_inds, axis=0)
    subX = np.delete(subX_, nan_inds, axis=1)
    if np.any(np.isnan(subX)):
        raise ValueError("NaN values must fill rows and columns.")
    mask = np.delete(np.eye(X.shape[0]), nan_inds, axis=1)
    return mask


def _get_masks_from_nan(X):
    masks = []
    for i in range(len(X)):
        masks.append(_get_mask_from_nan(X[i]))
    return masks


def _apply_masks(X, masks):
    maskedX = []
    for i in range(len(X)):
        maskedX.append(masks[i].T @ X[i] @ masks[i])
    return maskedX


def maskedmean_riemann(X=None, masks=None, tol=10e-9, maxiter=100, init=None,
                       sample_weight=None, covmats=None):
    """Masked Riemannian mean of SPD/HPD matrices.

    Given masks defined as semi-orthogonal matrices, the masked Riemannian mean
    of SPD/HPD matrices is obtained with a gradient descent minimizing the sum
    of affine-invariant Riemannian distances between masked SPD/HPD matrices
    and the masked mean [1]_.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices.
    masks : list of n_matrices ndarray of shape (n, n_i), \
            with different n_i, such that n_i <= n
        Masks, defined as semi-orthogonal matrices. See [1]_.
    tol : float, default=10e-9
        The tolerance to stop the gradient descent.
    maxiter : int, default=100
        The maximum number of iteration.
    init : None | ndarray, shape (n, n), default=None
        A SPD/HPD matrix used to initialize the gradient descent.
        If None, the Identity is used.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (n, n)
        Masked Riemannian mean.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    mean_riemann
    mean_covariance

    References
    ----------
    .. [1] `Geodesically-convex optimization for averaging partially observed
        covariance matrices
        <https://hal.archives-ouvertes.fr/hal-02984423>`_
        F. Yger, S. Chevallier, Q. Barthélemy, and S. Sra. Asian Conference on
        Machine Learning (ACML), Nov 2020, Bangkok, Thailand. pp.417 - 432.
    """
    if masks is None:
        raise ValueError("Input masks can not be None")
    X = _deprecate_covmats(covmats, X)
    n_matrices, n, _ = X.shape
    sample_weight = check_weights(sample_weight, n_matrices)
    maskedX = _apply_masks(X, masks)
    if init is None:
        M = np.eye(n)
    else:
        M = init

    nu = 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max
    for _ in range(maxiter):
        maskedM = _apply_masks(np.tile(M, (n_matrices, 1, 1)), masks)
        J = np.zeros((n, n), dtype=X.dtype)
        for i in range(n_matrices):
            M12, Mm12 = sqrtm(maskedM[i]), invsqrtm(maskedM[i])
            tmp = M12 @ logm(Mm12 @ maskedX[i] @ Mm12) @ M12
            J += sample_weight[i] * masks[i] @ tmp @ masks[i].T
        M12, Mm12 = sqrtm(M), invsqrtm(M)
        M = M12 @ expm(Mm12 @ (nu * J) @ Mm12) @ M12

        crit = np.linalg.norm(J, ord='fro')
        h = nu * crit
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu
        if crit <= tol or nu <= tol:
            break
    else:
        warnings.warn("Convergence not reached")

    return M


def nanmean_riemann(X=None, tol=10e-9, maxiter=100, init=None,
                    sample_weight=None, covmats=None):
    """Riemannian NaN-mean of SPD/HPD matrices.

    The Riemannian NaN-mean is the masked Riemannian mean applied to SPD/HPD
    matrices potentially corrupted by symmetric NaN values [1]_.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices, corrupted by symmetric NaN values [1]_.
    tol : float, default=10e-9
        The tolerance to stop the gradient descent.
    maxiter : int, default=100
        The maximum number of iteration.
    init : None | ndarray, shape (n, n), default=None
        A SPD/HPD matrix used to initialize the gradient descent.
        If None, a regularized Euclidean NaN-mean is used.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (n, n)
        Riemannian NaN-mean.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    maskedmean_riemann
    mean_covariance

    References
    ----------
    .. [1] `Geodesically-convex optimization for averaging partially observed
        covariance matrices
        <https://hal.archives-ouvertes.fr/hal-02984423>`_
        F. Yger, S. Chevallier, Q. Barthélemy, and S. Sra. Asian Conference on
        Machine Learning (ACML), Nov 2020, Bangkok, Thailand. pp.417 - 432.
    """
    X = _deprecate_covmats(covmats, X)
    n_matrices, n, _ = X.shape
    if init is None:
        Minit = np.nanmean(X, axis=0) + 1e-6 * np.eye(n)
    else:
        Minit = init

    M = maskedmean_riemann(
        np.nan_to_num(X),  # avoid nan contamination in matmul
        _get_masks_from_nan(X),
        tol=tol,
        maxiter=maxiter,
        init=Minit,
        sample_weight=sample_weight
    )
    return M
