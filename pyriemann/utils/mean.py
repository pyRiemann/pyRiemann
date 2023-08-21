"""Means of SPD/HPD matrices."""

from copy import deepcopy
import numpy as np
import warnings

from .ajd import ajd_pham
from .base import sqrtm, invsqrtm, logm, expm, powm
from .distance import distance_riemann
from .geodesic import geodesic_riemann
from .utils import check_weights


def mean_ale(covmats, tol=10e-7, maxiter=50, sample_weight=None):
    """AJD-based log-Euclidean (ALE) mean of SPD matrices.

    Return the mean of a set of SPD matrices using the approximate joint
    diagonalization (AJD) based log-Euclidean (ALE) mean [1]_.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n, n)
        Set of SPD matrices.
    tol : float, default=10e-7
        The tolerance to stop the gradient descent.
    maxiter : int, default=50
        The maximum number of iterations.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    C : ndarray, shape (n, n)
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
    n_matrices, n, _ = covmats.shape
    sample_weight = check_weights(sample_weight, n_matrices)

    # init with AJD
    B = ajd_pham(covmats)[0]

    eye_n = np.eye(n)
    crit = np.inf
    for _ in range(maxiter):
        J = np.einsum(
            'a,abc->bc',
            sample_weight,
            logm(B @ covmats @ B.conj().T)
        )
        delta = np.real(np.diag(expm(J)))
        B = (np.abs(delta) ** -.5)[:, np.newaxis] * B

        crit = distance_riemann(eye_n, np.diag(delta))
        if crit <= tol:
            break
    else:
        warnings.warn("Convergence not reached")

    J = np.einsum('a,abc->bc', sample_weight, logm(B @ covmats @ B.conj().T))
    A = np.linalg.inv(B)
    C = A @ expm(J) @ A.conj().T
    return C


def mean_alm(covmats, tol=1e-14, maxiter=100, sample_weight=None):
    r"""Ando-Li-Mathias (ALM) mean of SPD/HPD matrices.

    Return the geometric mean recursively [1]_, generalizing from:

    .. math::
        \mathbf{C} = X_1^{\frac{1}{2}} (X_1^{-\frac{1}{2}}X_2^{\frac{1}{2}}
                     X_1^{-\frac{1}{2}})^{\frac{1}{2}} X_1^{\frac{1}{2}}

    and requiring a high number of iterations.
    Extremely slow, due to the recursive formulation.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices.
    tol : float, default=10e-14
        The tolerance to stop the gradient descent.
    maxiter : int, default=100
        The maximum number of iterations.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    C : ndarray, shape (n, n)
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
    n_matrices, _, _ = covmats.shape
    sample_weight = check_weights(sample_weight, n_matrices)

    if n_matrices == 2:
        alpha = sample_weight[1] / sample_weight[0] / 2
        C = geodesic_riemann(covmats[0], covmats[1], alpha=alpha)
        return C

    C = covmats
    C_iter = np.zeros_like(C)
    for _ in range(maxiter):
        for h in range(n_matrices):
            s = np.mod(np.arange(h, h + n_matrices - 1) + 1, n_matrices)
            C_iter[h] = mean_alm(C[s], sample_weight=sample_weight[s])

        norm_iter = np.linalg.norm(C_iter[0] - C[0], 2)
        norm_c = np.linalg.norm(C[0], 2)
        if (norm_iter / norm_c) < tol:
            break
        C = deepcopy(C_iter)
    else:
        warnings.warn("Convergence not reached")

    return C_iter.mean(axis=0)


def mean_euclid(covmats, sample_weight=None):
    r"""Mean of matrices according to the Euclidean metric.

    .. math::
        \mathbf{C} = \sum_i w_i \ \mathbf{X}_i

    This mean is also called arithmetic.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n, m)
        Set of matrices.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    C : ndarray, shape (n, m)
        Euclidean mean.

    See Also
    --------
    mean_covariance
    """
    return np.average(covmats, axis=0, weights=sample_weight)


def mean_harmonic(covmats, sample_weight=None):
    r"""Harmonic mean of invertible matrices.

    .. math::
        \mathbf{C} = \left( \sum_i wi \ {\mathbf{X}_i}^{-1} \right)^{-1}

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n, n)
        Set of invertible matrices.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    C : ndarray, shape (n, n)
        Harmonic mean.

    See Also
    --------
    mean_covariance
    """
    T = mean_euclid(np.linalg.inv(covmats), sample_weight=sample_weight)
    C = np.linalg.inv(T)
    return C


def mean_identity(covmats, sample_weight=None):
    r"""Identity matrix corresponding to the matrices dimension.

    .. math::
        \mathbf{C} = \mathbf{I}_n

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n, n)
        Set of square matrices.
    sample_weight : None
        Not used, here for compatibility with other means.

    Returns
    -------
    C : ndarray, shape (n, n)
        Identity matrix.

    See Also
    --------
    mean_covariance
    """
    C = np.eye(covmats.shape[-1])
    return C


def mean_kullback_sym(covmats, sample_weight=None):
    """Mean of SPD/HPD matrices according to Kullback-Leibler divergence.

    Symmetrized Kullback-Leibler mean is the geometric mean between the
    Euclidean and the harmonic means [1]_.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    C : ndarray, shape (n, n)
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
    C_euclid = mean_euclid(covmats, sample_weight=sample_weight)
    C_harmonic = mean_harmonic(covmats, sample_weight=sample_weight)
    C = geodesic_riemann(C_euclid, C_harmonic, 0.5)
    return C


def mean_logdet(covmats, tol=10e-5, maxiter=50, init=None, sample_weight=None):
    r"""Mean of SPD/HPD matrices according to the log-det metric.

    Log-det mean is obtained by an iterative procedure where the update is:

    .. math::
        \mathbf{C} = \left( \sum_i w_i \ \left( 0.5 \mathbf{C}
                     + 0.5 \mathbf{X}_i \right)^{-1} \right)^{-1}

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n, n)
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
    C : ndarray, shape (n, n)
        Log-det mean.

    See Also
    --------
    mean_covariance
    """
    n_matrices, _, _ = covmats.shape
    sample_weight = check_weights(sample_weight, n_matrices)
    if init is None:
        C = mean_euclid(covmats, sample_weight=sample_weight)
    else:
        C = init

    crit = np.finfo(np.float64).max
    for _ in range(maxiter):
        icovmats = np.linalg.inv(0.5 * covmats + 0.5 * C)
        J = np.einsum('a,abc->bc', sample_weight, icovmats)
        Cnew = np.linalg.inv(J)

        crit = np.linalg.norm(Cnew - C, ord='fro')
        C = Cnew
        if crit <= tol:
            break
    else:
        warnings.warn("Convergence not reached")

    return C


def mean_logeuclid(covmats, sample_weight=None):
    r"""Mean of SPD/HPD matrices according to the log-Euclidean metric.

    Log-Euclidean mean is [1]_:

    .. math::
        \mathbf{C} = \exp{ \left( \sum_i w_i \ \log{\mathbf{X}_i} \right) }

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    C : ndarray, shape (n, n)
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
    C = expm(mean_euclid(logm(covmats), sample_weight=sample_weight))
    return C


def mean_power(covmats, p, *, sample_weight=None, zeta=10e-10, maxiter=100):
    r"""Power mean of SPD/HPD matrices.

    Power mean of order p is the solution of [1]_ [2]_:

    .. math::
        \mathbf{C} = \sum_i w_i \ \mathbf{C} \sharp_p \mathbf{X}_i

    where :math:`\mathbf{A} \sharp_p \mathbf{B}` is the geodesic between
    matrices :math:`\mathbf{A}` and :math:`\mathbf{B}`.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n, n)
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
    C : ndarray, shape (n, n)
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
    if not isinstance(p, (int, float)):
        raise ValueError("Power mean only defined for a scalar exponent")
    if p < -1 or 1 < p:
        raise ValueError("Exponent must be in [-1,+1]")

    if p == 1:
        return mean_euclid(covmats, sample_weight=sample_weight)
    elif p == 0:
        return mean_riemann(covmats, sample_weight=sample_weight)
    elif p == -1:
        return mean_harmonic(covmats, sample_weight=sample_weight)

    n_matrices, n, _ = covmats.shape
    sample_weight = check_weights(sample_weight, n_matrices)
    phi = 0.375 / np.abs(p)

    G = np.einsum('a,abc->bc', sample_weight, powm(covmats, p))
    if p > 0:
        X = invsqrtm(G)
    else:
        X = sqrtm(G)

    eye_n, sqrt_n = np.eye(n), np.sqrt(n)
    crit = 10 * zeta
    for _ in range(maxiter):
        H = np.einsum(
            'a,abc->bc',
            sample_weight,
            powm(X @ powm(covmats, np.sign(p)) @ X.conj().T, np.abs(p))
        )
        X = powm(H, -phi) @ X

        crit = np.linalg.norm(H - eye_n) / sqrt_n
        if crit <= zeta:
            break
    else:
        warnings.warn("Convergence not reached")

    C = X.conj().T @ X
    if p > 0:
        C = np.linalg.inv(C)

    return C


def mean_riemann(covmats, tol=10e-9, maxiter=50, init=None,
                 sample_weight=None):
    r"""Mean of SPD/HPD matrices according to the Riemannian metric.

    The affine-invariant Riemannian mean minimizes the sum of squared
    affine-invariant Riemannian distances :math:`d_R` to all SPD/HPD matrices
    [1]_ [2]_:

    .. math::
         \arg \min_{\mathbf{C}} \sum_i w_i \ d_R (\mathbf{C}, \mathbf{X}_i)^2

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n, n)
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
    C : ndarray, shape (n, n)
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
    n_matrices, _, _ = covmats.shape
    sample_weight = check_weights(sample_weight, n_matrices)
    if init is None:
        C = mean_euclid(covmats, sample_weight=sample_weight)
    else:
        C = init

    nu = 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max
    for _ in range(maxiter):
        C12, Cm12 = sqrtm(C), invsqrtm(C)
        J = np.einsum('a,abc->bc', sample_weight, logm(Cm12 @ covmats @ Cm12))
        C = C12 @ expm(nu * J) @ C12

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

    return C


def mean_wasserstein(covmats, tol=10e-4, maxiter=50, init=None,
                     sample_weight=None):
    r"""Mean of SPD/HPD matrices according to the Wasserstein metric.

    Wasserstein mean is obtained by an iterative procedure where the update is
    [1]_:

    .. math::
        \mathbf{K} = \left( \sum_i w_i \ \left( \mathbf{K} \mathbf{X}_i
                     \mathbf{K} \right)^{1/2} \right)^{1/2}

    with :math:`\mathbf{K} = \mathbf{C}^{1/2}`.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n, n)
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
    C : ndarray, shape (n, n)
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
    n_matrices, _, _ = covmats.shape
    sample_weight = check_weights(sample_weight, n_matrices)
    if init is None:
        C = mean_euclid(covmats, sample_weight=sample_weight)
    else:
        C = init
    K = sqrtm(C)

    crit = np.finfo(np.float64).max
    for _ in range(maxiter):
        J = np.einsum('a,abc->bc', sample_weight, sqrtm(K @ covmats @ K))
        Knew = sqrtm(J)

        crit = np.linalg.norm(Knew - K, ord='fro')
        K = Knew
        if crit <= tol:
            break
    else:
        warnings.warn("Convergence not reached")

    C = K @ K
    return C


###############################################################################


mean_functions = {
    'ale': mean_ale,
    'alm': mean_alm,
    'euclid': mean_euclid,
    'harmonic': mean_harmonic,
    'identity': mean_identity,
    'kullback_sym': mean_kullback_sym,
    'logdet': mean_logdet,
    'logeuclid': mean_logeuclid,
    'riemann': mean_riemann,
    'wasserstein': mean_wasserstein,
}


def _check_mean_function(metric):
    """Check mean function."""
    if isinstance(metric, str):
        if metric not in mean_functions.keys():
            raise ValueError(f"Unknown mean metric '{metric}'")
        else:
            metric = mean_functions[metric]
    elif not hasattr(metric, '__call__'):
        raise ValueError("Mean metric must be a function or a string "
                         f"(Got {type(metric)}.")
    return metric


def mean_covariance(covmats, metric='riemann', sample_weight=None, **kwargs):
    """Mean of matrices according to a metric.

    Compute the mean of a set of matrices according to a metric [1]_.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n, n)
        Set of matrices.
    metric : string, default='riemann'
        The metric for mean, can be: 'ale', 'alm', 'euclid', 'harmonic',
        'identity', 'kullback_sym', 'logdet', 'logeuclid', 'riemann',
        'wasserstein', or a callable function.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.
    **kwargs : dict
        The keyword arguments passed to the sub function.

    Returns
    -------
    C : ndarray, shape (n, n)
        Mean of matrices.

    References
    ----------
    .. [1] `Review of Riemannian distances and divergences, applied to
        SSVEP-based BCI
        <https://hal.archives-ouvertes.fr/LISV/hal-03015762v1>`_
        S. Chevallier, E. K. Kalunga, Q. Barthélemy, E. Monacelli.
        Neuroinformatics, Springer, 2021, 19 (1), pp.93-106
    """
    mean_function = _check_mean_function(metric)
    C = mean_function(
        covmats,
        sample_weight=sample_weight,
        **kwargs,
    )
    return C


###############################################################################


def _get_mask_from_nan(covmat):
    nan_col = np.all(np.isnan(covmat), axis=0)
    nan_row = np.all(np.isnan(covmat), axis=1)
    if not np.array_equal(nan_col, nan_row):
        raise ValueError("NaN values are not symmetric.")
    nan_inds = np.where(nan_col)
    subcovmat_ = np.delete(covmat, nan_inds, axis=0)
    subcovmat = np.delete(subcovmat_, nan_inds, axis=1)
    if np.any(np.isnan(subcovmat)):
        raise ValueError("NaN values must fill rows and columns.")
    mask = np.delete(np.eye(covmat.shape[0]), nan_inds, axis=1)
    return mask


def _get_masks_from_nan(covmats):
    masks = []
    for i in range(len(covmats)):
        masks.append(_get_mask_from_nan(covmats[i]))
    return masks


def _apply_masks(covmats, masks):
    maskedcovmats = []
    for i in range(len(covmats)):
        maskedcovmats.append(masks[i].T @ covmats[i] @ masks[i])
    return maskedcovmats


def maskedmean_riemann(covmats, masks, tol=10e-9, maxiter=100, init=None,
                       sample_weight=None):
    """Masked Riemannian mean of SPD/HPD matrices.

    Given masks defined as semi-orthogonal matrices, the masked Riemannian mean
    of SPD/HPD matrices is obtained with a gradient descent minimizing the sum
    of affine-invariant Riemannian distances between masked SPD/HPD matrices
    and the masked mean [1]_.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n, n)
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
    C : ndarray, shape (n, n)
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
    n_matrices, n, _ = covmats.shape
    sample_weight = check_weights(sample_weight, n_matrices)
    maskedcovmats = _apply_masks(covmats, masks)
    if init is None:
        C = np.eye(n)
    else:
        C = init

    nu = 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max
    for _ in range(maxiter):
        maskedC = _apply_masks(np.tile(C, (n_matrices, 1, 1)), masks)
        J = np.zeros((n, n), dtype=covmats.dtype)
        for i in range(n_matrices):
            C12, Cm12 = sqrtm(maskedC[i]), invsqrtm(maskedC[i])
            tmp = C12 @ logm(Cm12 @ maskedcovmats[i] @ Cm12) @ C12
            J += sample_weight[i] * masks[i] @ tmp @ masks[i].T
        C12, Cm12 = sqrtm(C), invsqrtm(C)
        C = C12 @ expm(Cm12 @ (nu * J) @ Cm12) @ C12

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

    return C


def nanmean_riemann(covmats, tol=10e-9, maxiter=100, init=None,
                    sample_weight=None):
    """Riemannian NaN-mean of SPD/HPD matrices.

    The Riemannian NaN-mean is the masked Riemannian mean applied to SPD/HPD
    matrices potentially corrupted by symmetric NaN values [1]_.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n, n)
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
    C : ndarray, shape (n, n)
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
    n_matrices, n, _ = covmats.shape
    if init is None:
        Cinit = np.nanmean(covmats, axis=0) + 1e-6 * np.eye(n)
    else:
        Cinit = init

    C = maskedmean_riemann(
        np.nan_to_num(covmats),  # avoid nan contamination in matmul
        _get_masks_from_nan(covmats),
        tol=tol,
        maxiter=maxiter,
        init=Cinit,
        sample_weight=sample_weight
    )
    return C
