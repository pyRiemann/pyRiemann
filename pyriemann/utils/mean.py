"""Mean covariance estimation."""

from copy import deepcopy
import numpy as np

from .base import sqrtm, invsqrtm, logm, expm, powm
from .ajd import ajd_pham
from .distance import distance_riemann
from .geodesic import geodesic_riemann


def _get_sample_weight(sample_weight, data):
    """Get the sample weights.

    If none provided, weights init to 1. otherwise, weights are normalized.
    """
    if sample_weight is None:
        sample_weight = np.ones(data.shape[0])
    if len(sample_weight) != data.shape[0]:
        raise ValueError("len of sample_weight must be equal to len of data.")
    sample_weight /= np.sum(sample_weight)
    return sample_weight


def mean_riemann(covmats, tol=10e-9, maxiter=50, init=None,
                 sample_weight=None):
    r"""Mean of SPD matrices according to the Riemannian metric.

    The procedure is similar to a gradient descent minimizing the sum of
    affine-invariant Riemannian distances :math:`d_R` to the mean:

    .. math::
        \mathbf{C} = \arg\min{ \sum_i d_R ( \mathbf{C} , \mathbf{C}_i)^2 }

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    tol : float (default 10e-9)
        The tolerance to stop the gradient descent.
    maxiter : int (default 50)
        The maximum number of iterations.
    init :  None | ndarray, shape (n_channels, n_channels) (default None)
        A SPD matrix used to initialize the gradient descent.
        If None the Euclidean mean is used.
    sample_weight : None | ndarray, shape (n_matrices,) (default None)
        The weight of each matrix.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels)
        Riemannian mean.
    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    n_matrices, n_channels, _ = covmats.shape
    if init is None:
        C = np.mean(covmats, axis=0)
    else:
        C = init

    k = 0
    nu = 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter) and (nu > tol):
        k = k + 1
        C12 = sqrtm(C)
        Cm12 = invsqrtm(C)
        J = np.zeros((n_channels, n_channels))

        for index in range(n_matrices):
            tmp = np.dot(np.dot(Cm12, covmats[index]), Cm12)
            J += sample_weight[index] * logm(tmp)

        crit = np.linalg.norm(J, ord='fro')
        h = nu * crit
        C = np.dot(np.dot(C12, expm(nu * J)), C12)
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    return C


def mean_logeuclid(covmats, sample_weight=None):
    r"""Mean of SPD matrices according to the log-Euclidean metric.

    .. math::
        \mathbf{C} = \exp{(\frac{1}{m} \sum_i \log{\mathbf{C}_i})}

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    sample_weight : None | ndarray, shape (n_matrices,) (default None)
        The weight of each matrix.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels)
        Log-Euclidean mean.
    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    n_matrices, n_channels, _ = covmats.shape
    T = np.zeros((n_channels, n_channels))
    for index in range(n_matrices):
        T += sample_weight[index] * logm(covmats[index])
    C = expm(T)

    return C


def mean_kullback_sym(covmats, sample_weight=None):
    """Mean of SPD matrices according to Kullback-Leibler divergence.

    This mean is the geometric mean between the Euclidean and the harmonic
    means, as shown in [1]_.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    sample_weight : None | ndarray, shape (n_matrices,) (default None)
        The weight of each matrix.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels)
        Kullback-Leibler mean.

    References
    ----------
    .. [1] Moakher, M, and Philipp G. B. "Symmetric positive-definite matrices:
        From geometry to applications and visualization", Visualization and
        Processing of Tensor Fields, pp. 285-298, 2006
    """
    C_euclid = mean_euclid(covmats, sample_weight)
    C_harmonic = mean_harmonic(covmats, sample_weight)
    C = geodesic_riemann(C_euclid, C_harmonic, 0.5)

    return C


def mean_harmonic(covmats, sample_weight=None):
    r"""Harmonic mean of SPD matrices.

    .. math::
        \mathbf{C} = \left(\frac{1}{m} \sum_i {\mathbf{C}_i}^{-1}\right)^{-1}

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    sample_weight : None | ndarray, shape (n_matrices,) (default None)
        The weight of each matrix.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels)
        Harmonic mean.
    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    n_matrices, n_channels, _ = covmats.shape
    T = np.zeros((n_channels, n_channels))
    for index in range(n_matrices):
        T += sample_weight[index] * np.linalg.inv(covmats[index])
    C = np.linalg.inv(T)

    return C


def mean_logdet(covmats, tol=10e-5, maxiter=50, init=None, sample_weight=None):
    r"""Mean of SPD matrices according to the log-det metric.

    Log-det mean is obtained by an iterative procedure where the update is:

    .. math::
        \mathbf{C} = \left(\sum_i \left( 0.5 \mathbf{C}
                     + 0.5 \mathbf{C}_i \right)^{-1} \right)^{-1}

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    tol : float (default 10e-5)
        The tolerance to stop the gradient descent.
    maxiter : int (default 50)
        The maximum number of iterations.
    init :  None | ndarray, shape (n_channels, n_channels) (default None)
        A SPD matrix used to initialize the gradient descent.
        If None the Euclidean mean is used.
    sample_weight : None | ndarray, shape (n_matrices,) (default None)
        The weight of each matrix.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels)
        Log-det mean.
    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    n_matrices, n_channels, _ = covmats.shape
    if init is None:
        C = np.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    crit = np.finfo(np.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter):
        k = k + 1

        J = np.zeros((n_channels, n_channels))

        for index, Ci in enumerate(covmats):
            J += sample_weight[index] * np.linalg.inv(0.5 * Ci + 0.5 * C)

        Cnew = np.linalg.inv(J)
        crit = np.linalg.norm(Cnew - C, ord='fro')

        C = Cnew
    return C


def mean_wasserstein(covmats, tol=10e-4, maxiter=50, init=None,
                     sample_weight=None):
    r"""Mean of SPD matrices according to the Wasserstein metric.

    Wasserstein mean is obtained by an iterative procedure where the update is
    [1]_:

    .. math::
        \mathbf{K} = \left(\sum_i \left( \mathbf{K} \mathbf{C}_i \mathbf{K}
                     \right)^{1/2} \right)^{1/2}

    with :math:`\mathbf{K} = \mathbf{C}^{1/2}`.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    tol : float (default 10e-4)
        The tolerance to stop the gradient descent.
    maxiter : int (default 50)
        The maximum number of iterations.
    init :  None | ndarray, shape (n_channels, n_channels) (default None)
        A SPD matrix used to initialize the gradient descent.
        If None the Euclidean mean is used.
    sample_weight : None | ndarray, shape (n_matrices,) (default None)
        The weight of each matrix.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels)
        Wasserstein mean.

    References
    ----------
    .. [1] Barbaresco, F. "Geometric Radar Processing based on Frechet distance
        : Information geometry versus Optimal Transport Theory", IRS, 2011
    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    n_matrices, n_channels, _ = covmats.shape
    if init is None:
        C = np.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    K = sqrtm(C)
    crit = np.finfo(np.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter):
        k = k + 1

        J = np.zeros((n_channels, n_channels))

        for index, Ci in enumerate(covmats):
            tmp = np.dot(np.dot(K, Ci), K)
            J += sample_weight[index] * sqrtm(tmp)

        Knew = sqrtm(J)
        crit = np.linalg.norm(Knew - K, ord='fro')
        K = Knew
    if k == maxiter:
        print('Max iter reach')
    C = np.dot(K, K)
    return C


def mean_euclid(covmats, sample_weight=None):
    r"""Mean of SPD matrices according to the Euclidean metric.

    .. math::
        \mathbf{C} = \frac{1}{m} \sum_i \mathbf{C}_i

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    sample_weight : None | ndarray, shape (n_matrices,) (default None)
        The weight of each matrix.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels)
        Euclidean mean.
    """
    return np.average(covmats, axis=0, weights=sample_weight)


def mean_ale(covmats, tol=10e-7, maxiter=50, sample_weight=None):
    """AJD-based log-Euclidean (ALE) mean of SPD matrices.

    Return the mean of a set of SPD matrices using the AJD-based log-Euclidean
    (ALE) mean [1]_.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    tol : float (default 10e-7)
        The tolerance to stop the gradient descent.
    maxiter : int (default 50)
        The maximum number of iterations.
    sample_weight : None | ndarray, shape (n_matrices,) (default None)
        The weight of each matrix.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels)
        ALE mean.

    Notes
    -----
    .. versionadded:: 0.2.4

    References
    ----------
    .. [1] M. Congedo, B. Afsari, A. Barachant, M. Moakher, 'Approximate Joint
        Diagonalization and Geometric Mean of Symmetric Positive Definite
        Matrices', PLoS ONE, 2015
    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    n_matrices, n_channels, _ = covmats.shape
    crit = np.inf
    k = 0

    # init with AJD
    B, _ = ajd_pham(covmats)
    while (crit > tol) and (k < maxiter):
        k += 1
        J = np.zeros((n_channels, n_channels))

        for index, Ci in enumerate(covmats):
            tmp = logm(np.dot(np.dot(B.T, Ci), B))
            J += sample_weight[index] * tmp

        update = np.diag(np.diag(expm(J)))
        B = np.dot(B, invsqrtm(update))

        crit = distance_riemann(np.eye(n_channels), update)

    A = np.linalg.inv(B)

    J = np.zeros((n_channels, n_channels))
    for index, Ci in enumerate(covmats):
        tmp = logm(np.dot(np.dot(B.T, Ci), B))
        J += sample_weight[index] * tmp

    C = np.dot(np.dot(A.T, expm(J)), A)
    return C


def mean_alm(covmats, tol=1e-14, maxiter=100,
             verbose=False, sample_weight=None):
    r"""Ando-Li-Mathias (ALM) mean of SPD matrices.

    Return the geometric mean recursively [1]_, generalizing from:

    .. math::
        \mathbf{C} = A^{\frac{1}{2}}(A^{-\frac{1}{2}}B^{\frac{1}{2}}
                     A^{-\frac{1}{2}})^{\frac{1}{2}}A^{\frac{1}{2}}

    and requiring a high number of iterations.

    This is the adaptation of the Matlab code proposed by Dario Bini and
    Bruno Iannazzo, http://bezout.dm.unipi.it/software/mmtoolbox/ .
    Extremely slow, due to the recursive formulation.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    tol : float (default 10e-14)
        The tolerance to stop the gradient descent.
    maxiter : int (default 100)
        The maximum number of iterations.
    verbose : bool (default False)
        Indicate when reaching maxiter.
    sample_weight : None | ndarray, shape (n_matrices,) (default None)
        The weight of each matrix.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels)
        ALM mean.

    Notes
    -----
    .. versionadded:: 0.2.8

    References
    ----------
    .. [1] T. Ando, C.-K. Li and R. Mathias, "Geometric Means", Linear Algebra
        Appl. 385 (2004), 305-334.
    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    C = covmats
    C_iter = np.zeros_like(C)
    n_matrices, _, _ = covmats.shape
    if n_matrices == 2:
        alpha = sample_weight[1] / sample_weight[0] / 2
        X = geodesic_riemann(covmats[0], covmats[1], alpha=alpha)
        return X
    else:
        for k in range(maxiter):
            for h in range(n_matrices):
                s = np.mod(np.arange(h, h + n_matrices - 1) + 1, n_matrices)
                C_iter[h] = mean_alm(C[s], sample_weight=sample_weight[s])

            norm_iter = np.linalg.norm(C_iter[0] - C[0], 2)
            norm_c = np.linalg.norm(C[0], 2)
            if (norm_iter / norm_c) < tol:
                break
            C = deepcopy(C_iter)
        else:
            if verbose:
                print('Max number of iterations reached')
        return C_iter.mean(axis=0)


def mean_power(covmats, p, *, sample_weight=None, zeta=10e-10):
    """Power mean of SPD matrices.

    :param covmats: Covariance matrices, (n_matrices, n_channels, n_channels)
    :param p: Exponent, in [-1,+1]
    :param sample_weight: Weight of each matrix
    :param zeta: Stopping criterion

    :returns: the mean covariance matrix

    Notes
    -----
    .. versionadded:: 0.2.8

    References
    ----------
    .. [1] Lim Y and Palfia M. "Matrix Power means and the Karcher mean", J.
           Funct. Anal., 2012
    .. [2] Congedo M, Barachant A and Kharati K E. "Fixed Point Algorithms for
           Estimating Power Means of Positive Definite Matrices", IEEE Trans.
           Sig. Process., 2017
    """
    if not isinstance(p, (int, float)):
        raise ValueError("Power mean only defined for a scalar exponent")
    if p < -1 or 1 < p:
        raise ValueError("Exponent must be in [-1,+1]")

    if p == 0:
        return mean_riemann(covmats, sample_weight=sample_weight)

    n_matrices, n_channels, _ = covmats.shape
    sample_weight = _get_sample_weight(sample_weight, covmats)
    phi = 0.375 / np.abs(p)

    G = np.sum(
        [w * powm(c, p) for (w, c) in zip(sample_weight, covmats)],
        axis=0
    )
    if p > 0:
        X = invsqrtm(G)
    else:
        X = sqrtm(G)

    test = 10 * zeta
    while test > zeta:
        H = np.sum(
            [w * powm(X @ powm(c, np.sign(p)) @ X.T, np.abs(p))
             for (w, c) in zip(sample_weight, covmats)],
            axis=0
        )
        X = powm(H, -phi) @ X
        test = np.linalg.norm(H - np.eye(n_channels)) / np.sqrt(n_channels)

    if p > 0:
        C = np.linalg.inv(X) @ np.linalg.inv(X.T)
    else:
        C = X.T @ X

    return C


def mean_identity(covmats, sample_weight=None):
    r"""Identity matrix corresponding to the matrices dimension.

    .. math::
        \mathbf{C} = \mathbf{I}_c

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    sample_weight : None
        Not used, here for compatibility with other means.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels)
        Identity.
    """
    _, n_channels, _ = covmats.shape
    C = np.eye(n_channels)
    return C


def mean_covariance(covmats, metric='riemann', sample_weight=None, *args):
    """Mean of SPD matrices according to a metric.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    metric : string (default 'riemann')
        The metric, can be: 'riemann', 'logeuclid', 'euclid', 'logdet',
        'identity', 'wasserstein', 'ale', 'alm', 'harmonic', 'kullback_sym'
        or a callable function.
    args : list of params
        The arguments passed to the sub function.
    sample_weight : None | ndarray, shape (n_matrices,) (default None)
        The weight of each matrix.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels)
        Mean of SPD matrices.
    """
    if callable(metric):
        C = metric(covmats, sample_weight=sample_weight, *args)
    else:
        C = mean_methods[metric](covmats, sample_weight=sample_weight, *args)
    return C


mean_methods = {
    'riemann': mean_riemann,
    'logeuclid': mean_logeuclid,
    'euclid': mean_euclid,
    'identity': mean_identity,
    'logdet': mean_logdet,
    'wasserstein': mean_wasserstein,
    'ale': mean_ale,
    'alm': mean_alm,
    'harmonic': mean_harmonic,
    'kullback_sym': mean_kullback_sym,
}


def _check_mean_method(method):
    """checks methods """
    if isinstance(method, str):
        if method not in mean_methods.keys():
            raise ValueError('Unknown mean method')
        else:
            method = mean_methods[method]
    elif not hasattr(method, '__call__'):
        raise ValueError('mean method must be a function or a string.')
    return method


###############################################################################


def _get_mask_from_nan(covmat):
    nan_col = np.all(np.isnan(covmat), axis=0)
    nan_row = np.all(np.isnan(covmat), axis=1)
    if not np.array_equal(nan_col, nan_row):
        raise ValueError('NaN values are not symmetric.')
    nan_inds = np.where(nan_col)
    subcovmat_ = np.delete(covmat, nan_inds, axis=0)
    subcovmat = np.delete(subcovmat_, nan_inds, axis=1)
    if np.any(np.isnan(subcovmat)):
        raise ValueError('NaN values must fill rows and columns.')
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


def nanmean_riemann(covmats, tol=10e-9, maxiter=50, init=None,
                    sample_weight=None):
    """Riemannian NaN-mean of SPD matrices.

    The Riemannian NaN-mean is the masked Riemannian mean applied to SPD
    matrices potentially corrupted by symmetric NaN values [1]_.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices, corrupted by symmetric NaN values [1]_.
    tol : float (default 10e-9)
        The tolerance to stop the gradient descent.
    maxiter : int (default 50)
        The maximum number of iteration.
    init : None | ndarray, shape (n_channels, n_channels) (default None)
        A SPD matrix used to initialize the gradient descent.
        If None, a regularized Euclidean NaN-mean is used.
    sample_weight : None | ndarray, shape (n_matrices,) (default None)
        The weight of each matrix.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels)
        Riemannian NaN-mean.

    Notes
    -----
    .. versionadded:: 0.2.8

    References
    ----------
    .. [1] F. Yger, S. Chevallier, Q. Barthélemy, S. Sra. "Geodesically-convex
        optimization for averaging partially observed covariance matrices",
        ACML 2020.
    """
    n_matrices, n_channels, _ = covmats.shape
    if init is None:
        Cinit = np.nanmean(covmats, axis=0) + 1e-6 * np.eye(n_channels)
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


def maskedmean_riemann(covmats, masks, tol=10e-9, maxiter=50, init=None,
                       sample_weight=None):
    """Masked Riemannian mean of SPD matrices.

    Given masks defined as semi-orthogonal matrices, the masked Riemannian mean
    of SPD matrices is obtained with a gradient descent minimizing the
    sum of Riemannian distances between masked SPD matrices and the
    masked mean [1]_.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    masks : list of n_matrices ndarray of shape (n_channels, n_channels_i), \
            with different n_channels_i, such that n_channels_i <= n_channels
        Masks, defined as semi-orthogonal matrices. See [1]_.
    tol : float (default 10e-9)
        The tolerance to stop the gradient descent.
    maxiter : int (default 50)
        The maximum number of iteration.
    init : None | ndarray, shape (n_channels, n_channels) (default None)
        A SPD matrix used to initialize the gradient descent.
        If None, the Identity is used.
    sample_weight : None | ndarray, shape (n_matrices,) (default None)
        The weight of each matrix.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels)
        Masked Riemannian mean.

    Notes
    -----
    .. versionadded:: 0.2.8

    References
    ----------
    .. [1] F. Yger, S. Chevallier, Q. Barthélemy, S. Sra. "Geodesically-convex
        optimization for averaging partially observed covariance matrices",
        ACML 2020.
    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    maskedcovmats = _apply_masks(covmats, masks)
    n_matrices, n_channels, _ = covmats.shape
    if init is None:
        C = np.eye(n_channels)
    else:
        C = init

    k = 0
    nu = 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max
    while (crit > tol) and (k < maxiter) and (nu > tol):
        k = k + 1
        maskedC = _apply_masks(np.tile(C, (n_matrices, 1, 1)), masks)
        J = np.zeros((n_channels, n_channels))
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

    return C
