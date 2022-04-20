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
    r"""Return the mean covariance matrix according to the Riemannian metric.

    The procedure is similar to a gradient descent minimizing the sum of
    Riemannian distances to the mean.

    .. math::
        \mathbf{C} = \arg\min{(\sum_i \delta_R ( \mathbf{C} , \mathbf{C}_i)^2)}

    :param covmats: Covariance matrices, (n_matrices, n_channels, n_channels)
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param init: A covariance matrix used to initialize the gradient descent. If None the Arithmetic mean is used
    :param sample_weight: the weight of each matrix
    :returns: the mean covariance matrix

    """  # noqa
    # init
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
    r"""Return the mean covariance matrix according to the log-Euclidean
    metric.

    .. math::
        \mathbf{C} = \exp{(\frac{1}{N} \sum_i \log{\mathbf{C}_i})}

    :param covmats: Covariance matrices, (n_matrices, n_channels, n_channels)
    :param sample_weight: the weight of each matrix

    :returns: the mean covariance matrix

    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    n_matrices, n_channels, _ = covmats.shape
    T = np.zeros((n_channels, n_channels))
    for index in range(n_matrices):
        T += sample_weight[index] * logm(covmats[index])
    C = expm(T)

    return C


def mean_kullback_sym(covmats, sample_weight=None):
    """Return the mean covariance matrix according to KL divergence.

    This mean is the geometric mean between the Arithmetic and the Harmonic
    mean, as shown in [1]_.

    :param covmats: Covariance matrices, (n_matrices, n_channels, n_channels)
    :param sample_weight: the weight of each matrix

    :returns: the mean covariance matrix

    References
    ----------
    .. [1]  Moakher, Maher, and Philipp G. Batchelor. "Symmetric
        positive-definite matrices: From geometry to applications and
        visualization." In Visualization and Processing of Tensor Fields, pp.
        285-298. Springer Berlin Heidelberg, 2006.
    """
    C_Arithmetic = mean_euclid(covmats, sample_weight)
    C_Harmonic = mean_harmonic(covmats, sample_weight)
    C = geodesic_riemann(C_Arithmetic, C_Harmonic, 0.5)

    return C


def mean_harmonic(covmats, sample_weight=None):
    r"""Return the harmonic mean of a set of covariance matrices.

    .. math::
        \mathbf{C} = \left(\frac{1}{N} \sum_i {\mathbf{C}_i}^{-1}\right)^{-1}

    :param covmats: Covariance matrices, (n_matrices, n_channels, n_channels)
    :param sample_weight: the weight of each matrix

    :returns: the mean covariance matrix

    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    n_matrices, n_channels, _ = covmats.shape
    T = np.zeros((n_channels, n_channels))
    for index in range(n_matrices):
        T += sample_weight[index] * np.linalg.inv(covmats[index])
    C = np.linalg.inv(T)

    return C


def mean_logdet(covmats, tol=10e-5, maxiter=50, init=None, sample_weight=None):
    r"""Return the mean covariance matrix according to the logdet metric.

    This is an iterative procedure where the update is:

    .. math::
        \mathbf{C} = \left(\sum_i \left( 0.5 \mathbf{C} + 0.5 \mathbf{C}_i \right)^{-1} \right)^{-1}

    :param covmats: Covariance matrices, (n_matrices, n_channels, n_channels)
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param init: A covariance matrix used to initialize the iterative procedure. If None the Arithmetic mean is used
    :param sample_weight: the weight of each matrix

    :returns: the mean covariance matrix

    """  # noqa
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
    r"""Return the mean covariance matrix according to the Wasserstein metric.

    This is an iterative procedure where the update is [1]_:

    .. math::
        \mathbf{K} = \left(\sum_i \left( \mathbf{K} \mathbf{C}_i \mathbf{K} \right)^{1/2} \right)^{1/2}

    with :math:`\mathbf{K} = \mathbf{C}^{1/2}`.

    :param covmats: Covariance matrices, (n_matrices, n_channels, n_channels)
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param init: A covariance matrix used to initialize the iterative procedure. If None the Arithmetic mean is used
    :param sample_weight: the weight of each matrix

    :returns: the mean covariance matrix

    References
    ----------
    .. [1] Barbaresco, F. "Geometric Radar Processing based on Frechet distance:
        Information geometry versus Optimal Transport Theory", Radar Symposium
        (IRS), 2011 Proceedings International.
    """  # noqa
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
    r"""Return the mean covariance matrix according to the Euclidean metric :

    .. math::
        \mathbf{C} = \frac{1}{N} \sum_i \mathbf{C}_i

    :param covmats: Covariance matrices, (n_matrices, n_channels, n_channels)
    :param sample_weight: the weight of each matrix

    :returns: the mean covariance matrix

    """
    return np.average(covmats, axis=0, weights=sample_weight)


def mean_ale(covmats, tol=10e-7, maxiter=50, sample_weight=None):
    """Return the mean covariance matrix according using the AJD-based
    log-Euclidean Mean (ALE). See [1].

    :param covmats: Covariance matrices, (n_matrices, n_channels, n_channels)
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param sample_weight: the weight of each matrix

    :returns: the mean covariance matrix

    Notes
    -----
    .. versionadded:: 0.2.4

    References
    ----------
    [1] M. Congedo, B. Afsari, A. Barachant, M. Moakher, 'Approximate Joint
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
    r"""Return Ando-Li-Mathias (ALM) mean

    Find the geometric mean recursively [1]_, generalizing from:

    .. math::
        \mathbf{C} = A^{\frac{1}{2}}(A^{-\frac{1}{2}}B^{\frac{1}{2}}A^{-\frac{1}{2}})^{\frac{1}{2}}A^{\frac{1}{2}}

    require a high number of iterations.

    This is the adaptation of the Matlab code proposed by Dario Bini and
    Bruno Iannazzo, http://bezout.dm.unipi.it/software/mmtoolbox/
    Extremely slow, due to the recursive formulation.

    :param covmats: Covariance matrices, (n_matrices, n_channels, n_channels)
    :param tol: the tolerance to stop iterations
    :param maxiter: maximum number of iteration, default 100
    :param verbose: indicate when reaching maxiter
    :param sample_weight: the weight of each matrix

    :returns: the mean covariance matrix

    Notes
    -----
    .. versionadded:: 0.2.8

    References
    ----------
    .. [1] T. Ando, C.-K. Li and R. Mathias, "Geometric Means", Linear Algebra
        Appl. 385 (2004), 305-334.
    """  # noqa
    sample_weight = _get_sample_weight(sample_weight, covmats)
    C = covmats
    C_iter = np.zeros_like(C)
    n_matrices = covmats.shape[0]
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
    """Return the power mean covariance matrix.

    :param covmats: Covariance matrices, (n_matrices, n_channels, n_channels)
    :param p: Exponent, not null
    :param sample_weight: Weight of each matrix
    :param zeta: Stopping criterion

    :returns: the mean covariance matrix

    Notes
    -----
    .. versionadded:: 0.2.8

    References
    ----------
    .. [1] Lim Y, Palfia M. "Matrix Power means and the Karcher mean", J.
           Funct. Anal., 2012
    .. [2] Congedo M, Barachant A, Kharati K E. "Fixed Point Algorithms for
           Estimating Power Means of Positive Definite Matrices", IEEE Trans.
           Sig. Process., 2017
    """
    if not isinstance(p, (int, float)):
        raise ValueError("Power mean only defined for a scalar exponent")
    if p == 0:
        raise ValueError("Power mean only defined for non-zero exponent")

    n_matrices, n_channels, _ = covmats.shape
    sample_weight = _get_sample_weight(sample_weight, covmats)
    phi = 0.375 / np.abs(p)

    G = np.sum([w * powm(c, p) for (w, c) in zip(sample_weight, covmats)],
               axis=0)
    if p > 0:
        X = invsqrtm(G)
    else:
        X = sqrtm(G)

    test = 10*zeta
    while test > zeta:
        H = np.sum(
            [w * powm(np.dot(X, np.dot(powm(c, np.sign(p)), X.T)), np.abs(p))
             for (w, c) in zip(sample_weight, covmats)],
            axis=0
        )
        X = np.dot(powm(H, -phi), X)
        test = np.linalg.norm(H - np.eye(n_channels)) / np.sqrt(n_channels)

    if p > 0:
        C = np.dot(np.linalg.inv(X), np.linalg.inv(X.T))
    else:
        C = np.dot(X.T, X)

    return C


def mean_identity(covmats, sample_weight=None):
    r"""Return the identity matrix corresponding to the covmats sit size

    .. math::
        \mathbf{C} = \mathbf{I}_d

    :param covmats: Covariance matrices, (n_matrices, n_channels, n_channels)
    :returns: the identity matrix of size n_channels

    """
    C = np.eye(covmats.shape[1])
    return C


def mean_covariance(covmats, metric='riemann', sample_weight=None, *args):
    """Return the mean covariance matrix according to the metric

    :param covmats: Covariance matrices, (n_matrices, n_channels, n_channels)
    :param metric: the metric (default 'riemann'), can be : 'riemann',
        'logeuclid', 'euclid', 'logdet', 'identity', 'wasserstein', 'ale',
        'alm', 'harmonic', 'kullback_sym' or a callable function
    :param sample_weight: the weight of each matrix
    :param args: the argument passed to the sub function
    :returns: the mean covariance matrix, (n_channels, n_channels)

    """
    if callable(metric):
        C = metric(covmats, sample_weight=sample_weight, *args)
    else:
        C = mean_methods[metric](covmats, sample_weight=sample_weight, *args)
    return C


mean_methods = {'riemann': mean_riemann,
                'logeuclid': mean_logeuclid,
                'euclid': mean_euclid,
                'identity': mean_identity,
                'logdet': mean_logdet,
                'wasserstein': mean_wasserstein,
                'ale': mean_ale,
                'harmonic': mean_harmonic,
                'kullback_sym': mean_kullback_sym,
                'alm': mean_alm}


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
    """Return the Riemannian NaN-mean of covariance matrices.

    The Riemannian NaN-mean is the masked Riemannian mean applied to covariance
    matrices corrupted by symmetric NaN values [1]_.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Covariance matrices, potentially corrupted by symmetric NaN values [1]_
    tol : float (default 10e-9)
        The tolerance to stop the gradient descent.
    maxiter : int (default 50)
        The maximum number of iteration.
    init : None | ndarray, shape (n_channels, n_channels) (default None)
        A covariance matrix used to initialize the gradient descent.
        If None, a regularized Euclidean NaN-mean is used.
    sample_weight : None | ndarray, shape (n_matrices) (default None)
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
    """Return the masked Riemannian mean of covariance matrices.

    Given masks defined as semi-orthogonal matrices, the masked Riemannian mean
    of covariance matrices is obtained with a gradient descent minimizing the
    sum of Riemannian distances between masked covariance matrices and the
    masked mean [1]_.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Covariance matrices.
    masks : list of n_matrices ndarray of shape (n_channels, n_channels_i), \
            with different n_channels_i, such that n_channels_i <= n_channels
        Masks, defined as semi-orthogonal matrices. See [1]_.
    tol : float (default 10e-9)
        The tolerance to stop the gradient descent.
    maxiter : int (default 50)
        The maximum number of iteration.
    init : None | ndarray, shape (n_channels, n_channels) (default None)
        A covariance matrix used to initialize the gradient descent.
        If None, the Identity is used.
    sample_weight : None | ndarray, shape (n_matrices) (default None)
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
