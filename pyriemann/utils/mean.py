"""Mean covariance estimation."""
import numpy

from .base import sqrtm, invsqrtm, logm, expm
from .ajd import ajd_pham
from .distance import distance_riemann
from .geodesic import geodesic_riemann


def _get_sample_weight(sample_weight, data):
    """Get the sample weights.

    If none provided, weights init to 1. otherwise, weights are normalized.
    """
    if sample_weight is None:
        sample_weight = numpy.ones(data.shape[0])
    if len(sample_weight) != data.shape[0]:
        raise ValueError("len of sample_weight must be equal to len of data.")
    sample_weight /= numpy.sum(sample_weight)
    return sample_weight


def mean_riemann(covmats, tol=10e-9, maxiter=50, init=None,
                 sample_weight=None):
    """Return the mean covariance matrix according to the Riemannian metric.

    The procedure is similar to a gradient descent minimizing the sum of
    riemannian distance to the mean.

    .. math::
            \mathbf{C} = \\arg\min{(\sum_i \delta_R ( \mathbf{C} , \mathbf{C}_i)^2)}  # noqa

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param init: A covariance matrix used to initialize the gradient descent. If None the Arithmetic mean is used
    :param sample_weight: the weight of each sample
    :returns: the mean covariance matrix

    """
    # init
    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    if init is None:
        C = numpy.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    nu = 1.0
    tau = numpy.finfo(numpy.float64).max
    crit = numpy.finfo(numpy.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter) and (nu > tol):
        k = k + 1
        C12 = sqrtm(C)
        Cm12 = invsqrtm(C)
        J = numpy.zeros((Ne, Ne))

        for index in range(Nt):
            tmp = numpy.dot(numpy.dot(Cm12, covmats[index, :, :]), Cm12)
            J += sample_weight[index] * logm(tmp)

        crit = numpy.linalg.norm(J, ord='fro')
        h = nu * crit
        C = numpy.dot(numpy.dot(C12, expm(nu * J)), C12)
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    return C


def mean_logeuclid(covmats, sample_weight=None):
    """Return the mean covariance matrix according to the log-euclidean metric.

    .. math::
            \mathbf{C} = \exp{(\\frac{1}{N} \sum_i \log{\mathbf{C}_i})}

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param sample_weight: the weight of each sample

    :returns: the mean covariance matrix

    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    T = numpy.zeros((Ne, Ne))
    for index in range(Nt):
        T += sample_weight[index] * logm(covmats[index, :, :])
    C = expm(T)

    return C


def mean_kullback_sym(covmats, sample_weight=None):
    """Return the mean covariance matrix according to KL divergence.

    This mean is the geometric mean between the Arithmetic and the Harmonic
    mean, as shown in Moakher, Maher, and Philipp G. Batchelor. "Symmetric
    positive-definite matrices: From geometry to applications and
    visualization." In Visualization and Processing of Tensor Fields, pp.
    285-298. Springer Berlin Heidelberg, 2006.

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param sample_weight: the weight of each sample

    :returns: the mean covariance matrix

    """
    C_Arithmetic = mean_euclid(covmats, sample_weight)
    C_Harmonic = mean_harmonic(covmats, sample_weight)
    C = geodesic_riemann(C_Arithmetic, C_Harmonic, 0.5)

    return C


def mean_harmonic(covmats, sample_weight=None):
    """Return the harmonic mean of a set of covariance matrices.

    .. math::
            \mathbf{C} = (\\frac{1}{N} \sum_i {\mathbf{C}_i}^{-1})^{-1}

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param sample_weight: the weight of each sample

    :returns: the mean covariance matrix

    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    T = numpy.zeros((Ne, Ne))
    for index in range(Nt):
        T += sample_weight[index] * numpy.linalg.inv(covmats[index, :, :])
    C = numpy.linalg.inv(T)

    return C


def mean_logdet(covmats, tol=10e-5, maxiter=50, init=None, sample_weight=None):
    """Return the mean covariance matrix according to the logdet metric.

    This is an iterative procedure where the update is:

    .. math::
            \mathbf{C} = \left(\sum_i \left( 0.5 \mathbf{C} + 0.5 \mathbf{C}_i \\right)^{-1} \\right)^{-1}  # noqa

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param init: A covariance matrix used to initialize the iterative procedure. If None the Arithmetic mean is used
    :param sample_weight: the weight of each sample

    :returns: the mean covariance matrix

    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    if init is None:
        C = numpy.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    crit = numpy.finfo(numpy.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter):
        k = k + 1

        J = numpy.zeros((Ne, Ne))

        for index, Ci in enumerate(covmats):
            J += sample_weight[index] * numpy.linalg.inv(0.5 * Ci + 0.5 * C)

        Cnew = numpy.linalg.inv(J)
        crit = numpy.linalg.norm(Cnew - C, ord='fro')

        C = Cnew
    return C


def mean_wasserstein(covmats, tol=10e-4, maxiter=50, init=None,
                     sample_weight=None):
    """Return the mean covariance matrix according to the wasserstein metric.

    This is an iterative procedure where the update is [1]:

    .. math::
            \mathbf{K} = \left(\sum_i \left( \mathbf{K} \mathbf{C}_i \mathbf{K} \\right)^{1/2} \\right)^{1/2}  # noqa

    with :math:`\mathbf{K} = \mathbf{C}^{1/2}`.

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param init: A covariance matrix used to initialize the iterative procedure. If None the Arithmetic mean is used
    :param sample_weight: the weight of each sample

    :returns: the mean covariance matrix

    References
    ----------
    [1] Barbaresco, F. "Geometric Radar Processing based on Frechet distance:
    Information geometry versus Optimal Transport Theory", Radar Symposium
    (IRS), 2011 Proceedings International.
    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    if init is None:
        C = numpy.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    K = sqrtm(C)
    crit = numpy.finfo(numpy.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter):
        k = k + 1

        J = numpy.zeros((Ne, Ne))

        for index, Ci in enumerate(covmats):
            tmp = numpy.dot(numpy.dot(K, Ci), K)
            J += sample_weight[index] * sqrtm(tmp)

        Knew = sqrtm(J)
        crit = numpy.linalg.norm(Knew - K, ord='fro')
        K = Knew
    if k == maxiter:
        print('Max iter reach')
    C = numpy.dot(K, K)
    return C


def mean_euclid(covmats, sample_weight=None):
    """Return the mean covariance matrix according to the euclidean metric :

    .. math::
            \mathbf{C} = \\frac{1}{N} \sum_i \mathbf{C}_i

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param sample_weight: the weight of each sample

    :returns: the mean covariance matrix

    """
    return numpy.average(covmats, axis=0, weights=sample_weight)


def mean_ale(covmats, tol=10e-7, maxiter=50, sample_weight=None):
    """Return the mean covariance matrix according using the AJD-based
    log-Euclidean Mean (ALE). See [1].

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param sample_weight: the weight of each sample

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
    Nt, Ne, Ne = covmats.shape
    crit = numpy.inf
    k = 0

    # init with AJD
    B, _ = ajd_pham(covmats)
    while (crit > tol) and (k < maxiter):
        k += 1
        J = numpy.zeros((Ne, Ne))

        for index, Ci in enumerate(covmats):
            tmp = logm(numpy.dot(numpy.dot(B.T, Ci), B))
            J += sample_weight[index] * tmp

        update = numpy.diag(numpy.diag(expm(J)))
        B = numpy.dot(B, invsqrtm(update))

        crit = distance_riemann(numpy.eye(Ne), update)

    A = numpy.linalg.inv(B)

    J = numpy.zeros((Ne, Ne))
    for index, Ci in enumerate(covmats):
        tmp = logm(numpy.dot(numpy.dot(B.T, Ci), B))
        J += sample_weight[index] * tmp

    C = numpy.dot(numpy.dot(A.T, expm(J)), A)
    return C


def mean_identity(covmats, sample_weight=None):
    """Return the identity matrix corresponding to the covmats sit size

    .. math::
            \mathbf{C} = \mathbf{I}_d

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :returns: the identity matrix of size Nchannels

    """
    C = numpy.eye(covmats.shape[1])
    return C


def mean_covariance(covmats, metric='riemann', sample_weight=None, *args):
    """Return the mean covariance matrix according to the metric


    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' , 'logeuclid' , 'euclid' , 'logdet', 'identity', 'wasserstein', 'ale',  # noqa
    'harmonic', 'kullback_sym' or a callable function
    :param sample_weight: the weight of each sample
    :param args: the argument passed to the sub function
    :returns: the mean covariance matrix

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
                'kullback_sym': mean_kullback_sym}


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
