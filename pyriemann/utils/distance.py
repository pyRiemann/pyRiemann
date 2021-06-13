"""Distance utils."""
import numpy as np
from scipy.linalg import eigvalsh

from .base import logm, sqrtm


def distance_kullback(A, B):
    """Kullback leibler divergence between two covariance matrices A and B.

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Kullback leibler divergence between A and B

    """
    dim = A.shape[0]
    logdet = np.log(np.linalg.det(B) / np.linalg.det(A))
    kl = np.trace(np.dot(np.linalg.inv(B), A)) - dim + logdet
    return 0.5 * kl


def distance_kullback_right(A, B):
    """wrapper for right kullblack leibler div."""
    return distance_kullback(B, A)


def distance_kullback_sym(A, B):
    """Symetrized kullback leibler divergence."""
    return distance_kullback(A, B) + distance_kullback_right(A, B)


def distance_euclid(A, B):
    r"""Euclidean distance between two covariance matrices A and B.

    The Euclidean distance is defined by the Froebenius norm between the two
    matrices.

    .. math::
        d = \Vert \mathbf{A} - \mathbf{B} \Vert_F

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Eclidean distance between A and B

    """
    return np.linalg.norm(A - B, ord='fro')


def distance_logeuclid(A, B):
    r"""Log Euclidean distance between two covariance matrices A and B.

    .. math::
        d = \Vert \log(\mathbf{A}) - \log(\mathbf{B}) \Vert_F

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Log-Eclidean distance between A and B

    """
    return distance_euclid(logm(A), logm(B))


def distance_riemann(A, B):
    r"""Riemannian distance between two covariance matrices A and B.

    .. math::
        d = {\left( \sum_i \log(\lambda_i)^2 \right)}^{-1/2}

    where :math:`\lambda_i` are the joint eigenvalues of A and B

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Riemannian distance between A and B

    """
    return np.sqrt((np.log(eigvalsh(A, B))**2).sum())


def distance_logdet(A, B):
    r"""Log-det distance between two covariance matrices A and B.

    .. math::
        d = \sqrt{\log(\det(\frac{\mathbf{A}+\mathbf{B}}{2})) - \frac{1}{2} \log(\det(\mathbf{A}) \det(\mathbf{B}))}

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Log-Euclid distance between A and B

    """  # noqa
    return np.sqrt(np.log(np.linalg.det(
        (A + B) / 2.0)) - 0.5 *
        np.log(np.linalg.det(A)*np.linalg.det(B)))


def distance_wasserstein(A, B):
    r"""Wasserstein distance between two covariances matrices.

    .. math::
        d = \left( {tr(A + B - 2(A^{1/2}BA^{1/2})^{1/2})} \right)^{1/2}

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Wasserstein distance between A and B

    """
    B12 = sqrtm(B)
    C = sqrtm(np.dot(np.dot(B12, A), B12))
    return np.sqrt(np.trace(A + B - 2*C))


def distance(A, B, metric='riemann'):
    """Distance between two covariance matrices A and B according to the
    metric.

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' ,
        'logeuclid' , 'euclid' , 'logdet', 'kullback', 'kullback_right',
        'kullback_sym'.
    :returns: the distance between A and B

    """
    if callable(metric):
        distance_function = metric
    else:
        distance_function = distance_methods[metric]

    if len(A.shape) == 3:
        d = np.empty((len(A), 1))
        for i in range(len(A)):
            d[i] = distance_function(A[i], B)
    else:
        d = distance_function(A, B)

    return d


def pairwise_distance(X, Y=None, metric='riemann'):
    """Pairwise distance matrix

    :param A: fist Covariances instance
    :param B: second Covariances instance (optional)
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' ,
    'logeuclid' , 'euclid' , 'logdet', 'kullback', 'kullback_right',
    'kullback_sym'.
    :returns: the distances between pairs of elements of X or between elements
    of X and Y.

    """
    Ntx, _, _ = X.shape

    if Y is None:
        dist = np.zeros((Ntx, Ntx))
        for i in range(Ntx):
            for j in range(i + 1, Ntx):
                dist[i, j] = distance(X[i], X[j], metric)
        dist += dist.T
    else:
        Nty, _, _ = Y.shape
        dist = np.empty((Ntx, Nty))
        for i in range(Ntx):
            for j in range(Nty):
                dist[i, j] = distance(X[i], Y[j], metric)
    return dist


distance_methods = {'riemann': distance_riemann,
                    'logeuclid': distance_logeuclid,
                    'euclid': distance_euclid,
                    'logdet': distance_logdet,
                    'kullback': distance_kullback,
                    'kullback_right': distance_kullback_right,
                    'kullback_sym': distance_kullback_sym,
                    'wasserstein': distance_wasserstein}


def _check_distance_method(method):
    """checks methods """
    if isinstance(method, str):
        if method not in distance_methods.keys():
            raise ValueError('Unknown mean method')
        else:
            method = distance_methods[method]
    elif not hasattr(method, '__call__'):
        raise ValueError('distance method must be a function or a string.')
    return method
