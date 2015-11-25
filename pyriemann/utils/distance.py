"""Distance utils."""
import numpy
from scipy.linalg import eigvalsh

from .base import logm, sqrtm


def distance_kullback(A, B):
    """Kullback leibler divergence between two covariance matrices A and B.

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Kullback leibler divergence between A and B

    """
    dim = A.shape[0]
    logdet = numpy.log(numpy.linalg.det(B) / numpy.linalg.det(A))
    kl = numpy.trace(numpy.dot(numpy.linalg.inv(B), A)) - dim + logdet
    return 0.5 * kl


def distance_kullback_right(A, B):
    """wrapper for right kullblack leibler div."""
    return distance_kullback(B, A)


def distance_kullback_sym(A, B):
    """Symetrized kullback leibler divergence."""
    return distance_kullback(A, B) + distance_kullback_right(A, B)


def distance_euclid(A, B):
    """Euclidean distance between two covariance matrices A and B.

    The Euclidean distance is defined by the Froebenius norm between the two
    matrices.

    .. math::
            d = \Vert \mathbf{A} - \mathbf{B} \Vert_F

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Eclidean distance between A and B

    """
    return numpy.linalg.norm(A - B, ord='fro')


def distance_logeuclid(A, B):
    """Log Euclidean distance between two covariance matrices A and B.

    .. math::
            d = \Vert \log(\mathbf{A}) - \log(\mathbf{B}) \Vert_F

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Log-Eclidean distance between A and B

    """
    return distance_euclid(logm(A), logm(B))


def distance_riemann(A, B):
    """Riemannian distance between two covariance matrices A and B.

    .. math::
            d = {\left( \sum_i \log(\lambda_i)^2 \\right)}^{-1/2}

    where :math:`\lambda_i` are the joint eigenvalues of A and B

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Riemannian distance between A and B

    """
    return numpy.sqrt((numpy.log(eigvalsh(A, B))**2).sum())


def distance_logdet(A, B):
    """Log-det distance between two covariance matrices A and B.

    .. math::
            d = \sqrt{\left(\log(\det(\\frac{\mathbf{A}+\mathbf{B}}{2})) - 0.5 \\times \log(\det(\mathbf{A}) \det(\mathbf{B}))\\right)}

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Log-Euclid distance between A and B

    """
    return numpy.sqrt(numpy.log(numpy.linalg.det(
        (A + B) / 2.0)) - 0.5 * numpy.log(numpy.linalg.det(A)*numpy.linalg.det(B)))


def distance_wasserstein(A, B):
    """Wasserstein distance between two covariances matrices.

    .. math::
        d = \left( {tr(A + B - 2(A^{1/2}BA^{1/2})^{1/2})}\\right )^{1/2}

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Wasserstein distance between A and B

    """
    B12 = sqrtm(B)
    C = sqrtm(numpy.dot(numpy.dot(B12, A), B12))
    return numpy.sqrt(numpy.trace(A + B - 2*C))


def distance(A, B, metric='riemann'):
    """Distance between two covariance matrices A and B according to the metric.

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' ,
    'logeuclid' , 'euclid' , 'logdet', 'kullback', 'kullback_right',
    'kullback_sym'.
    :returns: the distance between A and B

    """
    distance_methods = {'riemann': distance_riemann,
                        'logeuclid': distance_logeuclid,
                        'euclid': distance_euclid,
                        'logdet': distance_logdet,
                        'kullback': distance_kullback,
                        'kullback_right': distance_kullback_right,
                        'kullback_sym': distance_kullback_sym,
                        'wasserstein': distance_wasserstein}
    if len(A.shape) == 3:
        d = numpy.empty((len(A), 1))
        for i in range(len(A)):
            d[i] = distance_methods[metric](A[i], B)
    else:
        d = distance_methods[metric](A, B)

    return d
