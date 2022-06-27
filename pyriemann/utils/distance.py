"""Distances between SPD matrices."""

import numpy as np
from scipy.linalg import eigvalsh

from .base import logm, sqrtm


def distance_euclid(A, B):
    r"""Euclidean distance between two SPD matrices.

    Compute the Euclidean distance between two SPD matrices A and B, defined
    as the Frobenius norm of the difference of the two matrices:

    .. math::
        d(A,B) = \Vert \mathbf{A} - \mathbf{B} \Vert_F

    Parameters
    ----------
    A : ndarray, shape (n, n)
        First SPD matrix.
    B : ndarray, shape (n, n)
        Second SPD matrix.

    Returns
    -------
    d : float
        Euclidean distance between A and B.
    """
    return np.linalg.norm(A - B, ord='fro')


def distance_kullback(A, B):
    """Kullback-Leibler divergence between two SPD matrices.

    Compute the Kullback-Leibler divergence between two SPD matrices A and B.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        First SPD matrix.
    B : ndarray, shape (n, n)
        Second SPD matrix.

    Returns
    -------
    d : float
        Kullback-Leibler divergence between A and B.
    """
    n, _ = A.shape
    logdet = np.log(np.linalg.det(B) / np.linalg.det(A))
    kl = np.trace(np.dot(np.linalg.inv(B), A)) - n + logdet
    return 0.5 * kl


def distance_kullback_right(A, B):
    """Wrapper for right Kullback-Leibler divergence."""
    return distance_kullback(B, A)


def distance_kullback_sym(A, B):
    """Symmetrized Kullback-Leibler divergence between two SPD matrices."""
    return distance_kullback(A, B) + distance_kullback_right(A, B)


def distance_logdet(A, B):
    r"""Log-det distance between two SPD matrices.

    Compute the log-det distance between two SPD matrices A and B:

    .. math::
        d(A,B) = \sqrt{\log(\det(\frac{\mathbf{A}+\mathbf{B}}{2}))
                 - \frac{1}{2} \log(\det(\mathbf{A} \mathbf{B}))}

    Parameters
    ----------
    A : ndarray, shape (n, n)
        First SPD matrix.
    B : ndarray, shape (n, n)
        Second SPD matrix.

    Returns
    -------
    d : float
        Log-det distance between A and B.
    """
    _, logdet_ApB = np.linalg.slogdet((A + B) / 2.0)
    _, logdet_AxB = np.linalg.slogdet(np.dot(A, B))
    dist2 = logdet_ApB - 0.5 * logdet_AxB
    if dist2 > 0.0:
        return np.sqrt(dist2)
    else:
        return 0.0


def distance_logeuclid(A, B):
    r"""Log-Euclidean distance between two SPD matrices.

    Compute the Log-Euclidean distance between two SPD matrices A and B:

    .. math::
        d(A,B) = \Vert \log(\mathbf{A}) - \log(\mathbf{B}) \Vert_F

    Parameters
    ----------
    A : ndarray, shape (n, n)
        First SPD matrix.
    B : ndarray, shape (n, n)
        Second SPD matrix.

    Returns
    -------
    d : float
        Log-Euclidean distance between A and B.
    """
    return distance_euclid(logm(A), logm(B))


def distance_riemann(A, B):
    r"""Affine-invariant Riemannian distance between two SPD matrices.

    Compute the affine-invariant Riemannian distance between two SPD matrices A
    and B:

    .. math::
        d(A,B) = {\left( \sum_i \log(\lambda_i)^2 \right)}^{1/2}

    where :math:`\lambda_i` are the joint eigenvalues of A and B.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        First SPD matrix.
    B : ndarray, shape (n, n)
        Second SPD matrix.

    Returns
    -------
    d : float
        Affine-invariant Riemannian distance between A and B.
    """
    return np.sqrt((np.log(eigvalsh(A, B))**2).sum())


def distance_wasserstein(A, B):
    r"""Wasserstein distance between two SPD matrices.

    Compute the Wasserstein distance between two SPD matrices A and B:

    .. math::
        d(A,B) = \sqrt{ \text{tr}(A + B - 2(A^{1/2}BA^{1/2})^{1/2}) }

    Parameters
    ----------
    A : ndarray, shape (n, n)
        First SPD matrix.
    B : ndarray, shape (n, n)
        Second SPD matrix.

    Returns
    -------
    d : float
        Wasserstein distance between A and B.
    """
    B12 = sqrtm(B)
    C = sqrtm(np.dot(np.dot(B12, A), B12))
    return np.sqrt(np.trace(A + B - 2*C))


###############################################################################


distance_methods = {
    'euclid': distance_euclid,
    'kullback': distance_kullback,
    'kullback_right': distance_kullback_right,
    'kullback_sym': distance_kullback_sym,
    'logdet': distance_logdet,
    'logeuclid': distance_logeuclid,
    'riemann': distance_riemann,
    'wasserstein': distance_wasserstein,
}


def _check_distance_method(method):
    """Check distance methods."""
    if isinstance(method, str):
        if method not in distance_methods.keys():
            raise ValueError('Unknown mean method')
        else:
            method = distance_methods[method]
    elif not hasattr(method, '__call__'):
        raise ValueError('distance method must be a function or a string.')
    return method


def distance(A, B, metric='riemann'):
    """Distance between SPD matrices according to a metric.

    Compute the distance between two SPD matrices A and B according to a
    metric, or between a set of SPD matrices A and a SPD matrix B.

    Parameters
    ----------
    A : ndarray, shape (n, n) or shape (n_matrices, n, n)
        First SPD matrix.
    B : ndarray, shape (n, n)
        Second SPD matrix.
    metric : string, default='riemann'
        The metric for distance, can be: 'euclid', 'kullback',
        'kullback_right', 'kullback_sym', 'logdet', 'logeuclid', 'riemann',
        'wasserstein'.

    Returns
    -------
    d : float or ndarray, shape (n_matrices, 1)
        The distance(s) between A and B.
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
    """Pairwise distance matrix.

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    metric : string, default='riemann'
        The metric for distance, can be: 'euclid', 'kullback',
        'kullback_right', 'kullback_sym', 'logdet', 'logeuclid', 'riemann',
        'wasserstein'.

    Returns
    -------
    dist : ndarray, shape (n_matrices_X, n_matrices_X) or (n_matrices_X, \
            n_matrices_Y)
        The distances between pairs of elements of X if Y is None, or between
        elements of X and Y.
    """
    n_matrices_X, _, _ = X.shape

    if Y is None:
        dist = np.zeros((n_matrices_X, n_matrices_X))
        for i in range(n_matrices_X):
            for j in range(i + 1, n_matrices_X):
                dist[i, j] = distance(X[i], X[j], metric)
        dist += dist.T
    else:
        n_matrices_Y, _, _ = Y.shape
        dist = np.empty((n_matrices_X, n_matrices_Y))
        for i in range(n_matrices_X):
            for j in range(n_matrices_Y):
                dist[i, j] = distance(X[i], Y[j], metric)
    return dist
