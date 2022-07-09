"""Distances between SPD matrices."""

import numpy as np
from scipy.linalg import eigvalsh, solve

from .base import logm, sqrtm


def _recursive(fun, A, B, *args, **kwargs):
    """Recursive fuction with two inputs."""
    if not isinstance(A, np.ndarray) or A.ndim < 2:
        raise ValueError('Input must be at least a 2D ndarray')
    elif A.ndim == 2:
        return fun(A, B, *args, **kwargs)
    else:
        return np.asarray(
            [_recursive(fun, a, b, *args, **kwargs) for a, b in zip(A, B)]
        )


def distance_euclid(A, B):
    r"""Euclidean distance between SPD matrices.

    The Euclidean distance between two SPD matrices A and B is defined
    as the Frobenius norm of the difference of the two matrices:

    .. math::
        d(\mathbf{A},\mathbf{B}) = \Vert \mathbf{A} - \mathbf{B} \Vert_F

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD matrices, same dimensions as A.

    Returns
    -------
    d : ndarray, shape (...,) or float
        Euclidean distance between A and B.
    """
    return np.linalg.norm(A - B, ord='fro', axis=(-2, -1))


def distance_harmonic(A, B):
    r"""Harmonic distance between SPD matrices.

    The harmonic distance between two SPD matrices A and B is:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        \Vert \mathbf{A}^{-1} - \mathbf{B}^{-1} \Vert_F

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD matrices, same dimensions as A.

    Returns
    -------
    d : ndarray, shape (...,) or float
        Harmonic distance between A and B.
    """
    return distance_euclid(np.linalg.inv(A), np.linalg.inv(B))


def distance_kullback(A, B):
    r"""Kullback-Leibler divergence between SPD matrices.

    The left Kullback-Leibler divergence between two SPD matrices A and B is:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        \frac{1}{2} \left( \text{tr}(\mathbf{B}^{-1}\mathbf{A}) - n
        + \log(\frac{\det(\mathbf{B})}{\det(\mathbf{A})}) \right)

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD matrices, same dimensions as A.

    Returns
    -------
    d : ndarray, shape (...,) or float
        Left Kullback-Leibler divergence between A and B.
    """
    n = A.shape[-1]
    tr = np.trace(_recursive(solve, B, A, assume_a='pos'), axis1=-2, axis2=-1)
    logdet = np.log(np.linalg.det(B) / np.linalg.det(A))
    return 0.5 * (tr - n + logdet)


def distance_kullback_right(A, B):
    """Wrapper for right Kullback-Leibler divergence."""
    return distance_kullback(B, A)


def distance_kullback_sym(A, B):
    """Symmetrized Kullback-Leibler divergence between SPD matrices.

    The symmetrized Kullback-Leibler divergence between two SPD matrices A and
    B is the sum of left and right Kullback-Leibler divergences.
    """
    return distance_kullback(A, B) + distance_kullback_right(A, B)


def distance_logdet(A, B):
    r"""Log-det distance between SPD matrices.

    The log-det distance between two SPD matrices A and B is:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        \sqrt{\log(\det(\frac{\mathbf{A}+\mathbf{B}}{2}))
        - \frac{1}{2} \log(\det(\mathbf{A} \mathbf{B}))}

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD matrices, same dimensions as A.

    Returns
    -------
    d : ndarray, shape (...,) or float
        Log-det distance between A and B.
    """
    _, logdet_ApB = np.linalg.slogdet((A + B) / 2.0)
    _, logdet_AxB = np.linalg.slogdet(A @ B)
    dist2 = logdet_ApB - 0.5 * logdet_AxB
    dist2 = np.maximum(0, dist2)
    return np.sqrt(dist2)


def distance_logeuclid(A, B):
    r"""Log-Euclidean distance between SPD matrices.

    The Log-Euclidean distance between two SPD matrices A and B is:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        \Vert \log(\mathbf{A}) - \log(\mathbf{B}) \Vert_F

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD matrices, same dimensions as A.

    Returns
    -------
    d : ndarray, shape (...,) or float
        Log-Euclidean distance between A and B.
    """
    return distance_euclid(logm(A), logm(B))


def distance_riemann(A, B):
    r"""Affine-invariant Riemannian distance between SPD matrices.

    The affine-invariant Riemannian distance between two SPD matrices A and B
    is:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        {\left( \sum_i \log(\lambda_i)^2 \right)}^{1/2}

    where :math:`\lambda_i` are the joint eigenvalues of :math:`\mathbf{A}` and
    :math:`\mathbf{B}`.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD matrices, same dimensions as A.

    Returns
    -------
    d : ndarray, shape (...,) or float
        Affine-invariant Riemannian distance between A and B.
    """
    return np.sqrt((np.log(_recursive(eigvalsh, A, B))**2).sum(axis=-1))


def distance_wasserstein(A, B):
    r"""Wasserstein distance between SPD matrices.

    The Wasserstein distance between two SPD matrices A and B is:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        \sqrt{ \text{tr}(A + B - 2(B^{1/2} A B^{1/2})^{1/2}) }

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD matrices, same dimensions as A.

    Returns
    -------
    d : ndarray, shape (...,) or float
        Wasserstein distance between A and B.
    """
    B12 = sqrtm(B)
    dist2 = np.trace(A + B - 2 * sqrtm(B12 @ A @ B12), axis1=-2, axis2=-1)
    dist2 = np.maximum(0, dist2)
    return np.sqrt(dist2)


###############################################################################


distance_methods = {
    'euclid': distance_euclid,
    'harmonic': distance_harmonic,
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
            raise ValueError('Unknown distance method')
        else:
            method = distance_methods[method]
    elif not hasattr(method, '__call__'):
        raise ValueError('Distance method must be a function or a string.')
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
        The metric for distance, can be: 'euclid', 'harmonic', 'kullback',
        'kullback_right', 'kullback_sym', 'logdet', 'logeuclid', 'riemann',
        'wasserstein', or a callable function.

    Returns
    -------
    d : float or ndarray, shape (n_matrices, 1)
        The distance(s) between A and B.
    """
    if callable(metric):
        distance_function = metric
    else:
        distance_function = distance_methods[metric]

    shape_A = A.shape
    if len(shape_A) == B.ndim:
        d = distance_function(A, B)
    elif len(shape_A) == 3:
        d = np.empty((shape_A[0], 1))
        for i in range(shape_A[0]):
            d[i] = distance_function(A[i], B)
    else:
        raise ValueError("Inputs have incompatible dimensions.")

    return d


def pairwise_distance(X, Y=None, metric='riemann'):
    """Pairwise distance matrix.

    Compute the matrix of distances between pairs of elements of X and Y.

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    metric : string, default='riemann'
        The metric for distance, can be: 'euclid', 'harmonic', 'kullback',
        'kullback_right', 'kullback_sym', 'logdet', 'logeuclid', 'riemann',
        'wasserstein', or a callable function.

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
