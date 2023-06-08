"""Distances between SPD/HPD matrices."""

import numpy as np
from scipy.linalg import eigvalsh, solve

from .base import logm, sqrtm, invsqrtm


def _check_inputs(A, B):
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise ValueError("Inputs must be ndarrays")
    if not A.shape == B.shape:
        raise ValueError("Inputs must have equal dimensions")
    if A.ndim < 2:
        raise ValueError("Inputs must be at least a 2D ndarray")


def _recursive(fun, A, B, *args, **kwargs):
    """Recursive function with two inputs."""
    if A.ndim == 2:
        return fun(A, B, *args, **kwargs)
    else:
        return np.asarray(
            [_recursive(fun, a, b, *args, **kwargs) for a, b in zip(A, B)]
        )


###############################################################################
# Distances between matrices


def distance_euclid(A, B, squared=False):
    r"""Euclidean distance between matrices.

    The Euclidean distance between two matrices :math:`\mathbf{A}` and
    :math:`\mathbf{B}` is defined as the Frobenius norm of the difference of
    the two matrices:

    .. math::
        d(\mathbf{A},\mathbf{B}) = \Vert \mathbf{A} - \mathbf{B} \Vert_F

    Parameters
    ----------
    A : ndarray, shape (..., n, m)
        First matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, m)
        Second matrices, same dimensions as A.
    squared : bool, default False
        Return squared distance.

        .. versionadded:: 0.5

    Returns
    -------
    d : float or ndarray, shape (...,)
        Euclidean distance between A and B.

    See Also
    --------
    distance
    """
    _check_inputs(A, B)
    d = np.linalg.norm(A - B, ord='fro', axis=(-2, -1))
    return d ** 2 if squared else d


def distance_harmonic(A, B, squared=False):
    r"""Harmonic distance between invertible matrices.

    The harmonic distance between two invertible matrices :math:`\mathbf{A}`
    and :math:`\mathbf{B}` is:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        \Vert \mathbf{A}^{-1} - \mathbf{B}^{-1} \Vert_F

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First invertible matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second invertible matrices, same dimensions as A.
    squared : bool, default False
        Return squared distance.

        .. versionadded:: 0.5

    Returns
    -------
    d : float or ndarray, shape (...,)
        Harmonic distance between A and B.

    See Also
    --------
    distance
    """
    return distance_euclid(np.linalg.inv(A), np.linalg.inv(B), squared=squared)


def distance_kullback(A, B, squared=False):
    r"""Kullback-Leibler divergence between SPD/HPD matrices.

    The left Kullback-Leibler divergence between two SPD/HPD matrices
    :math:`\mathbf{A}` and :math:`\mathbf{B}` is [1]_:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        \frac{1}{2} \left( \text{tr}(\mathbf{B}^{-1}\mathbf{A}) - n
        + \log \left( \frac{\det(\mathbf{B})}{\det(\mathbf{A})}\right) \right)

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices, same dimensions as A.
    squared : bool, default False
        Return squared distance.

        .. versionadded:: 0.5

    Returns
    -------
    d : float or ndarray, shape (...,)
        Left Kullback-Leibler divergence between A and B.

    See Also
    --------
    distance

    References
    ----------
    .. [1] `On information and sufficiency
        <https://www.jstor.org/stable/2236703>`_
        S. Kullback S, R. Leibler.
        The Annals of Mathematical Statistics, 1951, 22 (1), pp. 79-86
    """
    _check_inputs(A, B)
    n = A.shape[-1]
    tr = np.trace(_recursive(solve, B, A, assume_a='pos'), axis1=-2, axis2=-1)
    logdet = np.linalg.slogdet(B)[1] - np.linalg.slogdet(A)[1]
    d = 0.5 * (tr - n + logdet)
    return d ** 2 if squared else d


def distance_kullback_right(A, B, squared=False):
    """Wrapper for right Kullback-Leibler divergence."""
    return distance_kullback(B, A, squared=squared)


def distance_kullback_sym(A, B, squared=False):
    r"""Symmetrized Kullback-Leibler divergence between SPD/HPD matrices.

    The symmetrized Kullback-Leibler divergence between two SPD/HPD matrices
    :math:`\mathbf{A}` and :math:`\mathbf{B}` is the sum of left and right
    Kullback-Leibler divergences.
    It is also called Jeffreys divergence [1]_.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices, same dimensions as A.
    squared : bool, default False
        Return squared distance.

        .. versionadded:: 0.5

    Returns
    -------
    d : float or ndarray, shape (...,)
        Symmetrized Kullback-Leibler divergence between A and B.

    See Also
    --------
    distance

    References
    ----------
    .. [1] `An invariant form for the prior probability in estimation problems
        <https://www.jstor.org/stable/97883>`_
        H. Jeffreys.
        Proceedings of the Royal Society of London A: mathematical, physical
        and engineering sciences, 1946, 186 (1007), pp. 453-461
    """
    d = distance_kullback(A, B) + distance_kullback_right(A, B)
    return d ** 2 if squared else d


def distance_logdet(A, B, squared=False):
    r"""Log-det distance between SPD/HPD matrices.

    The log-det distance between two SPD/HPD matrices :math:`\mathbf{A}` and
    :math:`\mathbf{B}` is [1]_:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        \sqrt{\log(\det \left( \frac{\mathbf{A}+\mathbf{B}}{2} \right))
        - \frac{1}{2} \log(\det(\mathbf{A} \mathbf{B}))}

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices, same dimensions as A.
    squared : bool, default False
        Return squared distance.

        .. versionadded:: 0.5

    Returns
    -------
    d : float or ndarray, shape (...,)
        Log-det distance between A and B.

    See Also
    --------
    distance

    References
    ----------
    .. [1] `Matrix nearness problems with Bregman divergences
        <https://epubs.siam.org/doi/abs/10.1137/060649021>`_
        I.S. Dhillon, J.A. Tropp.
        SIAM J Matrix Anal Appl, 2007, 29 (4), pp. 1120-1146
    """
    _check_inputs(A, B)
    logdet_ApB = np.linalg.slogdet((A + B) / 2.0)[1]
    logdet_AxB = np.linalg.slogdet(A @ B)[1]
    d2 = logdet_ApB - 0.5 * logdet_AxB
    d2 = np.maximum(0, d2)
    return d2 if squared else np.sqrt(d2)


def distance_logeuclid(A, B, squared=False):
    r"""Log-Euclidean distance between SPD/HPD matrices.

    The Log-Euclidean distance between two SPD/HPD matrices :math:`\mathbf{A}`
    and :math:`\mathbf{B}` is [1]_:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        \Vert \log(\mathbf{A}) - \log(\mathbf{B}) \Vert_F

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices, same dimensions as A.
    squared : bool, default False
        Return squared distance.

        .. versionadded:: 0.5

    Returns
    -------
    d : float or ndarray, shape (...,)
        Log-Euclidean distance between A and B.

    See Also
    --------
    distance

    References
    ----------
    .. [1] `Geometric means in a novel vector space structure on symmetric
        positive-definite matrices
        <https://epubs.siam.org/doi/abs/10.1137/050637996>`_
        V. Arsigny, P. Fillard, X. Pennec, N. Ayache.
        SIAM J Matrix Anal Appl, 2007, 29 (1), pp. 328-347
    """
    return distance_euclid(logm(A), logm(B), squared=squared)


def distance_riemann(A, B, squared=False):
    r"""Affine-invariant Riemannian distance between SPD/HPD matrices.

    The affine-invariant Riemannian distance between two SPD/HPD matrices
    :math:`\mathbf{A}` and :math:`\mathbf{B}` is [1]_:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        {\left( \sum_i \log(\lambda_i)^2 \right)}^{1/2}

    where :math:`\lambda_i` are the joint eigenvalues of :math:`\mathbf{A}` and
    :math:`\mathbf{B}`.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices, same dimensions as A.
    squared : bool, default False
        Return squared distance.

        .. versionadded:: 0.5

    Returns
    -------
    d : float or ndarray, shape (...,)
        Affine-invariant Riemannian distance between A and B.

    See Also
    --------
    distance

    References
    ----------
    .. [1] `A differential geometric approach to the geometric mean of
        symmetric positive-definite matrices
        <https://epubs.siam.org/doi/10.1137/S0895479803436937>`_
        M. Moakher. SIAM J Matrix Anal Appl, 2005, 26 (3), pp. 735-747
    """
    _check_inputs(A, B)
    d2 = (np.log(_recursive(eigvalsh, A, B))**2).sum(axis=-1)
    return d2 if squared else np.sqrt(d2)


def distance_wasserstein(A, B, squared=False):
    r"""Wasserstein distance between SPSD/HPSD matrices.

    The Wasserstein distance between two SPSD/HPSD matrices :math:`\mathbf{A}`
    and :math:`\mathbf{B}` is [1]_ [2]_:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        \sqrt{ \text{tr} \left(\mathbf{A} + \mathbf{B}
        - 2(\mathbf{B}^{1/2} \mathbf{A} \mathbf{B}^{1/2})^{1/2} \right) }

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPSD/HPSD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPSD/HPSD matrices, same dimensions as A.
    squared : bool, default False
        Return squared distance.

        .. versionadded:: 0.5

    Returns
    -------
    d : float or ndarray, shape (...,)
        Wasserstein distance between A and B.

    See Also
    --------
    distance

    References
    ----------
    .. [1] `Optimal transport: old and new
        <https://link.springer.com/book/10.1007/978-3-540-71050-9>`_
        C. Villani. Springer Science & Business Media, 2008, vol. 338
    .. [2] `An extension of Kakutani's theorem on infinite product measures to
        the tensor product of semifinite w*-algebras
        <https://www.ams.org/journals/tran/1969-135-00/S0002-9947-1969-0236719-2/S0002-9947-1969-0236719-2.pdf>`_
        D. Bures. Trans Am Math Soc, 1969, 135, pp. 199-212
    """  # noqa
    _check_inputs(A, B)
    B12 = sqrtm(B)
    d2 = np.trace(A + B - 2 * sqrtm(B12 @ A @ B12), axis1=-2, axis2=-1)
    d2 = np.maximum(0, d2)
    return d2 if squared else np.sqrt(d2)


###############################################################################


distance_functions = {
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


def _check_distance_function(metric):
    """Check distance function."""
    if isinstance(metric, str):
        if metric not in distance_functions.keys():
            raise ValueError(f"Unknown distance metric '{metric}'")
        else:
            metric = distance_functions[metric]
    elif not hasattr(metric, '__call__'):
        raise ValueError("Distance metric must be a function or a string "
                         f"(Got {type(metric)}.")
    return metric


def distance(A, B, metric='riemann', squared=False):
    """Distance between matrices according to a metric.

    Compute the distance between two matrices A and B according to a metric
    [1]_, or between a set of matrices A and another matrix B.

    Parameters
    ----------
    A : ndarray, shape (n, n) or shape (n_matrices, n, n)
        First matrix, or set of matrices.
    B : ndarray, shape (n, n)
        Second matrix.
    metric : string, default='riemann'
        The metric for distance, can be: 'euclid', 'harmonic', 'kullback',
        'kullback_right', 'kullback_sym', 'logdet', 'logeuclid', 'riemann',
        'wasserstein', or a callable function.
    squared : bool, default False
        Return squared distance.

        .. versionadded:: 0.5

    Returns
    -------
    d : float or ndarray, shape (n_matrices, 1)
        Distance between A and B.

    References
    ----------
    .. [1] `Review of Riemannian distances and divergences, applied to
        SSVEP-based BCI
        <https://hal.archives-ouvertes.fr/LISV/hal-03015762v1>`_
        S. Chevallier, E. K. Kalunga, Q. Barthélemy, E. Monacelli.
        Neuroinformatics, Springer, 2021, 19 (1), pp.93-106
    """
    distance_function = _check_distance_function(metric)

    shape_A, shape_B = A.shape, B.shape
    if shape_A == shape_B:
        d = distance_function(A, B, squared=squared)
    elif len(shape_A) == 3 and len(shape_B) == 2:
        d = np.empty((shape_A[0], 1))
        for i in range(shape_A[0]):
            d[i] = distance_function(A[i], B, squared=squared)
    else:
        raise ValueError("Inputs have incompatible dimensions.")

    return d


def pairwise_distance(X, Y=None, metric='riemann', squared=False):
    """Pairwise distance matrix.

    Compute the matrix of distances between pairs of elements of X and Y.

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of matrices. If None, Y is set to X.
    metric : string, default='riemann'
        The metric for distance, can be: 'euclid', 'harmonic', 'kullback',
        'kullback_right', 'kullback_sym', 'logdet', 'logeuclid', 'riemann',
        'wasserstein', or a callable function.
    squared : bool, default False
        Return squared distance.

        .. versionadded:: 0.5

    Returns
    -------
    dist : ndarray, shape (n_matrices_X, n_matrices_X) or (n_matrices_X, \
            n_matrices_Y)
        Distances between pairs of elements of X if Y is None, or between
        elements of X and Y.

    See Also
    --------
    distance
    """
    n_matrices_X, _, _ = X.shape

    # compute full pairwise matrix for non-symmetric metrics
    if Y is None and metric in ["kullback", "kullback_right"]:
        Y = X

    if Y is None:
        dist = np.zeros((n_matrices_X, n_matrices_X))
        for i in range(n_matrices_X):
            for j in range(i + 1, n_matrices_X):
                dist[i, j] = distance(X[i], X[j], metric, squared=squared)
        dist += dist.T
    else:
        n_matrices_Y, _, _ = Y.shape
        dist = np.empty((n_matrices_X, n_matrices_Y))
        for i in range(n_matrices_X):
            for j in range(n_matrices_Y):
                dist[i, j] = distance(X[i], Y[j], metric, squared=squared)

    return dist


###############################################################################
# Distances between vectors and matrices


def distance_mahalanobis(X, cov, mean=None, squared=False):
    r"""Mahalanobis distance between vectors and a Gaussian distribution.

    The Mahalanobis distance between a vector :math:`x \in \mathbb{C}^n` and a
    multivariate Gaussian distribution :math:`\mathcal{N}(\mu, C)`, with mean
    vector :math:`\mu \in \mathbb{C}^n` and covariance matrix
    :math:`C \in \mathbb{C}^{n \times n}` , is:

    .. math::
        d(x, \mathcal{N}(\mu, C)) = \sqrt{ (x - \mu)^H C^{-1} (x - \mu) }

    Parameters
    ----------
    X : ndarray, shape (n, n_vectors)
        Vectors.
    cov : ndarray, shape (n, n)
        Covariance matrix of the multivariate Gaussian distribution.
    mean : None | ndarray, shape (n, 1), default=None
        Mean vector of the multivariate Gaussian distribution.
        If None, distribution is considered as centered.
    squared : bool, default False
        Return squared distance.

        .. versionadded:: 0.5

    Returns
    -------
    d : ndarray, shape (n_vectors,)
        Mahalanobis distances.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html
    """  # noqa
    if mean is not None:
        X -= mean

    Xw = invsqrtm(cov) @ X
    d2 = np.einsum('ij,ji->i', Xw.conj().T, Xw).real
    return d2 if squared else np.sqrt(d2)
