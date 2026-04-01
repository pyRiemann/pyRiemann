"""Distances between SPD/HPD matrices."""

import numpy as np
from scipy.linalg import eigvalsh, solve

from ._backend import (
    pairwise_euclidean,
    check_matrix_pair,
    diag_indices,
    get_namespace,
    is_numpy_namespace,
    tril_indices,
    xpd,
)
from .base import _recursive, ctranspose, invsqrtm, logm, powm, sqrtm
from .test import is_real_type
from .utils import check_function


def _check_inputs(A, B):
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise ValueError("Inputs must be ndarrays")
    if not A.shape == B.shape:
        raise ValueError("Inputs must have equal dimensions")
    if A.ndim < 2:
        raise ValueError("Inputs must be at least a 2D ndarray")


###############################################################################
# Distances between matrices


def distance_chol(A, B, squared=False):
    r"""Cholesky distance between SPD/HPD matrices.

    The Cholesky distance between two SPD/HPD matrices :math:`\mathbf{A}`
    and :math:`\mathbf{B}` is [1]_:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        \Vert \text{chol}(\mathbf{A}) - \text{chol}(\mathbf{B}) \Vert_F

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices, same dimensions as A.
    squared : bool, default=False
        Return squared distance.

    Returns
    -------
    d : float or ndarray, shape (...,)
        Cholesky distance between A and B.

    Notes
    -----
    .. versionadded:: 0.7

    See Also
    --------
    distance

    References
    ----------
    .. [1] `Non-Euclidean statistics for covariance matrices, with applications
        to diffusion tensor imaging
        <https://doi.org/10.1214/09-AOAS249>`_
        I.L. Dryden, A. Koloydenko, D. Zhou.
        Ann Appl Stat, 2009, 3(3), pp. 1102-1123.
    """
    xp = get_namespace(A, B)
    return distance_euclid(
        xp.linalg.cholesky(A),
        xp.linalg.cholesky(B),
        squared=squared,
    )


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
    squared : bool, default=False
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
    xp = check_matrix_pair(A, B)
    d = xp.linalg.matrix_norm(A - B, ord="fro")
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
    squared : bool, default=False
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
    xp = get_namespace(A, B)
    eye_n = xp.eye(A.shape[-1], dtype=A.dtype, device=xpd(A))
    return distance_euclid(
        xp.linalg.solve(A, eye_n), 
        xp.linalg.solve(B, eye_n), 
        squared=squared,
    )


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
    squared : bool, default=False
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
    xp = check_matrix_pair(A, B, require_square=True)
    n = A.shape[-1]
    if is_numpy_namespace(xp) and A.shape == B.shape:
        tr = np.trace(
            _recursive(solve, B, A, assume_a='pos'), axis1=-2, axis2=-1,
        )
    else:
        # I could not find a way to compute the trace of the solution without computing
        # the whole solution, which is costly. If you know how to do it, please tell me.
        tr = xp.sum(xp.linalg.diagonal(xp.linalg.solve(B, A)), axis=-1)
    logdet = xp.linalg.slogdet(B)[1] - xp.linalg.slogdet(A)[1]
    d = 0.5 * xp.real(tr - n + logdet)
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
    squared : bool, default=False
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


def distance_logchol(A, B, squared=False):
    r"""Log-Cholesky distance between SPD/HPD matrices.

    The log-Cholesky distance between two SPD/HPD matrices :math:`\mathbf{A}`
    and :math:`\mathbf{B}` is [1]_:

    .. math::
        d(\mathbf{A},\mathbf{B}) = \left(
        \Vert \text{lower}(\text{chol}(\mathbf{A})) -
        \text{lower}(\text{chol}(\mathbf{B})) \Vert_F^2 +
        \Vert \log(\text{diag}(\text{chol}(\mathbf{A}))) -
        \log(\text{diag}(\text{chol}(\mathbf{B}))) \Vert_F^2
        \right)^\frac{1}{2}

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices, same dimensions as A.
    squared : bool, default=False
        Return squared distance.

    Returns
    -------
    d : float or ndarray, shape (...,)
        Log-Cholesky distance between A and B.

    Notes
    -----
    .. versionadded:: 0.7

    See Also
    --------
    distance

    References
    ----------
    .. [1] `Riemannian geometry of symmetric positive definite matrices via
        Cholesky decomposition
        <https://arxiv.org/pdf/1908.09326>`_
        Z. Lin. SIAM J Matrix Anal Appl, 2019, 40(4), pp. 1353-1370.
    """
    xp = check_matrix_pair(A, B)
    A_chol, B_chol = xp.linalg.cholesky(A), xp.linalg.cholesky(B)

    tri0, tri1 = tril_indices(A_chol.shape[-1], -1, xp=xp, like=A_chol)
    triangular_part = xp.linalg.vector_norm(
        A_chol[..., tri0, tri1] - B_chol[..., tri0, tri1], axis=-1,
    ) ** 2

    diag0, diag1 = diag_indices(A_chol.shape[-1], xp=xp, like=A_chol)
    diagonal_part = xp.linalg.vector_norm(
        xp.log(A_chol[..., diag0, diag1]) -
        xp.log(B_chol[..., diag0, diag1]), axis=-1,
    ) ** 2

    d2 = triangular_part + diagonal_part
    return d2 if squared else xp.sqrt(d2)


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
    squared : bool, default=False
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
    xp = check_matrix_pair(A, B, require_square=True)
    logdet_ApB = xp.linalg.slogdet((A + B) / 2.0)[1]
    logdet_AxB = xp.linalg.slogdet(A @ B)[1]
    d2 = logdet_ApB - 0.5 * logdet_AxB
    d2 = xp.maximum(xp.real(d2), xp.asarray(0, dtype=d2.dtype, device=xpd(d2)))
    return d2 if squared else xp.sqrt(d2)


def distance_logeuclid(A, B, squared=False):
    r"""Log-Euclidean distance between SPD/HPD matrices.

    The log-Euclidean distance between two SPD/HPD matrices :math:`\mathbf{A}`
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
    squared : bool, default=False
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


def distance_poweuclid(A, B, p, squared=False):
    r"""Power Euclidean distance between SPD/HPD matrices.

    The power Euclidean distance of order :math:`p` between two SPD/HPD
    matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` is [1]_:

    .. math::
        d(\mathbf{A},\mathbf{B}) = \frac{1}{|p|}
        \Vert \mathbf{A}^p - \mathbf{B}^p \Vert_F

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices, same dimensions as A.
    p : float
        Exponent. For p=0, it returns
        :func:`pyriemann.utils.distance.distance_logeuclid`.
    squared : bool, default=False
        Return squared distance.

    Returns
    -------
    d : float or ndarray, shape (...,)
        Power Euclidean distance between A and B.

    Notes
    -----
    .. versionadded:: 0.7

    See Also
    --------
    distance

    References
    ----------
    .. [1] `Power Euclidean metrics for covariance matrices with application to
        diffusion tensor imaging
        <https://arxiv.org/abs/1009.3045>`_
        I.L. Dryden, X. Pennec, & J.M. Peyrat. arXiv, 2010
    """
    if not isinstance(p, (int, float)):
        raise ValueError(f"Exponent p must be a scalar (Got {type(p)})")

    if p == 1:
        return distance_euclid(A, B, squared=squared)
    if p == 0:
        return distance_logeuclid(A, B, squared=squared)
    if p == -1:
        return distance_harmonic(A, B, squared=squared)

    return distance_euclid(
        powm(A, p),
        powm(B, p),
        squared=squared,
    ) / abs(p)


def distance_riemann(A, B, squared=False):
    r"""Affine-invariant Riemannian distance between SPD/HPD matrices.

    The affine-invariant Riemannian distance between two SPD/HPD matrices
    :math:`\mathbf{A}` and :math:`\mathbf{B}` is [1]_:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        \Vert \log(\mathbf{B}^{-1/2} \mathbf{A} \mathbf{B}^{-1/2}) \Vert_F =
        {\left( \sum_i \log(\lambda_i)^2 \right)}^{1/2}

    where :math:`\lambda_i` are the joint eigenvalues of :math:`\mathbf{A}` and
    :math:`\mathbf{B}`.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices, same dimensions as A.
    squared : bool, default=False
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
    .. [1] `A metric for covariance matrices
        <https://www.ipb.uni-bonn.de/pdfs/Forstner1999Metric.pdf>`_
        W. Förstner & B. Moonen.
        Geodesy-the Challenge of the 3rd Millennium, 2003
    """
    xp = check_matrix_pair(A, B)
    if is_numpy_namespace(xp) and A.shape == B.shape:
        # scipy eigvalsh(A, B) computes generalized eigenvalues directly
        d2 = (np.log(_recursive(eigvalsh, A, B))**2).sum(axis=-1)
    else:
        # torch has no generalized eigvalsh(A, B), so we reduce to
        # standard eigenvalues via Cholesky: L = chol(B), then
        # eigvalsh(L^{-1} A L^{-H}) gives the same joint eigenvalues.
        # This avoids the expensive invsqrtm (full eigen-decomposition).
        L = xp.linalg.cholesky(B)
        Y = xp.linalg.solve(L, A)                          # L^{-1} A
        Z = ctranspose(xp.linalg.solve(L, ctranspose(Y)))  # L^{-1} A L^{-H}
        eigvals = xp.linalg.eigvalsh(Z)
        d2 = xp.sum(xp.log(eigvals) ** 2, axis=-1)
    return d2 if squared else xp.sqrt(d2)


def distance_thompson(A, B, squared=False):
    r"""Thompson distance between SPD/HPD matrices.

    The Thompson distance between two SPD/HPD matrices :math:`\mathbf{A}` and
    :math:`\mathbf{B}` is [1]_:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        \Vert \log(\mathbf{B}^{-1/2} \mathbf{A} \mathbf{B}^{-1/2}) \Vert_2 =
        \max_i | \log(\lambda_i) |

    where :math:`\lambda_i` are the joint eigenvalues of :math:`\mathbf{A}` and
    :math:`\mathbf{B}`.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices, same dimensions as A.
    squared : bool, default=False
        Return squared distance.

    Returns
    -------
    d : float or ndarray, shape (...,)
        Thompson distance between A and B.

    Notes
    -----
    .. versionadded:: 0.10

    See Also
    --------
    distance

    References
    ----------
    .. [1] `On certain contraction mappings in a partially ordered vector space
        <https://www.cs.umd.edu/projects/reucaar/ThompsonGeom.pdf>`_
        A.C.Thompson. Proceedings of the American Mathematical Society, 1963.
    """
    xp = check_matrix_pair(A, B, require_square=True)
    if is_numpy_namespace(xp) and A.shape == B.shape:
        # scipy eigvalsh(A, B) computes generalized eigenvalues directly
        d = (np.abs(np.log(_recursive(eigvalsh, A, B)))).max(axis=-1)
    else:
        # Same Cholesky reduction as distance_riemann: L = chol(B),
        # eigvalsh(L^{-1} A L^{-H}) gives the joint eigenvalues.
        L = xp.linalg.cholesky(B)
        Y = xp.linalg.solve(L, A)
        Z = ctranspose(xp.linalg.solve(L, ctranspose(Y)))
        d = xp.max(xp.abs(xp.log(xp.linalg.eigvalsh(Z))), axis=-1)
    return d ** 2 if squared else d


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
    squared : bool, default=False
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
    xp = check_matrix_pair(A, B)
    B12 = sqrtm(B)
    d2 = xp.linalg.trace(A + B - 2 * sqrtm(B12 @ A @ B12))
    d2 = xp.maximum(xp.real(d2), xp.asarray(0, dtype=xp.real(d2).dtype,
                                             device=xpd(d2)))
    return d2 if squared else xp.sqrt(d2)


distance_functions = {
    "chol": distance_chol,
    "euclid": distance_euclid,
    "harmonic": distance_harmonic,
    "kullback": distance_kullback,
    "kullback_right": distance_kullback_right,
    "kullback_sym": distance_kullback_sym,
    "logchol": distance_logchol,
    "logdet": distance_logdet,
    "logeuclid": distance_logeuclid,
    "riemann": distance_riemann,
    "thompson": distance_thompson,
    "wasserstein": distance_wasserstein,
}


def distance(A, B, metric="riemann", squared=False):
    """Distance between matrices according to a metric.

    Compute the distance between two matrices A and B according to a metric
    [1]_, or between a set of matrices A and another matrix B.

    Parameters
    ----------
    A : ndarray, shape (n, n) or shape (n_matrices, n, n)
        First matrix, or set of matrices.
    B : ndarray, shape (n, n)
        Second matrix.
    metric : string | callable, default="riemann"
        Metric for distance, can be:
        "chol", "euclid", "harmonic", "kullback", "kullback_right",
        "kullback_sym", "logchol", "logdet", "logeuclid", "riemann",
        "thompson", "wasserstein",
        or a callable function.
    squared : bool, default=False
        Return squared distance.

        .. versionadded:: 0.5

    Returns
    -------
    d : float or ndarray, shape (n_matrices, 1)
        Distance between A and B.

    See Also
    --------
    distance_chol
    distance_euclid
    distance_harmonic
    distance_kullback
    distance_kullback_right
    distance_kullback_sym
    distance_logchol
    distance_logdet
    distance_logeuclid
    distance_riemann
    distance_thompson
    distance_wasserstein

    References
    ----------
    .. [1] `Review of Riemannian distances and divergences, applied to
        SSVEP-based BCI
        <https://hal.archives-ouvertes.fr/LISV/hal-03015762v1>`_
        S. Chevallier, E. K. Kalunga, Q. Barthélemy, E. Monacelli.
        Neuroinformatics, Springer, 2021, 19 (1), pp.93-106
    """
    distance_function = check_function(metric, distance_functions)

    shape_A, shape_B = A.shape, B.shape
    if shape_A == shape_B:
        d = distance_function(A, B, squared=squared)
    elif len(shape_A) == 3 and len(shape_B) == 2:
        # Small adjust for better broadcasting
        d = distance_function(A, B, squared=squared)
        d = d[..., None]
    else:
        raise ValueError("Inputs have incompatible dimensions.")

    return d


###############################################################################


def _euclidean_distances(X, Y=None, squared=False):
    """Function to extend euclidean_distances of sklearn to complex data."""
    xp = get_namespace(X, Y)
    if is_real_type(X):
        dist = pairwise_euclidean(X, X if Y is None else Y, xp=xp)
        return dist ** 2 if squared else dist

    if Y is None:
        Yreal, Yimag = None, None
    else:
        Yreal, Yimag = Y.real, Y.imag

    dist2 = _euclidean_distances(X.real, Yreal, squared=True) + \
        _euclidean_distances(X.imag, Yimag, squared=True)
    return dist2 if squared else xp.sqrt(dist2)


def _pairwise_distance_euclid(X, Y=None, squared=False):
    """Pairwise Euclidean distance matrix.

    Compute the matrix of Euclidean distances between pairs of elements of X
    and Y.

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of matrices. If None, Y is set to X.
    squared : bool, default=False
        Return squared distances.

    Returns
    -------
    dist : ndarray, shape (n_matrices_X, n_matrices_X) or (n_matrices_X, \
            n_matrices_Y)
        Euclidean distances between pairs of elements of X if Y is None, or
        between elements of X and Y.

    See Also
    --------
    pairwise_distance
    distance_euclid
    """
    if Y is not None:
        Y = Y.reshape(len(Y), -1)

    return _euclidean_distances(X.reshape(len(X), -1), Y, squared=squared)


def _pairwise_distance_harmonic(X, Y=None, squared=False):
    """Pairwise harmonic distance matrix.

    Compute the matrix of harmonic distances between pairs of elements of X and
    Y.

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of matrices. If None, Y is set to X.
    squared : bool, default=False
        Return squared distances.

    Returns
    -------
    dist : ndarray, shape (n_matrices_X, n_matrices_X) or (n_matrices_X, \
            n_matrices_Y)
        Harmonic distances between pairs of elements of X if Y is None, or
        between elements of X and Y.

    See Also
    --------
    pairwise_distance
    distance_harmonic
    """
    xp = get_namespace(X, Y)
    eye_n = xp.eye(X.shape[-1], dtype=X.dtype, device=xpd(X))
    if Y is None:
        Y_inv = None
    else:
        Y_inv = xp.linalg.solve(Y, eye_n)

    X_inv = xp.linalg.solve(X, eye_n)
    return _pairwise_distance_euclid(X_inv, Y_inv, squared=squared)


def _pairwise_distance_logchol(X, Y=None, squared=False):
    """Pairwise log-Cholesky distance matrix.

    Compute the matrix of log-Cholesky distances between pairs of elements of
    X and Y.

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of matrices. If None, Y is set to X.
    squared : bool, default=False
        Return squared distances.

    Returns
    -------
    dist : ndarray, shape (n_matrices_X, n_matrices_X) or (n_matrices_X, \
            n_matrices_Y)
        Log-Cholesky distances between pairs of elements of X if Y is None, or
        between elements of X and Y.

    See Also
    --------
    pairwise_distance
    distance_logchol
    """
    xp = get_namespace(X, Y)
    X_chol = xp.linalg.cholesky(X)
    tri0, tri1 = tril_indices(X_chol.shape[-1], -1, xp=xp, like=X_chol)
    diag0, diag1 = diag_indices(X_chol.shape[-1], xp=xp, like=X_chol)

    if Y is None:
        triagular_part = _euclidean_distances(
            X_chol[..., tri0, tri1],
            squared=True
        )
        diagonal_part = _euclidean_distances(
            xp.log(X_chol[..., diag0, diag1]),
            squared=True,
        )
    else:
        Y_chol = xp.linalg.cholesky(Y)
        triagular_part = _euclidean_distances(
            X_chol[..., tri0, tri1],
            Y_chol[..., tri0, tri1],
            squared=True,
        )
        diagonal_part = _euclidean_distances(
            xp.log(X_chol[..., diag0, diag1]),
            xp.log(Y_chol[..., diag0, diag1]),
            squared=True,
        )

    dist = triagular_part + diagonal_part
    return dist if squared else xp.sqrt(dist)


def _pairwise_distance_logeuclid(X, Y=None, squared=False):
    """Pairwise log-Euclidean distance matrix.

    Compute the matrix of log-Euclidean distances between pairs of elements of
    X and Y.

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of matrices. If None, Y is set to X.
    squared : bool, default=False
        Return squared distances.

    Returns
    -------
    dist : ndarray, shape (n_matrices_X, n_matrices_X) or (n_matrices_X, \
            n_matrices_Y)
        Log-Euclidean distances between pairs of elements of X if Y is None, or
        between elements of X and Y.

    See Also
    --------
    pairwise_distance
    distance_logeuclid
    """
    if Y is None:
        logY = None
    else:
        logY = logm(Y)

    return _pairwise_distance_euclid(logm(X), logY, squared=squared)


def _pairwise_distance_riemann(X, Y=None, squared=False):
    """Pairwise Riemannian distance matrix.

    Compute the matrix of Riemannian distances between pairs of elements of X
    and Y.

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of matrices. If None, Y is set to X.
    squared : bool, default=False
        Return squared distances.

    Returns
    -------
    dist : ndarray, shape (n_matrices_X, n_matrices_X) or (n_matrices_X, \
            n_matrices_Y)
        Riemannian distances between pairs of elements of X if Y is None, or
        between elements of X and Y.

    See Also
    --------
    pairwise_distance
    distance_riemann
    """
    XisY = False
    if Y is None:
        XisY = True
        Y = X
    xp = get_namespace(X, Y)

    n_matrices_X, n_matrices_Y = len(X), len(Y)
    Xinv12 = invsqrtm(X)
    dist = xp.zeros((n_matrices_X, n_matrices_Y), dtype=X.real.dtype,
                     device=xpd(X))

    # row by row so it fits in memory
    for i, x_ in enumerate(Xinv12):
        evals_ = xp.linalg.eigvalsh(x_ @ Y[i * XisY:] @ x_)
        d2 = xp.sum(xp.log(evals_) ** 2, axis=-1)
        dist[i, i * XisY:] = d2

    if XisY:
        dist = dist + dist.mT

    return dist if squared else xp.sqrt(dist)


def pairwise_distance(X, Y=None, metric="riemann", squared=False):
    """Pairwise distance matrix.

    Compute the matrix of distances between pairs of elements of X and Y.

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of matrices. If None, Y is set to X.
    metric : string, default="riemann"
        Metric for pairwise distance. For the list of supported metrics,
        see :func:`pyriemann.utils.distance.distance`.
    squared : bool, default=False
        Return squared distances.

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
    xp = get_namespace(X, Y)
    
    if metric == "euclid":
        return _pairwise_distance_euclid(X, Y=Y, squared=squared)
    elif metric == "harmonic":
        return _pairwise_distance_harmonic(X, Y=Y, squared=squared)
    elif metric == "logchol":
        return _pairwise_distance_logchol(X, Y=Y, squared=squared)
    elif metric == "logeuclid":
        return _pairwise_distance_logeuclid(X, Y=Y, squared=squared)
    elif metric == "riemann":
        return _pairwise_distance_riemann(X, Y=Y, squared=squared)

    n_matrices_X, _, _ = X.shape

    # compute full pairwise matrix for non-symmetric metrics
    if Y is None and metric in ["kullback", "kullback_right"]:
        Y = X

    if Y is None:
        dist = xp.zeros((n_matrices_X, n_matrices_X), dtype=X.real.dtype,
                        device=xpd(X))
        for i in range(n_matrices_X):
            for j in range(i + 1, n_matrices_X):
                dist[i, j] = distance(X[i], X[j], metric, squared=squared)
        dist = dist + dist.mT
    else:
        
        n_matrices_Y, _, _ = Y.shape

        dist = xp.empty((n_matrices_X, n_matrices_Y), dtype=X.real.dtype,
                        device=xpd(X))
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
    X : ndarray, shape (..., n, m)
        Vectors.
    cov : ndarray, shape (..., n, n)
        Covariance matrix of the multivariate Gaussian distribution.
    mean : None | ndarray, shape (..., n, 1), default=None
        Mean vector of the multivariate Gaussian distribution.
        If None, distribution is considered as centered.
    squared : bool, default=False
        Return squared distance.

        .. versionadded:: 0.5

    Returns
    -------
    d : ndarray, shape (..., m)
        Mahalanobis distances.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html
    """  # noqa
    xp = get_namespace(X, cov, mean)
    if mean is not None:
        X = X - mean

    Xw = invsqrtm(cov) @ X
    d2 = xp.sum(xp.abs(Xw)**2, axis=-2)
    return d2 if squared else xp.sqrt(d2)
