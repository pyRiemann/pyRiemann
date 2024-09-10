"""Geodesics for SPD/HPD matrices."""

from .base import sqrtm, invsqrtm, powm, logm, expm
from .utils import check_function
import numpy as np
from numpy.linalg import cholesky


def geodesic_euclid(A, B, alpha=0.5):
    r"""Euclidean geodesic between matrices.

    The matrix at position :math:`\alpha` on the Euclidean geodesic
    between two matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` is:

    .. math::
        \mathbf{C} = (1-\alpha) \mathbf{A} + \alpha \mathbf{B}

    :math:`\mathbf{C}` is equal to :math:`\mathbf{A}` if :math:`\alpha` = 0,
    and :math:`\mathbf{B}` if :math:`\alpha` = 1.

    Parameters
    ----------
    A : ndarray, shape (..., n, m)
        First matrices.
    B : ndarray, shape (..., n, m)
        Second matrices.
    alpha : float, default=0.5
        Position on the geodesic.

    Returns
    -------
    C : ndarray, shape (..., n, m)
        Matrices on the Euclidean geodesic.
    """
    return (1 - alpha) * A + alpha * B


def geodesic_logcholesky(A, B, alpha=0.5):
    r"""Log-Cholesky geodesic between SPD/HPD matrices.

    The matrix at position :math:`\alpha` on the Log-Cholesky geodesic
    between two SPD/HPD matrices :math:`\mathbf{A}` and :math:`\mathbf{B}`
    given in [1]_.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices.
    alpha : float, default=0.5
        Position on the geodesic.

    Returns
    -------
    C : ndarray, shape (..., n, n)
        SPD/HPD matrices on the Log-Cholesky geodesic.

    Notes
    -----
    ..versionadded:: 0.7

    References
    ----------
    .. [1] `Riemannian geometry of symmetric positive definite matrices via
        Cholesky decomposition
        <https://arxiv.org/pdf/1908.09326>`_
        Z. Lin. SIAM J Matrix Anal Appl, 2019, 40(4), pp. 1353-1370.
    """
    A_chol = cholesky(A)
    B_chol = cholesky(B)
    geo_t = np.zeros_like(A)

    tr0, tr1 = np.tril_indices(A_chol.shape[-1], -1)
    geo_t[..., tr0, tr1] = (1 - alpha) * A_chol[..., tr0, tr1] + \
        alpha * B_chol[..., tr0, tr1]

    diag0, diag1 = np.diag_indices(A_chol.shape[-1])
    geo_t[..., diag0, diag1] = A_chol[..., diag0, diag1] ** (1 - alpha) * \
        B_chol[..., diag0, diag1] ** alpha

    return geo_t @ geo_t.conj().swapaxes(-1, -2)


def geodesic_logeuclid(A, B, alpha=0.5):
    r"""Log-Euclidean geodesic between SPD/HPD matrices.

    The matrix at position :math:`\alpha` on the Log-Euclidean geodesic
    between two SPD/HPD matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` is:

    .. math::
        \mathbf{C} = \exp \left( (1-\alpha) \log(\mathbf{A})
                     + \alpha \log(\mathbf{B}) \right)

    :math:`\mathbf{C}` is equal to :math:`\mathbf{A}` if :math:`\alpha` = 0,
    and :math:`\mathbf{B}` if :math:`\alpha` = 1.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices.
    alpha : float, default=0.5
        Position on the geodesic.

    Returns
    -------
    C : ndarray, shape (..., n, n)
        SPD/HPD matrices on the Log-Euclidean geodesic.
    """
    return expm((1 - alpha) * logm(A) + alpha * logm(B))


def geodesic_riemann(A, B, alpha=0.5):
    r"""Affine-invariant Riemannian geodesic between SPD/HPD matrices.

    The matrix at position :math:`\alpha` on the affine-invariant Riemannian
    geodesic between two SPD/HPD matrices :math:`\mathbf{A}` and
    :math:`\mathbf{B}` is:

    .. math::
        \mathbf{C} = \mathbf{A}^{1/2} \left( \mathbf{A}^{-1/2} \mathbf{B}
                     \mathbf{A}^{-1/2} \right)^\alpha \mathbf{A}^{1/2}

    :math:`\mathbf{C}` is equal to :math:`\mathbf{A}` if :math:`\alpha` = 0,
    and :math:`\mathbf{B}` if :math:`\alpha` = 1.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices.
    alpha : float, default=0.5
        Position on the geodesic.

    Returns
    -------
    C : ndarray, shape (..., n, n)
        SPD/HPD matrices on the affine-invariant Riemannian geodesic.
    """
    sA, isA = sqrtm(A), invsqrtm(A)
    C = isA @ B @ isA
    D = powm(C, alpha)
    E = sA @ D @ sA
    return E


###############################################################################


geodesic_functions = {
    "euclid": geodesic_euclid,
    "logcholesky": geodesic_logcholesky,
    "logeuclid": geodesic_logeuclid,
    "riemann": geodesic_riemann,
}


def geodesic(A, B, alpha, metric="riemann"):
    """Geodesic between matrices according to a metric.

    Return the matrix at the position alpha on the geodesic between matrices
    A and B according to a metric.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First matrices.
    B : ndarray, shape (..., n, n)
        Second matrices.
    alpha : float
        Position on the geodesic.
    metric : string | callable, default="riemann"
        Metric used for geodesic, can be: "euclid", "logeuclid", "riemann",
        or a callable function.

    Returns
    -------
    C : ndarray, shape (..., n, n)
        Matrices on the geodesic.
    """
    geodesic_function = check_function(metric, geodesic_functions)
    C = geodesic_function(A, B, alpha)
    return C
