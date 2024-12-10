"""Geodesics for SPD/HPD matrices."""
import numpy as np

from .base import sqrtm, invsqrtm, powm, logm, expm
from .utils import check_function


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


def geodesic_logchol(A, B, alpha=0.5):
    r"""Log-Cholesky geodesic between SPD/HPD matrices.

    The matrix at position :math:`\alpha` on the log-Cholesky geodesic
    between two SPD/HPD matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` is
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
        SPD/HPD matrices on the log-Cholesky geodesic.

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
    A_chol, B_chol = np.linalg.cholesky(A), np.linalg.cholesky(B)

    geo = np.zeros_like(A)

    tri0, tri1 = np.tril_indices(A_chol.shape[-1], -1)
    geo[..., tri0, tri1] = (1 - alpha) * A_chol[..., tri0, tri1] + \
        alpha * B_chol[..., tri0, tri1]

    diag0, diag1 = np.diag_indices(A_chol.shape[-1])
    geo[..., diag0, diag1] = A_chol[..., diag0, diag1] ** (1 - alpha) * \
        B_chol[..., diag0, diag1] ** alpha

    return geo @ geo.conj().swapaxes(-1, -2)


def geodesic_logeuclid(A, B, alpha=0.5):
    r"""Log-Euclidean geodesic between SPD/HPD matrices.

    The matrix at position :math:`\alpha` on the log-Euclidean geodesic
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
        SPD/HPD matrices on the log-Euclidean geodesic.
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
    C = sA @ powm(isA @ B @ isA, alpha) @ sA
    return C


def geodesic_wasserstein(A, B, alpha=0.5):
    r"""Wasserstein geodesic between SPD/HPD matrices.

    The matrix at position :math:`\alpha` on the Wasserstein geodesic between
    two SPD/HPD matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` is
    given in [1]_:

    .. math::
        \mathbf{C} = (1-\alpha)^2\mathbf{A} + \alpha^2\mathbf{B} +
          \alpha(1-\alpha)((\mathbf{AB})^{1/2} + (\mathbf{BA})^{1/2})

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
        SPD/HPD matrices on the Wasserstein geodesic.

    Notes
    -----
    ..versionadded:: 0.8

    References
    ----------
    .. [1] `Wasserstein Riemannian geometry of Gaussian densities
        <https://link.springer.com/article/10.1007/s41884-018-0014-4>`_
        L. Malagò, L. Montrucchio, G. Pistone. Information Geometry, 2018, 1,
        pp. 137–179.
    """
    A12 = sqrtm(A)
    A12inv = invsqrtm(A)
    AB12 = A12 @ sqrtm(A12 @ B @ A12) @ A12inv
    return (1-alpha)**2 * A + alpha**2 * B + \
        alpha*(1-alpha) * (AB12 + AB12.conj().swapaxes(-1, -2))


###############################################################################


geodesic_functions = {
    "euclid": geodesic_euclid,
    "logchol": geodesic_logchol,
    "logeuclid": geodesic_logeuclid,
    "riemann": geodesic_riemann,
    "wasserstein": geodesic_wasserstein,
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
        Metric used for geodesic, can be:
        "euclid", "logchol", "logeuclid", "riemann", "wasserstein",
        or a callable function.

    Returns
    -------
    C : ndarray, shape (..., n, n)
        Matrices on the geodesic.
    """
    geodesic_function = check_function(metric, geodesic_functions)
    C = geodesic_function(A, B, alpha)
    return C
