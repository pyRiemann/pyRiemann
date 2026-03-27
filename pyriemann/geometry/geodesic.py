"""Geodesics for SPD/HPD matrices."""
import numpy as np
from scipy.linalg import eigvalsh

from .base import _recursive, ctranspose, sqrtm, invsqrtm, powm, logm, expm
from .utils import check_function


def geodesic_chol(A, B, alpha=0.5):
    r"""Cholesky geodesic between SPD/HPD matrices.

    The matrix at position :math:`\alpha` on the Cholesky geodesic
    between two SPD/HPD matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` is
    :math:`\mathbf{C} = \mathbf{L} \mathbf{L}^H`,
    where :math:`\mathbf{L}` is computed as [1]_:

    .. math::
        \mathbf{L} = (1-\alpha) \text{chol}(\mathbf{A}) +
                     \alpha \text{chol}(\mathbf{B})

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
        SPD/HPD matrices on the Cholesky geodesic.

    Notes
    -----
    ..versionadded:: 0.10

    See Also
    --------
    geodesic

    References
    ----------
    .. [1] `Non-Euclidean statistics for covariance matrices, with applications
        to diffusion tensor imaging
        <https://doi.org/10.1214/09-AOAS249>`_
        I.L. Dryden, A. Koloydenko, D. Zhou.
        Ann Appl Stat, 2009, 3(3), pp. 1102-1123.
    """
    geo = (1 - alpha) * np.linalg.cholesky(A) + alpha * np.linalg.cholesky(B)
    return geo @ ctranspose(geo)


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

    See Also
    --------
    geodesic
    """
    return (1 - alpha) * A + alpha * B


def geodesic_logchol(A, B, alpha=0.5):
    r"""Log-Cholesky geodesic between SPD/HPD matrices.

    The matrix at position :math:`\alpha` on the log-Cholesky geodesic
    between two SPD/HPD matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` is:

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

    See Also
    --------
    geodesic

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

    return geo @ ctranspose(geo)


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

    See Also
    --------
    geodesic

    References
    ----------
    .. [1] `Geometric means in a novel vector space structure on symmetric
        positive-definite matrices
        <https://epubs.siam.org/doi/abs/10.1137/050637996>`_
        V. Arsigny, P. Fillard, X. Pennec, N. Ayache.
        SIAM J Matrix Anal Appl, 2007, 29 (1), pp. 328-347
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

    See Also
    --------
    geodesic

    References
    ----------
    .. [1] `Riemannian geometry and matrix geometric means
        <https://www.sciencedirect.com/science/article/pii/S0024379505004350>`_
        R. Bhatia and J. Holbrook.
        Linear Algebra and its Applications, 2006
    """
    sA, isA = sqrtm(A), invsqrtm(A)
    C = sA @ powm(isA @ B @ isA, alpha) @ sA
    return C


def geodesic_thompson(A, B, alpha=0.5):
    r"""Thompson geodesic between SPD/HPD matrices.

    The matrix at position :math:`\alpha` on a possible Thompson geodesic
    between two SPD/HPD matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` is
    given in [1]_.

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
        SPD/HPD matrices on the Thompson geodesic.

    Notes
    -----
    ..versionadded:: 0.10

    See Also
    --------
    geodesic

    References
    ----------
    .. [1] `Differential geometry with extreme eigenvalues in the positive
        semidefinite cone
        <https://arxiv.org/pdf/2304.07347>`_
        C. Mostajeran, N. Da Costa, G. Van Goffrier and R. Sepulchre.
        SIAM Journal on Matrix Analysis and Applications, 2024
    """
    A_ndim = A.ndim
    while A.ndim < 3:
        A, B = A[np.newaxis, ...], B[np.newaxis, ...]
    E = _recursive(eigvalsh, B, A)
    Emin, Emax = E.min(axis=-1), E.max(axis=-1)

    C = np.zeros_like(A)

    mask = (Emin == Emax)
    C[mask] = (Emin[mask][..., np.newaxis, np.newaxis] ** alpha) * A[mask]

    Emin_a, Emax_a = Emin ** alpha, Emax ** alpha
    b = (Emax_a - Emin_a)[~mask][..., np.newaxis, np.newaxis]
    a = (Emax * Emin_a - Emin * Emax_a)[~mask][..., np.newaxis, np.newaxis]
    c = b * B[~mask] + a * A[~mask]
    C[~mask] = c / (Emax - Emin)[~mask][..., np.newaxis, np.newaxis]

    if A_ndim < 3:
        C = C[0]

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

    See Also
    --------
    geodesic

    References
    ----------
    .. [1] `Wasserstein Riemannian geometry of Gaussian densities
        <https://link.springer.com/article/10.1007/s41884-018-0014-4>`_
        L. Malagò, L. Montrucchio, G. Pistone.
        Information Geometry, 2018, 1, pp. 137–179.
    """
    A12 = sqrtm(A)
    A12inv = invsqrtm(A)
    AB12 = A12 @ sqrtm(A12 @ B @ A12) @ A12inv
    return (1-alpha)**2 * A + alpha**2 * B + \
        alpha*(1-alpha) * (AB12 + ctranspose(AB12))


###############################################################################


geodesic_functions = {
    "chol": geodesic_chol,
    "euclid": geodesic_euclid,
    "logchol": geodesic_logchol,
    "logeuclid": geodesic_logeuclid,
    "riemann": geodesic_riemann,
    "thompson": geodesic_thompson,
    "wasserstein": geodesic_wasserstein,
}


def geodesic(A, B, alpha, metric="riemann"):
    r"""Geodesic between matrices according to a metric.

    Return the matrix at the position alpha on the geodesic between matrices
    A and B according to a metric.

    :math:`\mathbf{C}` is equal to :math:`\mathbf{A}` if :math:`\alpha` = 0,
    and :math:`\mathbf{B}` if :math:`\alpha` = 1.

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
        "chol", "euclid", "logchol", "logeuclid", "riemann", "thompson",
        "wasserstein",
        or a callable function.

    Returns
    -------
    C : ndarray, shape (..., n, n)
        Matrices on the geodesic.

    See Also
    --------
    geodesic_chol
    geodesic_euclid
    geodesic_logchol
    geodesic_logeuclid
    geodesic_riemann
    geodesic_thompson
    geodesic_wasserstein
    """
    geodesic_function = check_function(metric, geodesic_functions)
    C = geodesic_function(A, B, alpha)
    return C
