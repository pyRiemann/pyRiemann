"""Geodesics for SPD matrices."""

from .base import sqrtm, invsqrtm, powm, logm, expm


def geodesic_euclid(A, B, alpha=0.5):
    r"""Euclidean geodesic between SPD matrices.

    The matrix at the position alpha on the Euclidean geodesic between two SPD
    matrices A and B is:

    .. math::
        \mathbf{C} = (1-\alpha) \mathbf{A} + \alpha \mathbf{B}

    C is equal to A if alpha = 0 and B if alpha = 1.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD matrices.
    B : ndarray, shape (..., n, n)
        Second SPD matrices.
    alpha : float, default=0.5
        The position on the geodesic.

    Returns
    -------
    C : ndarray, shape (..., n, n)
        SPD matrices on the Euclidean geodesic.
    """
    return (1 - alpha) * A + alpha * B


def geodesic_logeuclid(A, B, alpha=0.5):
    r"""Log-Euclidean geodesic between SPD matrices.

    The matrix at the position alpha on the Log-Euclidean geodesic between two
    SPD matrices A and B is:

    .. math::
        \mathbf{C} = \exp \left( (1-\alpha) \log(\mathbf{A})
                     + \alpha \log(\mathbf{B}) \right)

    C is equal to A if alpha = 0 and B if alpha = 1.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD matrices.
    B : ndarray, shape (..., n, n)
        Second SPD matrices.
    alpha : float, default=0.5
        The position on the geodesic.

    Returns
    -------
    C : ndarray, shape (..., n, n)
        SPD matrices on the Log-Euclidean geodesic.
    """
    return expm((1 - alpha) * logm(A) + alpha * logm(B))


def geodesic_riemann(A, B, alpha=0.5):
    r"""Affine-invariant Riemannian geodesic between SPD matrices.

    The matrix at the position alpha on the affine-invariant Riemannian
    geodesic between two SPD matrices A and B is:

    .. math::
        \mathbf{C} = \mathbf{A}^{1/2} \left( \mathbf{A}^{-1/2} \mathbf{B}
                     \mathbf{A}^{-1/2} \right)^\alpha \mathbf{A}^{1/2}

    C is equal to A if alpha = 0 and B if alpha = 1.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD matrices.
    B : ndarray, shape (..., n, n)
        Second SPD matrices.
    alpha : float, default=0.5
        The position on the geodesic.

    Returns
    -------
    C : ndarray, shape (..., n, n)
        SPD matrices on the affine-invariant Riemannian geodesic.
    """
    sA, isA = sqrtm(A), invsqrtm(A)
    C = isA @ B @ isA
    D = powm(C, alpha)
    E = sA @ D @ sA
    return E


###############################################################################


def geodesic(A, B, alpha, metric='riemann'):
    """Geodesic between SPD matrices according to a metric.

    Return the matrix at the position alpha on the geodesic between SPD
    matrices A and B according to a metric.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD matrices.
    B : ndarray, shape (..., n, n)
        Second SPD matrices.
    alpha : float
        The position on the geodesic.
    metric : string, default='riemann'
        The metric used for geodesic, can be: 'euclid', 'logeuclid', 'riemann'.

    Returns
    -------
    C : ndarray, shape (..., n, n)
        SPD matrices on the geodesic.
    """
    options = {
        'euclid': geodesic_euclid,
        'logeuclid': geodesic_logeuclid,
        'riemann': geodesic_riemann,
    }
    C = options[metric](A, B, alpha)
    return C
