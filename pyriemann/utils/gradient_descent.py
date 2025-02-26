"""Tools used for Riemannian Gradient Descent"""
import numpy as np

from .base import multisym, multitransp


def retraction(point, tangent_vector):
    """
    Retracts a tangent vector back to the manifold.

    Parameters
    ----------
    point : array, shape (n, n)
        A point on the manifold.
    tangent_vector : array, shape (n, n)
        A tangent vector at the point.
    Returns
    -------
    retracted_point : array, shape (n, n)
        The tangent vector retracted on the manifold.

    This code is an adaptation from pyManopt [1]_.

    References
    ----------
    .. [1] `Pymanopt: A Python Toolbox for Optimization on Manifolds using
        Automatic Differentiation
        <http://jmlr.org/papers/v17/16-177.html>`_
        J. Townsend, N. Koep and S. Weichwald, Journal of Machine Learning
        Research, 2016
    """

    p_inv_tv = np.linalg.solve(point, tangent_vector)
    return multisym(point + tangent_vector + tangent_vector @ p_inv_tv / 2)


def norm_SPD(point, tangent_vector):
    """
    Compute the norm of a tangent vector at a point on the manifold of
    SPD matrices.
    Parameters
    ----------
    point : ndarray, shape (..., n, n)
        A point on the SPD manifold.
    tangent_vector : ndarray, shape (..., n, n)
        A tangent vector at the point on the SPD manifold.
    Returns
    -------
    norm : float
        The norm of the tangent vector at the given point.

    This code is an adaptation from pyManopt [1]_.

    References
    ----------
    .. [1] `Pymanopt: A Python Toolbox for Optimization on Manifolds using
        Automatic Differentiation
        <http://jmlr.org/papers/v17/16-177.html>`_
        J. Townsend, N. Koep and S. Weichwald, Journal of Machine Learning
        Research, 2016
    """

    p_inv_tv = np.linalg.solve(point, tangent_vector)
    return np.sqrt(
        np.tensordot(p_inv_tv, multitransp(p_inv_tv), axes=tangent_vector.ndim)
    )
