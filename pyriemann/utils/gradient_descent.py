"""Tools used for Riemannian Gradient Descent"""
import numpy as np


def symmetrize(A):
    """Symmetrize an array.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        Input array to be symmetrized.

    Returns
    -------
    ndarray, shape (..., n, n)
        Symmetrized array.
    """

    return (A + np.swapaxes(A, -1, -2)) / 2


def retraction(point, tangent_vector):
    """Retracts an array of tangent vector back to the manifold.

    This code is an adaptation from pyManopt [1]_.

    Parameters
    ----------
    point : ndarray, shape (n_matrices, n, n)
        A point on the manifold.
    tangent_vector : array, shape (n_matrices, n, n)
        A tangent vector at the point.

    Returns
    retracted_point : array, shape (n_matrices, n, n)
        The tangent vector retracted on the manifold.


    References
    ----------
    .. [1] `Pymanopt: A Python Toolbox for Optimization on Manifolds using
        Automatic Differentiation
        <http://jmlr.org/papers/v17/16-177.html>`_
        J. Townsend, N. Koep and S. Weichwald, Journal of Machine Learning
        Research, 2016
    """

    p_inv_tv = np.linalg.solve(point, tangent_vector)
    return symmetrize(point + tangent_vector + tangent_vector @ p_inv_tv / 2)


def norm(point, tangent_vector):
    """Compute the norm.
    This code is an adaptation from pyManopt [1]_.

    Parameters
    point : ndarray, shape (..., n, n)
        A point on the SPD manifold.
    tangent_vector : ndarray, shape (..., n, n)
        A tangent vector at the point on the SPD manifold.
    Returns
    -------
    norm : float
        The norm of the tangent vector at the given point.


    .. [1] `Pymanopt: A Python Toolbox for Optimization on Manifolds using
        Automatic Differentiation
        <http://jmlr.org/papers/v17/16-177.html>`_
        J. Townsend, N. Koep and S. Weichwald, Journal of Machine Learning
        Research, 2016
    """

    p_inv_tv = np.linalg.solve(point, tangent_vector)

    if p_inv_tv.ndim == 2:
        p_inv_tv_transposed = p_inv_tv.T
    else:
        p_inv_tv_transposed = np.transpose(p_inv_tv, (0, 2, 1))

    return np.sqrt(
        np.tensordot(p_inv_tv, p_inv_tv_transposed, axes=tangent_vector.ndim)
    )
