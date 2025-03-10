"""Optimization on the manifold of positive definite matrices"""

from time import time
import warnings

import numpy as np

from ..utils.base import sqrtm, invsqrtm, logm, ddlogm
from ..datasets import sample_gaussian_spd


def _symmetrize(A):
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


def _retraction(point, tangent_vector, metric):
    """Retracts an array of tangent vector back to the manifold.

    This code is an adaptation from pyManopt [1]_.

    Parameters
    ----------
    point : ndarray, shape (n_matrices, n, n)
        A point on the manifold.
    tangent_vector : array, shape (n_matrices, n, n)
        A tangent vector at the point.

    Returns
    -------
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
    if metric == "euclid":
        retracted_point = point + tangent_vector
    else:
        p_inv_tv = np.linalg.solve(point, tangent_vector)
        retracted_point = _symmetrize(
            point + tangent_vector + tangent_vector @ p_inv_tv / 2
        )

    return retracted_point


def _norm(point, tangent_vector, metric):
    """Compute the norm.

    This code is an adaptation from pyManopt [1]_.

    Parameters
    ----------
    point : ndarray, shape (..., n, n)
        A point on the SPD manifold.
    tangent_vector : ndarray, shape (..., n, n)
        A tangent vector at the point on the SPD manifold.

    Returns
    -------
    norm : float or ndarray, shape (...,)
        The norm of the tangent vector at the given point.

    References
    ----------
    .. [1] `Pymanopt: A Python Toolbox for Optimization on Manifolds using
        Automatic Differentiation
        <http://jmlr.org/papers/v17/16-177.html>`_
        J. Townsend, N. Koep and S. Weichwald, Journal of Machine Learning
        Research, 2016
    """
    if metric == "euclid":
        norm = np.linalg.norm(tangent_vector)
    else:
        p_inv_tv = np.linalg.solve(point, tangent_vector)

        if p_inv_tv.ndim == 2:
            p_inv_tv_transposed = p_inv_tv.T
        else:
            p_inv_tv_transposed = np.transpose(p_inv_tv, (0, 2, 1))
        norm = np.sqrt(
            np.tensordot(
                p_inv_tv, p_inv_tv_transposed, axes=tangent_vector.ndim
            )
        )

    return norm


def _loss(P, Q):
    """Compute the loss of the t-SNE, that is the Kullback-Leibler
    divergence between P and Q.

    Parameters
    ----------
    P : ndarray, shape (n_matrices, n_matrices)
        The matrix of the symmetrized conditional probabilities of X.
    Q : ndarray, shape (n_matrices, n_matrices)
        The matrix of the low dimensional similarities conditional
        probabilities of Y.

    Returns
    -------
    _ : float
        The loss of the t-SNE.
    """
    eye_matrix = np.eye(P.shape[0])
    return np.sum(P * np.log((P + eye_matrix) / (Q + eye_matrix)))


def _riemannian_gradient(Y, P, Q, Dsq, n_components, metric):
    """Compute the Riemannian gradient of the loss of the t-SNE.

    Parameters
    ----------
    Y : ndarray, shape (n_matrices, n_components, n_components)
        Set of low-dimensional SPD matrices.
    P : ndarray, shape (n_matrices, n_matrices)
        Symmetrized conditional probabilities of X.
    Q : ndarray, shape (n_matrices, n_matrices)
        Low dimensional similarities conditional probabilities of Y.
    Dsq : ndarray, shape (n_matrices, n_matrices)
        The Riemannian distance matrix of Y.
    n_components : int
        Dimension of the matrices in the embedded space.
    metric : {"euclid", "logeuclid", "riemann"}
        Metric for the gradient descent.

    Returns
    -------
    grad : ndarray, shape (n_matrices, n_components, n_components)
        The Riemannian gradient of the loss of the t-SNE.
    """
    n_matrices, _ = P.shape
    Y_invsqrt = invsqrtm(Y)
    Y_sqrt = sqrtm(Y)

    grad = np.zeros((n_matrices, n_components, n_components))
    for i in range(n_matrices):
        if metric == "riemann":
            grad_dist = - (
                Y_sqrt[i] @ logm(Y_invsqrt[i] @ Y @ Y_invsqrt[i]) @ Y_sqrt[i]
            )
        elif metric == "logeuclid":
            grad_dist = Y[i] @ ddlogm(logm(Y[i]) - logm(Y), Y[i]) @ Y[i]
        elif metric == "euclid":
            grad_dist = (Y[i] - Y)

        grad[i] = 4 * np.sum(
            ((P[i] - Q[i]) / (1 + Dsq[i]))[:, np.newaxis, np.newaxis]
            * grad_dist,
            axis=0,
        )
    return grad


def _get_initial_solution(n_matrices, n_components, random_state):
    """Generate an initial solution.

    Generate an initial solution for the t-SNE optimization by sampling
    Gaussian SPD matrices.

    Parameters
    ----------
    n_matrices : int
        Number of SPD matrices to generate.
    n_components : int
        Dimension of SPD matrices to generate.
    random_state : int, RandomState instance or None
        The seed or random number generator for reproducibility.

    Returns
    -------
    initial_sol : ndarray, shape (n_matrices, n_components, n_components)
        Generated SPD matrices.
    """

    initial_sol = sample_gaussian_spd(
        n_matrices,
        mean=np.eye(n_components),
        sigma=1,
        random_state=random_state,
    )
    return initial_sol


def _run_minimization(
    P,
    initial_sol,
    metric,
    max_iter,
    max_time,
    verbosity,
    compute_low_affinities,
):
    """Run the minimization to solve the t-SNE optimization.

    Parameters
    ----------
    P : ndarray, shape (n_matrices, n_matrices)
        Symmetrized conditional probabilities of high-dimensional matrices.
    initial_sol : ndarray, shape (n_matrices, n_components, n_components)
        Initial point for the optimization.
    metric : {"euclid", "logeuclid", "riemann"}
        Metric for the gradient descent.
    max_iter : int
        Maximum number of iterations for the optimization.
    max_time : float
        Maximum time (in seconds) allowed for the optimization.
    verbosity : int
        Level of verbosity. Higher values result in more detailed output.
    compute_low_affinities : callable
        Function to compute low affinities.

    Returns
    -------
    current_sol : ndarray, shape (n_matrices, n_components, n_components)
        The solution of the t-SNE problem.
    """
    tol_step = 1e-6
    current_sol = initial_sol
    loss_evolution = []
    initial_time = time()
    _, n_components, _ = initial_sol.shape

    for i in range(max_iter):
        if verbosity >= 2 and i % 100 == 0:
            print("Iteration : ", i)

        # get the current value for the loss function
        Q, Dsq = compute_low_affinities(current_sol)
        loss = _loss(P, Q)
        loss_evolution.append(loss)

        # get the direction of steepest descent
        direction = _riemannian_gradient(
            current_sol, P, Q, Dsq, n_components, metric
        )
        norm_direction = _norm(current_sol, direction, metric)

        # backtracking line search
        if i == 0:
            alpha = 1.0 / norm_direction
        else:
            # Pick initial step size based on where we were last time and
            # look a bit further
            # See Boumal, 2023, Section 4.3 for more insights.
            alpha = 4 * (loss_evolution[-2] - loss) / norm_direction**2

        tau = 0.50
        r = 1e-4
        maxiter_linesearch = 25

        retracted = _retraction(current_sol, -alpha * direction, metric)
        Q_retract, Dsq_retract = compute_low_affinities(retracted)
        loss_retracted = _loss(P, Q_retract)

        # Backtrack while the Armijo criterion is not satisfied
        for _ in range(maxiter_linesearch):
            if loss - loss_retracted > r * alpha * norm_direction**2:
                break
            alpha = tau * alpha

            retracted = _retraction(current_sol, -alpha * direction, metric)
            Q_retract, Dsq_retract = compute_low_affinities(retracted)
            loss_retracted = _loss(P, Q_retract)
        else:
            warnings.warn("Maximum iteration in linesearched reached.")

        # update variable for next iteration
        current_sol = retracted

        # test if the step size is small
        crit = _norm(current_sol, -alpha * direction, metric)
        if crit <= tol_step:
            if verbosity >= 1:
                print("Min stepsize reached")
            break

        # test if the maximum time has been reached
        if time() - initial_time >= max_time:
            warnings.warn(f"Time limit reached after {i} iterations.")
            break

    else:
        warnings.warn("Convergence not reached.")

    if verbosity >= 1:
        print("Optimization done in {:.2f} seconds.".format(
            time() - initial_time))
    return current_sol


def _get_tsne_embedding(
    P,
    n_components,
    metric,
    max_iter,
    max_time,
    verbosity,
    random_state,
    compute_low_affinities,
):
    """
    Compute the t-SNE embedding.

    Parameters
    ----------
    P : ndarray, shape (n_matrices, n_matrices)
        Symmetrized conditional probabilities of high-dimensional matrices.
    initial_point : ndarray, shape (n_matrices, n_components, n_components)
        Initial point for the optimization.
    metric : {"euclid", "logeuclid", "riemann"}
        Metric for the gradient descent.
    max_iter : int
        Maximum number of iterations for the optimization.
    max_time : float
        Maximum time (in seconds) allowed for the optimization.
    verbosity : int
        Level of verbosity. Higher values result in more detailed output.
    random_state : int, RandomState instance or None
        The seed or random number generator for reproducibility.
    compute_low_affinities : callable
        Function to compute low affinities.

    Returns
    -------
    embedding : ndarray, shape (n_matrices, n_components)
        The computed t-SNE embedding.
    """

    n_matrices, _ = P.shape

    # Sample initial solution close to the identity
    embedding_ini = _get_initial_solution(
        n_matrices,
        n_components,
        random_state,
    )

    embedding = _run_minimization(
        P,
        embedding_ini,
        metric,
        max_iter,
        max_time,
        verbosity,
        compute_low_affinities,
    )

    return embedding
