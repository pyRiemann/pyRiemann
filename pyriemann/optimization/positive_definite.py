"""Optimization on the manifold of positive definite matrices"""

from time import time
import warnings

import numpy as np

from ..utils.base import sqrtm, invsqrtm, logm


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


def _retraction(point, tangent_vector):
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

    p_inv_tv = np.linalg.solve(point, tangent_vector)
    return _symmetrize(point + tangent_vector + tangent_vector @ p_inv_tv / 2)


def _norm(point, tangent_vector):
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

    p_inv_tv = np.linalg.solve(point, tangent_vector)

    if p_inv_tv.ndim == 2:
        p_inv_tv_transposed = p_inv_tv.T
    else:
        p_inv_tv_transposed = np.transpose(p_inv_tv, (0, 2, 1))

    return np.sqrt(
        np.tensordot(p_inv_tv, p_inv_tv_transposed, axes=tangent_vector.ndim)
    )


def _d_log_array(A, H):
    """
    Compute the directional derivative of the matrix logarithm.

    Parameters:
    A (np.ndarray): Positive definite matrix.
    H (np.ndarray): Symmetric matrix.

    Returns:
    np.ndarray: The directional derivative Dlog(A)[H].
    """
    # Diagonalize A
    eigenvalues, U = np.linalg.eigh(A)

    # Compute H_tilde
    H_tilde = U.T @ H @ U

    # Compute Z matrix using broadcasting for efficiency
    eigen_diff = eigenvalues[:, np.newaxis] - eigenvalues[np.newaxis, :]
    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # Handle division by zero safely
        Z = np.where(
            eigen_diff != 0,
            (np.log(eigenvalues[:, np.newaxis])
             - np.log(eigenvalues[np.newaxis, :]))
            / eigen_diff,
            1 / eigenvalues[:, np.newaxis],
        )
    # Compute the result
    result = (
        U @ (H_tilde * Z[np.newaxis, np.newaxis, :]) @ U.T
    )
    return result[0]


def _cost(P, Q):
    """Computed the loss of the t-SNE, that is the Kullback-Leibler
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
        The cost of the t-SNE.
    """
    eye_matrix = np.eye(P.shape[0])
    return np.sum(P * np.log((P + eye_matrix) / (Q + eye_matrix)))


def _riemannian_gradient(Y, P, Q, Dsq, n_components, metric):
    """Computed the Riemannian gradient of the loss of the t-SNE.

    Parameters
    ----------
    Y : ndarray, shape (n_matrices, n_components, n_components)
        Set of SPD matrices.
    P : ndarray, shape (n_matrices, n_matrices)
        The matrix of the symmetrized conditional probabilities of X.
    Q : ndarray, shape (n_matrices, n_matrices)
        The matrix of the low dimensional similarities conditional
        probabilities of Y.
    Dsq : ndarray, shape (n_matrices, n_matrices)
        The Riemannian distance matrix of Y.
    n_components : int
        Dimension of the matrices in the embedded space.
    metric : {"logeuclid", "riemann"}
        Metric for the gradient descent.

    Returns
    -------
    grad : ndarray, shape (n_matrices, n_components, n_components)
        The Riemannian gradient of the cost of the t-SNE.
    """
    n_matrices, _ = P.shape
    grad = np.zeros((n_matrices, n_components, n_components))
    Y_i_invsqrt = invsqrtm(Y)
    Y_i_sqrt = sqrtm(Y)
    for i in range(n_matrices):
        if metric == "riemann":
            grad_dist = (
                Y_i_sqrt[i] @ logm(
                    Y_i_invsqrt[i] @ Y @ Y_i_invsqrt[i]
                ) @ Y_i_sqrt[i]
            )
        elif metric == "logeuclid":
            grad_dist = -Y[i]@_d_log_array(Y[i], logm(Y[i]) - logm(Y))@Y[i]

        grad[i] = -4 * np.sum(
            ((P[i] - Q[i]) / (1 + Dsq[i]))[:, np.newaxis, np.newaxis]
            * grad_dist,
            axis=0,
        )
    return grad


def _run_minimization(
    P,
    initial_point,
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
        The matrix of the symmetrized conditional probabilities of X.
    initial_point : ndarray, shape (n_matrices, n_components, n_components)
        The initial point for the optimization.
    metric : {"logeuclid", "riemann"}
        Metric for the gradient descent.
    max_iter : int
        The maximum number of iterations for the optimization.
    max_time : float
        The maximum time (in seconds) allowed for the optimization.
    verbosity : int
        The level of verbosity. Higher values result in more detailed output.
    compute_low_affinities : callable
        Function to compute low affinities.

    Returns
    -------
    current_sol : ndarray, shape (n_matrices, n_components, n_components)
        The solution of the t-SNE problem.
    """
    tol_step = 1e-6
    current_sol = initial_point
    loss_evolution = []
    initial_time = time()
    n_components = initial_point.shape[1]

    for i in range(max_iter):
        if verbosity >= 2 and i % 100 == 0:
            print("Iteration : ", i)

        # get the current value for the loss function
        Q, Dsq = compute_low_affinities(current_sol)
        loss = _cost(P, Q)
        loss_evolution.append(loss)

        # get the direction of steepest descent
        direction = _riemannian_gradient(
            current_sol, P, Q, Dsq, n_components, metric
        )
        norm_direction = _norm(current_sol, direction)

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

        retracted = _retraction(current_sol, -alpha * direction)
        Q_retract, Dsq_retract = compute_low_affinities(retracted)
        loss_retracted = _cost(P, Q_retract)

        # Backtrack while the Armijo criterion is not satisfied
        for _ in range(maxiter_linesearch):
            if loss - loss_retracted > r * alpha * norm_direction**2:
                break
            alpha = tau * alpha

            retracted = _retraction(current_sol, -alpha * direction)
            Q_retract, Dsq_retract = compute_low_affinities(retracted)
            loss_retracted = _cost(P, Q_retract)
        else:
            warnings.warn("Maximum iteration in linesearched reached.")

        # update variable for next iteration
        current_sol = retracted

        # test if the step size is small
        crit = _norm(current_sol, -alpha * direction)
        if crit <= tol_step:
            if verbosity >= 1:
                print("Min stepsize reached")
            break

        # test if the maximum time has been reached
        if time() - initial_time >= max_time:
            warnings.warn(
                "Time limite reached after " + str(i) + " iterations."
            )
            break

    else:
        warnings.warn("Convergence not reached.")

    if verbosity >= 1:
        print("Optimization done in {:.2f} seconds.".format(
            time() - initial_time))
    return current_sol, loss_evolution
