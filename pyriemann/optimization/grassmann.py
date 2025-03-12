"""Optimization for rotations on Grassmann manifold"""

import warnings

import numpy as np

from ..utils.distance import distance
from ..utils.utils import check_weights


def _project(X, U):
    """Projection.

    Project U on the tangent space of orthogonal matrices with base point X.
    """

    return X @ ((X.T @ U) - (X.T @ U).T) / 2


def _retract(X, v):
    """Retraction.

    Retraction taking tangent vector v at base point X back to manifold of
    orthogonal matrices.
    """

    Q, R = np.linalg.qr(X + v)
    for i, ri in enumerate(np.diag(R)):
        if ri < 0:
            Q[:, i] = -1 * Q[:, i]
    return Q


def _loss(Q, X, Y, weights, metric="euclid"):
    """Loss function for estimating the rotation matrix."""

    return weights @ distance(X, Q @ Y @ Q.T, metric=metric)


def _grad(Q, X, Y, weights, metric="euclid"):
    """Gradient of loss function for estimating the rotation matrix."""

    if metric == "euclid":
        return np.einsum("a,abc->bc", weights, -4 * (X - Q @ Y @ Q.T) @ Q @ Y)

    elif metric == "riemann":
        M = np.linalg.inv(X) @ Q @ Y @ Q.T
        eigvals, eigvecs = np.linalg.eig(M)
        logeigvals = np.expand_dims(np.log(eigvals), -2)
        logM = (eigvecs * logeigvals) @ np.linalg.inv(eigvecs)
        return np.einsum("a,abc->bc", weights, 4 * logM @ Q)

    else:
        raise ValueError(
            f"Unknown metric {metric}. "
            "TLRotate supports only 'euclid' and 'riemann' metrics."
        )


def _warm_start(X, Y, weights, metric="euclid"):
    """Smart initialization of the minimization procedure.

    The loss function being optimized is a weighted sum of loss functions with
    the same structure and for which we know the exact analytic solution [1]_.

    As such, a natural way to warm start the optimization procedure is to list
    the minimizers of "local" loss function and set as Q0 the matrix that
    yields the smallest value for the "global" loss function.

    Note that the initialization is the same for both 'euclid' and 'riemann'.

    References
    ----------
    .. [1] `Procrustes problems in Riemannian manifolds of positive definite
        matrices
        <https://hal.archives-ouvertes.fr/hal-02023293>`_
        R. Bhatia and M. Congedo, Linear Algebra and its Applications,
        Elsevier, 2019, 563, pp.440-445.
    """

    # obtain the solution of the local loss function
    def _get_local_solution(Xi, Yi):
        wX, qX = np.linalg.eig(Xi)
        idx = wX.argsort()[::-1]
        qX = qX[:, idx]
        wY, qY = np.linalg.eig(Yi)
        idx = wY.argsort()[::-1]
        qY = qY[:, idx]
        Qstar = qX @ qY.T
        return Qstar

    Q0_candidates, losses = [], []
    for i in range(len(X)):
        Q0_candidates.append(_get_local_solution(X[i], Y[i]))
        losses.append(_loss(Q0_candidates[i], X, Y, weights, metric=metric))

    i_min = np.argmin(losses)

    return Q0_candidates[i_min]


def _run_minimization(
    Q_ini,
    X_source,
    X_target,
    weights=None,
    metric="euclid",
    tol_step=1e-9,
    maxiter=10_000,
    maxiter_linesearch=32,
):

    Q_1 = Q_ini

    # loop over iterations
    for _ in range(maxiter):

        # get the current value for the loss function
        F_1 = _loss(Q_1, X_target, X_source, weights, metric=metric)

        # get the direction of steepest descent
        direction = _project(
            Q_1,
            _grad(Q_1, X_target, X_source, weights, metric=metric),
        )

        # backtracking line search
        alpha = 1.0
        tau = 0.50
        r = 1e-4
        Q = _retract(Q_1, -alpha * direction)
        F = _loss(Q, X_target, X_source, weights, metric=metric)
        for _ in range(maxiter_linesearch):
            if F_1 - F > r * alpha * np.linalg.norm(direction)**2:
                break
            alpha = tau * alpha
            Q = _retract(Q_1, -alpha * direction)
            F = _loss(Q, X_target, X_source, weights, metric=metric)

        # test if the step size is small
        crit = np.linalg.norm(-alpha * direction)
        if crit <= tol_step:
            break

        # update variables for next iteration
        Q_1 = Q
        F_1 = F

    else:
        warnings.warn("Convergence not reached.")

    return Q, F


def _get_rotation_manifold(
    X_source,
    X_target,
    weights=None,
    metric="euclid",
    tol_step=1e-9,
    maxiter=10_000,
    maxiter_linesearch=32,
):
    r"""Calculate rotation matrix in the manifold.

    Riemannian Procustes analysis in the matrix manifold calculates the
    rotation matrix :math:`\mathbf{Q}` that minimizes the loss function:

    .. math::
        L(Q) = \sum_i w_i \delta^2(X^\text{target}_i, Q X^\text{source}_i Q^T)

    This rotation transforms SPD matrices :math:`\mathbf{M^\text{source}}` from
    the source domain close to matrices :math:`\mathbf{M^\text{target}}` from
    the target domain.
    This manifold optimization problem was first defined in Eq.(35) of [1]_.
    The optimization procedure follows the usual setup for optimization on
    manifolds as described in [2]_.

    Parameters
    ----------
    X_source : ndarray, shape (n_matrices, n, n)
        Set of SPD matrices from the source domain.
    X_target : ndarray, shape (n_matrices, n, n)
        Set of SPD matrices from the target domain.
    weights : None | ndarray, shape (n_matrices,), default=None
        Weights for each pair of matrices. If None, it uses equal weights.
    metric : {"euclid", "riemann"}, default="euclid"
        Distance to minimize between SPD matrices.
    tol_step : float, default=1e-9
        Stopping criterion based on the norm of the descent direction.
    maxiter : int, default=10_000
        Maximum number of iterations in the optimization procedure.
    maxiter_linesearch : int, default=32
        Maximum number of iterations in the line search procedure.

    Returns
    -------
    Q : ndarray, shape (n, n)
        Orthogonal matrix that minimizes the distance between the SPD matrices
        for the source and target domains.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] `Riemannian Procrustes analysis: transfer
        learning for brain-computer interfaces
        <https://hal.archives-ouvertes.fr/hal-01971856>`_
        PLC Rodrigues et al, IEEE Transactions on Biomedical Engineering,
        vol. 66, no. 8, pp. 2390-2401, December, 2018
    .. [2] `An introduction to optimization on smooth manifolds
        <https://www.nicolasboumal.net/book/>`_
        N. Boumal. To appear with Cambridge University Press. June, 2022
    """
    if X_source.shape[0] != X_target.shape[0]:
        raise ValueError("Number of matrices in each domain doesn't match. "
                         f"Got {X_source.shape[0]} and {X_target.shape[0]}.")
    if X_source.shape[1:] != X_target.shape[1:]:
        raise ValueError("Number of channels in each domain doesn't match. "
                         f"Got {X_source.shape[1:]} and {X_target.shape[1:]}.")
    weights = check_weights(weights, len(X_source))

    # initialize the solution with an educated guess
    Q_ini = _warm_start(X_target, X_source, weights, metric=metric)
    if np.linalg.det(Q_ini) < 0:  # make sure it is a rotation
        Q_ini[[0, 1], :] = Q_ini[[1, 0], :]

    # run the optimization procedure on rotation matrices
    Q_posdet, F_posdet = _run_minimization(
        Q_ini,
        X_source,
        X_target,
        weights,
        metric,
        tol_step,
        maxiter,
        maxiter_linesearch,
    )

    # run the optimization procedure on reflection matrices
    Q_ini[[0, 1], :] = Q_ini[[1, 0], :]
    Q_negdet, F_negdet = _run_minimization(
        Q_ini,
        X_source,
        X_target,
        weights,
        metric,
        tol_step,
        maxiter,
        maxiter_linesearch,
    )

    # check which of the options yield a smaller loss value
    if F_posdet < F_negdet:
        return Q_posdet
    else:
        return Q_negdet


###############################################################################


def _get_rotation_tangentspace(X_source, X_target, expl_var):
    r"""Calculate rotation matrix in the tangent space.

    Procustes analysis in the Euclidean tangent space calculates the rotation
    matrix :math:`\mathbf{Q}` that minimizes:

    .. math::
        || X^\text{source} Q - X^\text{target} ||_F^2

    From code provided by the authors [1]_.

    Parameters
    ----------
    X_source : ndarray, shape (n_vectors, n)
        Set of tangent vectors from the source domain.
    X_target : ndarray, shape (n_vectors, n)
        Set of tangent vectors from the target domain.
    expl_var : float
        Dimension reduction applied to the cross product matrix during
        Procrustes analysis.
        If float in (0,1], percentage of variance that needs to be explained.
        Else, number of components.

    Returns
    -------
    Q : ndarray, shape (n, n)
        Orthogonal matrix that minimizes the distance between the tangent
        vectors for the source and target domains.

    Notes
    -----
    .. versionadded:: 0.8

    References
    ----------
    .. [1] `Tangent space alignment: Transfer learning for brain-computer
        interface
        <https://www.frontiersin.org/articles/10.3389/fnhum.2022.1049985/pdf>`_
        A. Bleuzé, J. Mattout and M. Congedo, Frontiers in Human Neuroscience,
        2022
    """
    if X_source.shape != X_target.shape:
        raise ValueError("Inputs shapes don't match. "
                         f"Got {X_source.shape} and {X_target.shape}.")

    C = X_source.T @ X_target
    u, s, vh = np.linalg.svd(C)

    if expl_var <= 1:
        n_comps = np.sum(np.cumsum(s) < expl_var * np.sum(s)) + 1
    else:
        n_comps = int(expl_var)

    u = u[:, :n_comps]
    vh = vh[:n_comps, :]
    return vh.T @ u.T
