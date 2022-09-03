import numpy as np
from ..utils.distance import distance
import warnings


def _project(X, U):

    """
    Project U on the tangent space of orthogonal matrices with base point X
    """

    return X @ ((X.T @ U) - (X.T @ U).T) / 2


def _retract(X, v):

    """
    Retraction taking tangent vector v at base point X back to manifold of
    orthogonal matrices
    """

    Q, R = np.linalg.qr(X + v)
    for i, ri in enumerate(np.diag(R)):
        if ri < 0:
            Q[:, i] = -1 * Q[:, i]
    return Q


def _check_dimensions(M):
    if isinstance(M, list):
        M = np.stack(M)
    if isinstance(M, np.ndarray):
        if M.ndim == 2:
            M = M[None, :, :]
    return M


def _loss(Q, X, Y, weights=None, metric='euclid'):

    """
    Loss function for estimating the rotation matrix in RPA.
    """

    X = _check_dimensions(X)
    Y = _check_dimensions(Y)

    if len(X) != len(Y):
        raise ValueError("The number of classes in each domain don't match")

    if weights is None:
        weights = np.ones(len(X)) / len(X)

    return weights @ distance(X, Q @ Y @ Q.T, metric=metric)


def _grad_rie(Q, X, Y, weights=None):

    """
    Gradient of loss function using Riemannian distances between class means.
    """

    X = _check_dimensions(X)
    Y = _check_dimensions(Y)

    if len(X) != len(Y):
        raise ValueError("The number of classes in each domain don't match")

    if weights is None:
        weights = np.ones(len(X)) / len(X)

    M = np.linalg.inv(X) @ Q @ Y @ Q.T
    eigvals, eigvecs = np.linalg.eig(M)
    logeigvals = np.expand_dims(np.log(eigvals), -2)
    logM = (eigvecs * logeigvals) @ np.linalg.inv(eigvecs)
    G = np.einsum('a,abc->bc', weights, 4 * logM @ Q)

    return G


def _grad_euc(Q, X, Y, weights=None):

    """
    Gradient of loss function using Euclidean distances between class means.
    """

    X = _check_dimensions(X)
    Y = _check_dimensions(Y)

    if len(X) != len(Y):
        raise ValueError("The number of classes in each domain don't match")

    if weights is None:
        weights = np.ones(len(X)) / len(X)

    G = np.einsum('a,abc->bc', weights, -4 * (X - Q @ Y @ Q.T) @ Q @ Y)

    return G


def _warm_start(X, Y, metric='euclid'):

    """Smart initialization of the minimization procedure

    The loss function being optimized is a weighted sum of loss functions with
    the same structure and for which we know the exact analytic solution [1].

    As such, a natural way to warm start the optimization procedure is to list
    the minimizers of "local" loss function and set as Q0 the matrix that
    yields the smallest value for the "global" loss function.

    Note that the initialization is the same for both 'euclid' and 'riemann'.

    [1] R. Bhatia and M. Congedo "Procrustes problems in Riemannian manifolds
    of positive definite matrices" (2019)

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

    X = _check_dimensions(X)
    Y = _check_dimensions(Y)

    if len(X) != len(Y):
        raise ValueError("The number of classes in each domain don't match")

    Q0_candidates = []

    for i in range(len(X)):
        Q0_candidates.append(_get_local_solution(X[i], Y[i]))

    i_min = np.argmin(
        [_loss(Q0_i, X, Y, metric=metric) for Q0_i in Q0_candidates])

    return Q0_candidates[i_min]


def get_rotation_matrix(M_source, M_target, weights=None, metric='euclid',
                        tol_step=1e-9, maxiter=10_000, maxiter_linesearch=32):

    """Calculate rotation matrix for the Riemannian Procustes Analysis

    Get the rotation matrix that minimizes the loss function

    .. math::
        L(Q) = \sum_i w_i delta^2(M_target_i, Q M_source_i Q^T)

    The solution can then be used to transform the data points from the source
    domain so that their class means are close to the those from the target
    domain. This manifold optimization problem was first defined in Eq(35) of
    [1]_. The optimization procedure follows the usual setup for optimization
    on manifolds as described in [2]_.

    Parameters
    ----------
    M_source : ndarray, shape (n_classes, n, n)
        Set with the means of the n_classes from the source domain
    M_target : ndarray, shape (n_classes, n, n)
        Set with the means of the n_classes from the target domain
    weights : None | array, shape (n_classes,), default=None
        Set with the weights to assign for each class. If None, then give the
        same weight for each class
    metric : str, default='euclid'
        Which type of distance to minimize between the class means, either
        'euclid' for euclidean distance or 'riemann' for the AIRM-induced
        distance between SPD matrices
    maxiter : int, default=10_000
        Maximum number of iterations in the optimization procedure
    maxiter_linesearch : int, default=32
        Maximum number of iterations in the line search procedure

    Returns
    -------
    Q : ndarray, shape (n, n)
        Orthogonal matrix that minimizes the distance between the class means
        for the source and target domains

    References
    ----------
    .. [1] P. Rodrigues et al. "Riemannian Procrustes Analysis: Transfer
           Learning for Brain-Computer Interfaces" (2018)

    .. [2] N. Boumal "An introduction to optimization on smooth manifolds"
    """

    M_source = _check_dimensions(M_source)
    M_target = _check_dimensions(M_target)

    if len(M_source) != len(M_target):
        raise ValueError("The number of classes in each domain don't match")

    # decide which setup to use and the associated loss/grad functions
    if metric == 'euclid':
        _grad = _grad_euc
    elif metric == 'riemann':
        _grad = _grad_rie

    # initialize the solution with an educated guess
    Qk_1 = _warm_start(M_target, M_source, metric=metric)

    # loop over iterations
    for iter in range(maxiter):

        # get the current value for the loss function
        Fk_1 = _loss(Qk_1, M_target, M_source, weights, metric=metric)

        # get the direction of steepest descent
        direction = _project(Qk_1, _grad(Qk_1, M_target, M_source, weights))

        # backtracking line search
        alpha = 1.0
        tau = 0.50
        r = 1e-4
        Qk = _retract(Qk_1, -alpha * direction)
        Fk = _loss(Qk, M_target, M_source, metric=metric)
        for iter_linesearch in range(maxiter_linesearch):
            if Fk_1 - Fk > r * alpha * np.linalg.norm(direction)**2:
                break
            alpha = tau * alpha
            Qk = _retract(Qk_1, -alpha * direction)
            Fk = _loss(Qk, M_target, M_source, weights, metric=metric)

        # test if the step size is small
        crit = np.linalg.norm(-alpha * direction)
        if crit <= tol_step:
            break

        # update variables for next iteration
        Qk_1 = Qk
        Fk_1 = Fk

    else:
        warnings.warn('Convergence not reached.')

    return Qk
