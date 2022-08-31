import numpy as np
from ..utils.distance import distance_riemann, distance_euclid


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


def _loss_rie(Q, X, Y, weights=None):

    """
    Loss function for estimating the rotation matrix in RPA using the
    riemannian distance between the class means.
    """

    def loss_i_rie(Q, Xi, Yi):
        return distance_riemann(Xi, Q @ Yi @ Q.T)**2

    X = _check_dimensions(X)
    Y = _check_dimensions(Y)

    if len(X) != len(Y):
        raise ValueError("The number of classes in each domain don't match")

    if weights is None:
        weights = np.ones(len(X)) / len(X)

    L = 0
    for i in range(len(X)):
        L = L + weights[i] * loss_i_rie(Q, X[i], Y[i])

    return L


def _grad_rie(Q, X, Y, weights=None):

    """
    Gradient of loss function using Riemannian distances between class means.
    """

    def _grad_i_rie(Q, Xi, Yi):
        M = np.linalg.inv(Xi) @ Q @ Yi @ Q.T
        w, v = np.linalg.eig(M)
        logM = v @ np.diag(np.log(w)) @ np.linalg.inv(v)
        return 4 * logM @ Q

    X = _check_dimensions(X)
    Y = _check_dimensions(Y)

    if len(X) != len(Y):
        raise ValueError("The number of classes in each domain don't match")

    if weights is None:
        weights = np.ones(len(X)) / len(X)

    G = np.zeros_like(Q)
    for i in range(len(X)):
        G = G + weights[i] * _grad_i_rie(Q, X[i], Y[i])

    return G


def _loss_euc(Q, X, Y, weights=None):

    """
    Loss function for estimating the rotation matrix in RPA using the euclidean
    distance between the class means. Although not being geometry-aware, the
    optimization problem with this option is much faster and more stable to
    minimize compared to using a Riemannian distance between the class means,
    besides yielding comparable performance for transfer learning.
    """

    def _loss_i_euc(Q, Xi, Yi):
        return distance_euclid(Xi, Q @ Yi @ Q.T)**2

    X = _check_dimensions(X)
    Y = _check_dimensions(Y)

    if len(X) != len(Y):
        raise ValueError("The number of classes in each domain don't match")
    if weights is None:
        weights = np.ones(len(X)) / len(X)
    L = 0
    for i in range(len(X)):
        L = L + weights[i] * _loss_i_euc(Q, X[i], Y[i])
    return L


def _grad_euc(Q, X, Y, weights=None):

    """
    Gradient of loss function using Euclidean distances between class means.
    """

    def _grad_i_euc(Q, Xi, Yi):
        return -4 * (Xi - Q @ Yi @ Q.T) @ Q @ Yi

    X = _check_dimensions(X)
    Y = _check_dimensions(Y)

    if len(X) != len(Y):
        raise ValueError("The number of classes in each domain don't match")

    if weights is None:
        weights = np.ones(len(X)) / len(X)

    G = np.zeros_like(Q)
    for i in range(len(X)):
        G = G + weights[i] * _grad_i_euc(Q, X[i], Y[i])

    return G


def _warm_start(X, Y, setup='euc'):

    """Smart initialization of the minimization procedure

    The loss function being optimized is a weighted sum of loss functions with
    the same structure and for which we know the exact analytic solution [1].

    As such, a natural way to warm start the optimization procedure is to list
    the minimizers of "local" loss function and set as Q0 the matrix that
    yields the smallest value for the "global" loss function.

    Note that the initialization is the same for both 'euc' and 'rie'.

    [1] R. Bhatia and M. Congedo "Procrustes problems in Riemannian manifolds
    of positive definite matrices" (2019)

    """

    # obtain the solution of the local loss function
    def _get_local_solution(Xi, Yi):
        wX, qX = np.linalg.eig(Xi)
        idx = wX.argsort()[::-1]
        wX = wX[idx]
        qX = qX[:, idx]
        wY, qY = np.linalg.eig(Yi)
        idx = wY.argsort()[::-1]
        wY = wY[idx]
        qY = qY[:, idx]
        Qstar = qX @ qY.T
        return Qstar

    # decide which setup to use and the associated loss function
    if setup == 'euc':
        loss = _loss_euc
    elif setup == 'rie':
        loss = _loss_rie

    X = _check_dimensions(X)
    Y = _check_dimensions(Y)

    if len(X) != len(Y):
        raise ValueError("The number of classes in each domain don't match")

    Q0_candidates = []

    for i in range(len(X)):
        Q0_candidates.append(_get_local_solution(X[i], Y[i]))

    i_min = np.argmin([loss(Q0_i, X, Y) for Q0_i in Q0_candidates])

    return Q0_candidates[i_min]


def get_rotation_matrix(M_source, M_target, weights=None, setup='euc',
                        tol_step=1e-9, maxiter=10_000, maxiter_linesearch=32):

    """Calculate rotation matrix for the Riemannian Procustes Analysis

    Get the rotation matrix that minimizes the loss function

        L(Q) = sum_{i = 1}^L w_i delta^2(M_target_i, Q M_source_i Q^T)

    The solution can then be used to transform the data points from the source
    domain so that their class means are close to the those from the target
    domain. This manifold optimization problem was first defined in Eq(35) of
    [1]. Our optimization procedure follows the usual setup for optimization
    on manifolds as described in [2].

    Parameters
    ----------
    M_source : ndarray, shape (L, n, n)
        Set with the means of the L class from the source domain
    M_target : ndarray, shape (L, n, n)
        Set with the means of the L class from the source domain
    weights : None | array, shape (L), default=None
        Set with the weights to assign for each class. If None, then give the
        same weight for each class
    setup : str, default='euc'
        Which type of distance to minimize between the class means, either
        'euc' for euclidean distance or 'rie' for the AIRM-induced distance
        between SPD matrices
    maxiter : int, default=10_000
        Maximum number of iterations in the optimization procedure
    maxiter_linesearch : int, default=32
        Maximum number in the line search procedure

    Returns
    -------
    Q : ndarray, shape (n, n)
        Orthogonal matrix that minimizes the distance between the class means
        for the source and target domains

    References
    ----------
    .. [1] P. Rodrigues et al. "Riemannian Procrustes Analysis : Transfer
    Learning for Brain-Computer Interfaces" (2018)

    .. [2] N. Boumal "An introduction to optimization on smooth manifolds"
    """

    M_source = _check_dimensions(M_source)
    M_target = _check_dimensions(M_target)

    if len(M_source) != len(M_target):
        raise ValueError("The number of classes in each domain don't match")

    # decide which setup to use and the associated loss/grad functions
    if setup == 'euc':
        loss = _loss_euc
        grad = _grad_euc
    elif setup == 'rie':
        loss = _loss_rie
        grad = _grad_rie

    # initialize the solution with an educated guess
    Qk_1 = _warm_start(M_target, M_source)

    # loop over iterations
    for iter in range(maxiter):

        # get the current value for the loss function
        Fk_1 = loss(Qk_1, M_target, M_source, weights)

        # get the direction of steepest descent
        direction = _project(Qk_1, grad(Qk_1, M_target, M_source, weights))

        # backtracking line search
        alpha = 1.0
        tau = 0.50
        r = 1e-4
        Qk = _retract(Qk_1, -alpha * direction)
        Fk = loss(Qk, M_target, M_source)
        for iter_linesearch in range(maxiter_linesearch):
            if Fk_1 - Fk > r * alpha * np.linalg.norm(direction)**2:
                break
            alpha = tau * alpha
            Qk = _retract(Qk_1, -alpha * direction)
            Fk = loss(Qk, M_target, M_source, weights)

        # test if the step size is small
        crit = np.linalg.norm(-alpha * direction)
        if crit <= tol_step:
            break

        # update variables for next iteration
        Qk_1 = Qk
        Fk_1 = Fk

    return Qk
