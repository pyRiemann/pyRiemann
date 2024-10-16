"""Aproximate joint diagonalization algorithms."""

import warnings

import numpy as np

from .utils import check_weights, check_function, check_init


def rjd(X, *, init=None, eps=1e-8, n_iter_max=100):
    """Approximate joint diagonalization based on JADE.

    This is an implementation of the orthogonal AJD algorithm [1]_: joint
    approximate diagonalization of eigen-matrices (JADE), based on Jacobi
    angles.
    The code is a translation of the Matlab code provided on the author's
    website.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of symmetric matrices to diagonalize.
    init : None | ndarray, shape (n, n), default=None
        Initialization for the diagonalizer.
    eps : float, default=1e-8
        Tolerance for stopping criterion.
    n_iter_max : int, default=100
        The maximum number of iterations to reach convergence.

    Returns
    -------
    V : ndarray, shape (n, n)
        The diagonalizer, an orthogonal matrix.
    D : ndarray, shape (n_matrices, n, n)
        Set of quasi diagonal matrices, D = V^T X V.

    Notes
    -----
    .. versionadded:: 0.2.4

    See Also
    --------
    ajd

    References
    ----------
    .. [1] `Jacobi angles for simultaneous diagonalization
        <https://epubs.siam.org/doi/abs/10.1137/S0895479893259546>`_
        J.-F. Cardoso and A. Souloumiac, SIAM Journal on Matrix Analysis and
        Applications, 17(1), pp. 161–164, 1996.
    """
    n_matrices, _, _ = X.shape
    # reshape input matrix
    A = np.concatenate(X, 0).T
    n, n_matrices_x_n = A.shape

    # init variables
    if init is None:
        V = np.eye(n)
    else:
        V = check_init(init, n)

    for _ in range(n_iter_max):
        crit = False
        for p in range(n):
            for q in range(p + 1, n):
                Ip = np.arange(p, n_matrices_x_n, n)
                Iq = np.arange(q, n_matrices_x_n, n)

                # computation of Givens rotations
                g = np.array([A[p, Ip] - A[q, Iq], A[p, Iq] + A[q, Ip]])
                gg = g @ g.T
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton**2 + toff**2))
                c = np.cos(theta)
                s = np.sin(theta)
                crit = crit | (np.abs(s) > eps)

                # update of A and V matrices
                if (np.abs(s) > eps):
                    tmp = A[:, Ip].copy()
                    A[:, Ip] = c * A[:, Ip] + s * A[:, Iq]
                    A[:, Iq] = c * A[:, Iq] - s * tmp

                    tmp = A[p, :].copy()
                    A[p, :] = c * A[p, :] + s * A[q, :]
                    A[q, :] = c * A[q, :] - s * tmp

                    tmp = V[:, p].copy()
                    V[:, p] = c * V[:, p] + s * V[:, q]
                    V[:, q] = c * V[:, q] - s * tmp

        if not crit:
            break
    else:
        warnings.warn("Convergence not reached")

    D = np.reshape(A, (n, n_matrices, n)).transpose(1, 0, 2)
    return V, D


def ajd_pham(X, *, init=None, eps=1e-6, n_iter_max=20, sample_weight=None):
    """Approximate joint diagonalization based on Pham's algorithm.

    This is a direct implementation of the AJD algorithm [1]_, optimizing a
    log-likelihood criterion based on the Kullback-Leibler divergence.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices to diagonalize.
    init : None | ndarray, shape (n, n), default=None
        Initialization for the diagonalizer.
    eps : float, default=1e-6
        Tolerance for stoping criterion.
    n_iter_max : int, default=20
        The maximum number of iterations to reach convergence.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix, strictly positive.
        If None, it uses equal weights.

    Returns
    -------
    V : ndarray, shape (n, n)
        The diagonalizer, an invertible matrix.
    D : ndarray, shape (n_matrices, n, n)
        Set of quasi diagonal matrices, D = V X V^H.

    Notes
    -----
    .. versionadded:: 0.2.4

    See Also
    --------
    ajd

    References
    ----------
    .. [1] `Joint approximate diagonalization of positive definite
        Hermitian matrices
        <https://epubs.siam.org/doi/10.1137/S089547980035689X>`_
        D.-T. Pham. SIAM Journal on Matrix Analysis and Applications, 22(4),
        pp. 1136-1152, 2000.
    """
    n_matrices, _, _ = X.shape
    normalized_weight = check_weights(
        sample_weight,
        n_matrices,
        check_positivity=True,
    )  # sum = 1

    # reshape input matrix
    A = np.concatenate(X, axis=0).T
    n, n_matrices_x_n = A.shape

    # init variables
    if init is None:
        V = np.eye(n)
    else:
        V = check_init(init, n)
    V = V.astype(X.dtype)
    epsilon = n * (n - 1) * eps

    for _ in range(n_iter_max):
        crit = 0
        for ii in range(1, n):
            for jj in range(ii):
                Ii = np.arange(ii, n_matrices_x_n, n)
                Ij = np.arange(jj, n_matrices_x_n, n)

                c1 = A[ii, Ii]
                c2 = A[jj, Ij]

                g12 = np.average(A[ii, Ij] / c1, weights=normalized_weight)
                g21 = np.average(A[ii, Ij] / c2, weights=normalized_weight)

                omega21 = np.average(c1 / c2, weights=normalized_weight)
                omega12 = np.average(c2 / c1, weights=normalized_weight)
                omega = np.sqrt(omega12 * omega21)

                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                if np.isrealobj(X):
                    omega = max(omega - 1, 1e-9)
                tmp2 = (tmp * g12 - g21) / omega

                h12 = tmp1 + tmp2
                h21 = np.conj((tmp1 - tmp2) / tmp)

                crit += n_matrices * (g12 * np.conj(h12) + g21 * h21) / 2.0

                tmp = 1 + 0.5j * np.imag(h12 * h21)
                tmp = tmp + np.sqrt(tmp ** 2 - h12 * h21)
                if np.isrealobj(X):
                    tmp = np.real(tmp)
                tau = np.array([[1, np.conj(-h12 / tmp)],
                                [np.conj(-h21 / tmp), 1]])

                A[[ii, jj], :] = tau.conj() @ A[[ii, jj], :]
                tmp = np.c_[A[:, Ii], A[:, Ij]]
                tmp = np.reshape(tmp, (n * n_matrices, 2), order="F")
                tmp = tmp @ tau.T

                tmp = np.reshape(tmp, (n, n_matrices * 2), order="F")
                A[:, Ii] = tmp[:, :n_matrices]
                A[:, Ij] = tmp[:, n_matrices:]
                V[[ii, jj], :] = tau @ V[[ii, jj], :]

        if crit < epsilon:
            break
    else:
        warnings.warn("Convergence not reached")

    D = np.reshape(A, (n, -1, n)).transpose(1, 0, 2).conj()
    return V.astype(X.dtype), D.astype(X.dtype)


def uwedge(X, *, init=None, eps=1e-7, n_iter_max=100):
    """Approximate joint diagonalization based on UWEDGE.

    This is an implementation of the AJD algorithm [1]_ [2]_: uniformly
    weighted exhaustive diagonalization using Gauss iterations (U-WEDGE).
    It is a translation from the Matlab code provided by the authors.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of symmetric matrices to diagonalize.
    init : None | ndarray, shape (n, n), default=None
        Initialization for the diagonalizer.
    eps : float, default=1e-7
        Tolerance for stoping criterion.
    n_iter_max : int, default=100
        The maximum number of iterations to reach convergence.

    Returns
    -------
    V : ndarray, shape (n, n)
        The diagonalizer.
    D : ndarray, shape (n_matrices, n, n)
        Set of quasi diagonal matrices, D = V X V^T.

    Notes
    -----
    .. versionadded:: 0.2.4

    See Also
    --------
    ajd

    References
    ----------
    .. [1] `A Fast Approximate Joint Diagonalization Algorithm Using a
        Criterion with a Block Diagonal Weight Matrix
        <https://ieeexplore.ieee.org/abstract/document/4518361>`_
        P. Tichavsky, A. Yeredor and J. Nielsen. 2008 IEEE International
        Conference on Acoustics, Speech and Signal Processing ICASSP.
    .. [2] `Fast Approximate Joint Diagonalization Incorporating Weight
        Matrices
        <https://ieeexplore.ieee.org/document/4671095>`_
        P. Tichavsky and A. Yeredor. IEEE Trans Signal Process, 57(3), pp.
        878 - 891, 2009.
    """
    n_matrices, _, _ = X.shape
    # reshape input matrix
    M = np.concatenate(X, 0).T
    n, n_matrices_x_n = M.shape

    # init variables
    if init is None:
        E, H = np.linalg.eig(M[:, 0:n])
        V = H.T / np.sqrt(np.abs(E))[:, np.newaxis]
    else:
        V = check_init(init, n)

    Ms = np.array(M)
    Rs = np.zeros((n, n_matrices))

    for k in range(n_matrices):
        ini = k * n
        Il = np.arange(ini, ini + n)
        M[:, Il] = 0.5 * (M[:, Il] + M[:, Il].T)
        Ms[:, Il] = V @ M[:, Il] @ V.T
        Rs[:, k] = np.diag(Ms[:, Il])
    crit = np.sum(Ms ** 2) - np.sum(Rs ** 2)

    for _ in range(n_iter_max):
        B = Rs @ Rs.T
        C1 = np.zeros((n, n))
        for i in range(n):
            C1[:, i] = np.sum(Ms[:, i:n_matrices_x_n:n] * Rs, axis=1)

        D0 = B * B.T - np.outer(np.diag(B), np.diag(B))
        A0 = (C1 * B - np.diag(B)[:, np.newaxis] * C1.T) / (D0 + np.eye(n))
        A0.flat[:: n + 1] += 1
        V = np.linalg.solve(A0, V)

        Raux = V @ M[:, 0:n] @ V.T
        aux = 1. / np.sqrt(np.abs(np.diag(Raux)))
        V = aux[:, np.newaxis] * V

        for k in range(n_matrices):
            ini = k * n
            Il = np.arange(ini, ini + n)
            Ms[:, Il] = V @ M[:, Il] @ V.T
            Rs[:, k] = np.diag(Ms[:, Il])
        crit_new = np.sum(Ms ** 2) - np.sum(Rs ** 2)

        if np.abs(crit_new - crit) < eps:
            break
        crit = crit_new
    else:
        warnings.warn("Convergence not reached")

    D = np.reshape(Ms, (n, n_matrices, n)).transpose(1, 0, 2)
    return V, D


###############################################################################


ajd_functions = {
    "ajd_pham": ajd_pham,
    "rjd": rjd,
    "uwedge": uwedge,
}


def ajd(X, method="ajd_pham", init=None, eps=1e-6, n_iter_max=100, **kwargs):
    """Aproximate joint diagonalization (AJD) according to a method.

    Compute the AJD of a set of matrices according to a method [1]_, estimating
    the joint diagonalizer matrix, diagonalizing the set as much as possible.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of symmetric matrices to diagonalize.
    method : string | callable, default="ajd_pham"
        Method for AJD, can be: "ajd_pham", "rjd", "uwedge", or a callable
        function.
    init : None | ndarray, shape (n, n), default=None
        Initialization for the diagonalizer.
    eps : float, default=1e-6
        Tolerance for stopping criterion.
    n_iter_max : int, default=100
        The maximum number of iterations to reach convergence.
    kwargs : dict
        The keyword arguments passed to the sub function.

    Returns
    -------
    V : ndarray, shape (n, n)
        The diagonalizer.
    D : ndarray, shape (n_matrices, n, n)
        Set of quasi diagonal matrices.

    Notes
    -----
    .. versionadded:: 0.6

    See Also
    --------
    ajd_pham
    rjd
    uwedge

    References
    ----------
    .. [1] `Joint Matrices Decompositions and Blind Source Separation: A survey
        of methods, identification, and applications
        <http://library.utia.cas.cz/separaty/2014/SI/tichavsky-0427607.pdf>`_
        G. Chabriel, M. Kleinsteuber, E. Moreau, H. Shen; P. Tichavsky and A.
        Yeredor. IEEE Signal Process Mag, 31(3), pp. 34-43, 2014.
    """
    ajd_function = check_function(method, ajd_functions)
    V, D = ajd_function(
        X,
        init=init,
        eps=eps,
        n_iter_max=n_iter_max,
        **kwargs,
    )
    return V, D
