"""Aproximate joint diagonalization algorithms."""

import numpy as np


def _get_normalized_weight(sample_weight, data):
    """Get the normalized sample weights, strictly positive.

    If None provided, weights are initialized to 1. In any case, weights are
    normalized (sum equal to 1).
    """
    if sample_weight is None:
        sample_weight = np.ones(data.shape[0])
    else:
        if len(sample_weight) != data.shape[0]:
            raise ValueError("len of weight must be equal to len of data.")
        if any(sample_weight <= 0):
            raise ValueError("values of weight must be strictly positive.")
    normalized_weight = sample_weight / np.sum(sample_weight)
    return normalized_weight


def rjd(X, eps=1e-8, n_iter_max=1000):
    """Approximate joint diagonalization based on Jacobi angles.

    This is a direct implementation of the AJD algorithm by Cardoso and
    Souloumiac [1]_ used in JADE. The code is a translation of the Matlab code
    provided in the author website.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of symmetric matrices to diagonalize.
    eps : float, default=1e-8
        Tolerance for stopping criterion.
    n_iter_max : int, default=1000
        The maximum number of iterations to reach convergence.

    Returns
    -------
    V : ndarray, shape (n_channels, n_channels)
        The diagonalizer, an orthogonal matrix.
    D : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of quasi diagonal matrices.

    Notes
    -----
    .. versionadded:: 0.2.4

    See Also
    --------
    ajd_pham
    uwedge

    References
    ----------
    .. [1] J.-F. Cardoso and A. Souloumiac, "Jacobi angles for simultaneous
        diagonalization", SIAM J Matrix Anal Appl, 1996
    """

    # reshape input matrix
    A = np.concatenate(X, 0).T

    # init variables
    m, nm = A.shape  # n_channels, n_matrices_x_channels
    V = np.eye(m)
    encore = True
    k = 0

    while encore:
        encore = False
        k += 1
        if k > n_iter_max:
            break
        for p in range(m - 1):
            for q in range(p + 1, m):

                Ip = np.arange(p, nm, m)
                Iq = np.arange(q, nm, m)

                # computation of Givens angle
                g = np.array([A[p, Ip] - A[q, Iq], A[p, Iq] + A[q, Ip]])
                gg = np.dot(g, g.T)
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * np.arctan2(
                    toff, ton + np.sqrt(ton * ton + toff * toff))
                c = np.cos(theta)
                s = np.sin(theta)
                encore = encore | (np.abs(s) > eps)
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

    D = np.reshape(A, (m, int(nm / m), m)).transpose(1, 0, 2)
    return V, D


def ajd_pham(X, eps=1e-6, n_iter_max=15, sample_weight=None):
    """Approximate joint diagonalization based on Pham's algorithm.

    This is a direct implementation of the Pham's AJD algorithm [1]_.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices to diagonalize.
    eps : float, default=1e-6
        Tolerance for stoping criterion.
    n_iter_max : int, default=15
        The maximum number of iterations to reach convergence.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix, strictly positive.
        If None, it uses equal weights.

    Returns
    -------
    V : ndarray, shape (n_channels, n_channels)
        The diagonalizer, an invertible matrix.
    D : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of quasi diagonal matrices.

    Notes
    -----
    .. versionadded:: 0.2.4

    See Also
    --------
    rjd
    uwedge

    References
    ----------
    .. [1] D.-T. Pham, "Joint approximate diagonalization of positive definite
        Hermitian matrices", SIAM J Matrix Anal Appl, 2001
    """
    normalized_weight = _get_normalized_weight(sample_weight, X)  # sum = 1

    n_matrices, _, _ = X.shape
    # Reshape input matrix
    A = np.concatenate(X, axis=0).T

    # Init variables
    n_channels, n_matrices_x_channels = A.shape
    V = np.eye(n_channels)
    epsilon = n_channels * (n_channels - 1) * eps

    for it in range(n_iter_max):
        decr = 0
        for ii in range(1, n_channels):
            for jj in range(ii):
                Ii = np.arange(ii, n_matrices_x_channels, n_channels)
                Ij = np.arange(jj, n_matrices_x_channels, n_channels)

                c1 = A[ii, Ii]
                c2 = A[jj, Ij]

                g12 = np.average(A[ii, Ij] / c1, weights=normalized_weight)
                g21 = np.average(A[ii, Ij] / c2, weights=normalized_weight)

                omega21 = np.average(c1 / c2, weights=normalized_weight)
                omega12 = np.average(c2 / c1, weights=normalized_weight)
                omega = np.sqrt(omega12 * omega21)

                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9)

                h12 = tmp1 + tmp2
                h21 = np.conj((tmp1 - tmp2) / tmp)

                decr += n_matrices * (g12 * np.conj(h12) + g21 * h21) / 2.0

                tmp = 1 + 1.j * 0.5 * np.imag(h12 * h21)
                tmp = np.real(tmp + np.sqrt(tmp ** 2 - h12 * h21))
                tau = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])

                A[[ii, jj], :] = np.dot(tau, A[[ii, jj], :])
                tmp = np.c_[A[:, Ii], A[:, Ij]]
                tmp = np.reshape(tmp, (n_channels * n_matrices, 2), order='F')
                tmp = np.dot(tmp, tau.T)

                tmp = np.reshape(tmp, (n_channels, n_matrices * 2), order='F')
                A[:, Ii] = tmp[:, :n_matrices]
                A[:, Ij] = tmp[:, n_matrices:]
                V[[ii, jj], :] = np.dot(tau, V[[ii, jj], :])
        if decr < epsilon:
            break
    D = np.reshape(A, (n_channels, -1, n_channels)).transpose(1, 0, 2)
    return V, D


def uwedge(X, init=None, eps=1e-7, n_iter_max=100):
    """Approximate joint diagonalization based on UWEDGE.

    Implementation of the AJD algorithm by Tichavsky and Yeredor [1]_ [2]_:
    uniformly weighted exhaustive diagonalization using Gauss iterations
    (U-WEDGE). This is a translation from the matlab code provided by the
    authors.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of symmetric matrices to diagonalize.
    init : None | ndarray, shape (n_channels, n_channels), default=None
        Initialization for the diagonalizer.
    eps : float, default=1e-7
        Tolerance for stoping criterion.
    n_iter_max : int, default=100
        The maximum number of iterations to reach convergence.

    Returns
    -------
    V : ndarray, shape (n_channels, n_channels)
        The diagonalizer.
    D : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of quasi diagonal matrices.

    Notes
    -----
    .. versionadded:: 0.2.4

    See Also
    --------
    ajd_pham
    rjd

    References
    ----------
    .. [1] P. Tichavsky, A. Yeredor and J. Nielsen, "A Fast Approximate Joint
        Diagonalization Algorithm Using a Criterion with a Block Diagonal
        Weight Matrix", ICASSP, 2008
    .. [2] P. Tichavsky and A. Yeredor, "Fast Approximate Joint Diagonalization
        Incorporating Weight Matrices", IEEE Trans Signal Process, 2009
    """
    n_matrices, d, _ = X.shape
    # reshape input matrix
    M = np.concatenate(X, 0).T

    # init variables
    d, Md = M.shape  # n_channels, n_matrices_x_channels
    iteration = 0
    improve = 10

    if init is None:
        E, H = np.linalg.eig(M[:, 0:d])
        W_est = np.dot(np.diag(1. / np.sqrt(np.abs(E))), H.T)
    else:
        W_est = init

    Ms = np.array(M)
    Rs = np.zeros((d, n_matrices))

    for k in range(n_matrices):
        ini = k*d
        Il = np.arange(ini, ini + d)
        M[:, Il] = 0.5*(M[:, Il] + M[:, Il].T)
        Ms[:, Il] = np.dot(np.dot(W_est, M[:, Il]), W_est.T)
        Rs[:, k] = np.diag(Ms[:, Il])

    crit = np.sum(Ms**2) - np.sum(Rs**2)
    while (improve > eps) & (iteration < n_iter_max):
        B = np.dot(Rs, Rs.T)
        C1 = np.zeros((d, d))
        for i in range(d):
            C1[:, i] = np.sum(Ms[:, i:Md:d]*Rs, axis=1)

        D0 = B*B.T - np.outer(np.diag(B), np.diag(B))
        A0 = (C1 * B - np.dot(np.diag(np.diag(B)), C1.T)) / (D0 + np.eye(d))
        A0 += np.eye(d)
        W_est = np.linalg.solve(A0, W_est)

        Raux = np.dot(np.dot(W_est, M[:, 0:d]), W_est.T)
        aux = 1./np.sqrt(np.abs(np.diag(Raux)))
        W_est = np.dot(np.diag(aux), W_est)

        for k in range(n_matrices):
            ini = k*d
            Il = np.arange(ini, ini + d)
            Ms[:, Il] = np.dot(np.dot(W_est, M[:, Il]), W_est.T)
            Rs[:, k] = np.diag(Ms[:, Il])

        crit_new = np.sum(Ms**2) - np.sum(Rs**2)
        improve = np.abs(crit_new - crit)
        crit = crit_new
        iteration += 1

    D = np.reshape(Ms, (d, n_matrices, d)).transpose(1, 0, 2)
    return W_est, D
