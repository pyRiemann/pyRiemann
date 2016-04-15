"""Aproximate joint diagonalization algorithm."""
import numpy as np
from scipy.linalg import eig, solve
from numpy import array, arange, reshape, arctan2, cos, sin, sqrt, \
     conj, imag, concatenate, eye, zeros, diag, outer

def rjd(X, eps=1e-8, n_iter_max=1000):
    """Approximate joint diagonalization based on jacobi angle.

    This is a direct implementation of the Cardoso AJD algorithm [1] used in
    JADE. The code is a translation of the matlab code provided in the author
    website.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_channels)
        A set of covariance matrices to diagonalize
    eps : float (default 1e-8)
        Tolerance for stopping criterion.
    n_iter_max : int (default 1000)
        The maximum number of iteration to reach convergence.

    Returns
    -------
    V : ndarray, shape (n_channels, n_channels)
        the diagonalizer
    D : ndarray, shape (n_trials, n_channels, n_channels)
        the set of quasi diagonal matrices

    Notes
    -----
    .. versionadded:: 0.2.4

    See Also
    --------
    ajd_pham
    uwedge

    References
    ----------
    [1] Cardoso, Jean-Francois, and Antoine Souloumiac. Jacobi angles for
    simultaneous diagonalization. SIAM journal on matrix analysis and
    applications 17.1 (1996): 161-164.


    """

    # reshape input matrix
    A = concatenate(X, 0).T

    # init variables
    m, nm = A.shape
    V = eye(m)
    encore = True
    k = 0

    while encore:
        encore = False
        k += 1
        if k > n_iter_max:
            break
        for p in range(m - 1):
            for q in range(p + 1, m):

                Ip = arange(p, nm, m)
                Iq = arange(q, nm, m)

                # computation of Givens angle
                g = array([A[p, Ip] - A[q, Iq], A[p, Iq] + A[q, Ip]])
                gg = g.dot(g.T)
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * arctan2(toff, ton +
                                      sqrt(ton * ton + toff * toff))
                c = cos(theta)
                s = sin(theta)
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

    D = reshape(A, (m, nm/m, m)).transpose(1, 0, 2)
    return V, D


def ajd_pham(X, eps=1e-6, n_iter_max=15):
    """Approximate joint diagonalization based on pham's algorithm.

    This is a direct implementation of the PHAM's AJD algorithm [1].

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_channels)
        A set of covariance matrices to diagonalize
    eps : float (default 1e-6)
        tolerance for stoping criterion.
    n_iter_max : int (default 1000)
        The maximum number of iteration to reach convergence.

    Returns
    -------
    V : ndarray, shape (n_channels, n_channels)
        the diagonalizer
    D : ndarray, shape (n_trials, n_channels, n_channels)
        the set of quasi diagonal matrices

    Notes
    -----
    .. versionadded:: 0.2.4

    See Also
    --------
    rjd
    uwedge

    References
    ----------
    [1] Pham, Dinh Tuan. "Joint approximate diagonalization of positive
    definite Hermitian matrices." SIAM Journal on Matrix Analysis and
    Applications 22, no. 4 (2001): 1136-1152.

    """
    nmat = X.shape[0]

    # reshape input matrix
    A = concatenate(X, 0).T

    # init variables
    m, nm = A.shape
    V = eye(m)
    epsi = m * (m - 1) * eps

    for it in range(n_iter_max):
        decr = 0
        for i in range(1, m):
            for j in range(i):
                Ii = arange(i, nm, m)
                Ij = arange(j, nm, m)

                c1 = A[i, Ii]
                c2 = A[j, Ij]

                g12 = np.mean(A[i, Ij] / c1)
                g21 = np.mean(A[i, Ij] / c2)

                omega21 = np.mean(c1 / c2)
                omega12 = np.mean(c2 / c1)
                omega = sqrt(omega12*omega21)

                tmp = sqrt(omega21/omega12)
                tmp1 = (tmp*g12 + g21)/(omega + 1)
                tmp2 = (tmp*g12 - g21)/np.max(omega - 1, 1e-9)

                h12 = tmp1 + tmp2
                h21 = conj((tmp1 - tmp2)/tmp)

                decr = decr + nmat*(g12 * np.conj(h12) + g21 * h21) / 2.0

                tmp = 1 + 1.j * 0.5 * imag(h12 * h21)
                tmp = np.real(tmp + sqrt(tmp ** 2 - h12 * h21))
                T = array([[1, -h12/tmp], [-h21/tmp, 1]])

                A[[i, j], :] = T.dot(A[[i, j], :])
                tmp = np.c_[A[:, Ii], A[:, Ij]]
                tmp = reshape(tmp, (m * nmat, 2), order='F').dot(T.T)

                tmp = reshape(tmp, (m, nmat * 2), order='F')
                A[:, Ii] = tmp[:, :nmat]
                A[:, Ij] = tmp[:, nmat:]
                V[[i, j], :] = T.dot(V[[i, j], :])
        if decr < epsi:
            break
    D = reshape(A, (m, nm/m, m)).transpose(1, 0, 2)
    return V, D

import scipy as sp


def uwedge(X, init=None, eps=1e-7, n_iter_max=100):
    """Approximate joint diagonalization algorithm UWEDGE.

    Uniformly Weighted Exhaustive Diagonalization using Gauss iteration
    (U-WEDGE). Implementation of the AJD algorithm by Tichavsky and Yeredor.
    This is a translation from the matlab code provided by the authors.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_channels)
        A set of covariance matrices to diagonalize
    init: None | ndarray, shape (n_channels, n_channels) (default None)
        Initialization for the diagonalizer.
    eps : float (default 1e-7)
        tolerance for stoping criterion.
    n_iter_max : int (default 1000)
        The maximum number of iteration to reach convergence.

    Returns
    -------
    V : ndarray, shape (n_channels, n_channels)
        the diagonalizer
    D : ndarray, shape (n_trials, n_channels, n_channels)
        the set of quasi diagonal matrices

    Notes
    -----
    .. versionadded:: 0.2.4

    See Also
    --------
    rjd
    ajd_pham

    References
    ----------
    [1] P. Tichavsky, A. Yeredor and J. Nielsen,
        "A Fast Approximate Joint Diagonalization Algorithm
        Using a Criterion with a Block Diagonal Weight Matrix",
        ICASSP 2008, Las Vegas
    [2] P. Tichavsky and A. Yeredor, "Fast Approximate Joint Diagonalization
        Incorporating Weight Matrices" IEEE Transactions of Signal Processing,
        2009.
    """
    L, d, _ = X.shape

    # reshape input matrix
    M = concatenate(X, 0).T

    # init variables
    d, Md = M.shape
    iteration = 0
    improve = 10

    if init is None:
        E, H = eig(M[:, 0:d])
        W_est = diag(1. / np.sqrt(np.abs(E))).dot(H.T)
    else:
        W_est = init

    Ms = array(M)
    Rs = zeros((d, L))

    for k in range(L):
        ini = k*d
        Il = arange(ini, ini + d)
        M[:, Il] = 0.5*(M[:, Il] + M[:, Il].T)
        Ms[:, Il] = W_est.dot(M[:, Il]).dot(W_est.T)
        Rs[:, k] = diag(Ms[:, Il])

    crit = (Ms**2).sum() - (Rs**2).sum()
    while (improve > eps) & (iteration < n_iter_max):
        B = Rs.dot(Rs.T)
        C1 = zeros((d, d))
        for i in range(d):
            C1[:, i] = np.sum(Ms[:, i:Md:d]*Rs, axis=1)

        D0 = B*B.T - outer(diag(B), diag(B))
        A0 = (C1 * B - diag(diag(B)).dot(C1.T)) / (D0 + eye(d))
        A0 += np.eye(d)
        W_est = solve(A0, W_est)

        Raux = W_est.dot(M[:, 0:d]).dot(W_est.T)
        aux = 1./sqrt(np.abs(diag(Raux)))
        W_est = diag(aux).dot(W_est)

        for k in range(L):
            ini = k*d
            Il = arange(ini, ini + d)
            Ms[:, Il] = W_est.dot(M[:, Il]).dot(W_est.T)
            Rs[:, k] = diag(Ms[:, Il])

        crit_new = (Ms**2).sum() - (Rs**2).sum()
        improve = np.abs(crit_new - crit)
        crit = crit_new
        iteration += 1

    D = reshape(Ms, (d, L, d)).transpose(1, 0, 2)
    return W_est, D
