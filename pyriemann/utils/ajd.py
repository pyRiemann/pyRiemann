"""Aproximate joint diagonalization algorithms."""

import warnings

import numpy as np
from einops import rearrange

from ._backend import get_namespace, diag_indices, xpd
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
    xp = get_namespace(X, init)
    n_matrices, _, _ = X.shape
    # reshape input matrix: (n_matrices, n, n) -> (n, n_matrices*n)
    A = rearrange(X, 'matrices n1 n2 -> n1 (matrices n2)')
    n, n_matrices_x_n = A.shape

    # init variables
    if init is None:
        V = xp.eye(n, dtype=X.dtype, device=xpd(X))
    else:
        V = check_init(init, n, like=X)

    for _ in range(n_iter_max):
        crit = False
        for p in range(n):
            for q in range(p + 1, n):
                Ip = list(range(p, n_matrices_x_n, n))
                Iq = list(range(q, n_matrices_x_n, n))

                # computation of Givens rotations
                g = xp.stack(
                    (A[p, Ip] - A[q, Iq], A[p, Iq] + A[q, Ip]),
                    axis=0,
                )
                gg = g @ g.mT
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * xp.atan2(
                    toff,
                    ton + xp.sqrt(ton**2 + toff**2),
                )
                c = xp.cos(theta)
                s = xp.sin(theta)
                abs_s = float(xp.abs(s))
                crit = crit or (abs_s > eps)

                # update of A and V matrices
                if abs_s > eps:
                    tmp = xp.zeros_like(A[:, Ip])
                    tmp[...] = A[:, Ip]
                    A[:, Ip] = c * A[:, Ip] + s * A[:, Iq]
                    A[:, Iq] = c * A[:, Iq] - s * tmp

                    tmp = xp.zeros_like(A[p, :])
                    tmp[...] = A[p, :]
                    A[p, :] = c * A[p, :] + s * A[q, :]
                    A[q, :] = c * A[q, :] - s * tmp

                    tmp = xp.zeros_like(V[:, p])
                    tmp[...] = V[:, p]
                    V[:, p] = c * V[:, p] + s * V[:, q]
                    V[:, q] = c * V[:, q] - s * tmp

        if not crit:
            break
    else:
        warnings.warn("Convergence not reached")

    D = rearrange(A, 'n1 (matrices n2) -> matrices n1 n2', matrices=n_matrices)
    return (
        xp.asarray(V, dtype=X.dtype, device=xpd(X)),
        xp.asarray(D, dtype=X.dtype, device=xpd(X)),
    )


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
    xp = get_namespace(X)
    n_matrices, _, _ = X.shape
    normalized_weight = check_weights(
        sample_weight,
        n_matrices,
        check_positivity=True,
        like=X,
    )  # sum = 1

    # reshape input matrix: (n_matrices, n, n) -> (n, n_matrices*n)
    # Matches np.concatenate(X, axis=0).T column ordering for HPD correctness
    n = X.shape[-1]
    A = xp.asarray(xp.reshape(X, (-1, n)).mT, copy=True)
    _, n_matrices_x_n = A.shape

    if init is None:
        V = xp.eye(n, dtype=X.dtype, device=xpd(X))
    else:
        V = check_init(init, n, like=X)
    V = xp.asarray(V, dtype=X.dtype, device=xpd(X))
    epsilon = n * (n - 1) * eps
    is_real = xp.isdtype(X.dtype, "real floating")

    for _ in range(n_iter_max):
        crit = 0
        for ii in range(1, n):
            for jj in range(ii):
                Ii = list(range(ii, n_matrices_x_n, n))
                Ij = list(range(jj, n_matrices_x_n, n))

                c1 = A[ii, Ii]
                c2 = A[jj, Ij]

                g12 = xp.sum(normalized_weight * (A[ii, Ij] / c1))
                g21 = xp.sum(normalized_weight * (A[ii, Ij] / c2))

                omega21 = xp.sum(normalized_weight * (c1 / c2))
                omega12 = xp.sum(normalized_weight * (c2 / c1))
                omega = xp.sqrt(omega12 * omega21)

                tmp = xp.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                if is_real:
                    omega = xp.maximum(
                        omega - 1,
                        xp.asarray(1e-9, dtype=omega.dtype, device=xpd(omega)),
                    )
                tmp2 = (tmp * g12 - g21) / omega

                h12 = tmp1 + tmp2
                h21 = xp.conj((tmp1 - tmp2) / tmp)

                crit += float(xp.real(
                    n_matrices * (
                        g12 * xp.conj(h12) + g21 * h21
                    ) / 2.0
                ))

                if is_real:
                    tau_den = 1 + xp.real(xp.sqrt(1 - h12 * h21))
                else:
                    tau_den = 1 + 0.5j * xp.imag(h12 * h21)
                    tau_den = tau_den + xp.sqrt(
                        tau_den ** 2 - h12 * h21
                    )

                tau = xp.eye(2, dtype=X.dtype, device=xpd(X))
                tau[0, 1] = xp.conj(-h12 / tau_den)
                tau[1, 0] = xp.conj(-h21 / tau_den)

                A[[ii, jj], :] = xp.conj(tau) @ A[[ii, jj], :]
                tmp = xp.stack((A[:, Ii], A[:, Ij]), axis=-1)
                tmp = tmp @ tau.mT
                A[:, Ii] = tmp[..., 0]
                A[:, Ij] = tmp[..., 1]
                V[[ii, jj], :] = tau @ V[[ii, jj], :]

        if crit < epsilon:
            break
    else:
        warnings.warn("Convergence not reached")

    D = xp.conj(
        xp.permute_dims(xp.reshape(A, (n, n_matrices, n)), (1, 0, 2))
    )
    return (
        xp.asarray(V, dtype=X.dtype, device=xpd(X)),
        xp.asarray(D, dtype=X.dtype, device=xpd(X)),
    )


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
    xp = get_namespace(X, init)
    n_matrices, _, _ = X.shape
    # reshape input matrix: (n_matrices, n, n) -> (n, n_matrices*n)
    M = rearrange(X, 'matrices n1 n2 -> n1 (matrices n2)')
    n, n_matrices_x_n = M.shape

    # init variables
    if init is None:
        E, H = xp.linalg.eig(M[:, 0:n])
        if xp.isdtype(X.dtype, "real floating"):
            E = xp.real(E)
            H = xp.real(H)
        V = H.mT / xp.sqrt(
            xp.abs(E)
        )[:, np.newaxis]
    else:
        V = check_init(init, n, like=X)

    Ms = xp.zeros_like(M)
    Ms[...] = M
    Rs = xp.zeros((n, n_matrices), dtype=X.dtype, device=xpd(X))

    for k in range(n_matrices):
        ini = k * n
        Il = list(range(ini, ini + n))
        M[:, Il] = 0.5 * (M[:, Il] + M[:, Il].mT)
        Ms[:, Il] = V @ M[:, Il] @ V.mT
        Rs[:, k] = xp.linalg.diagonal(Ms[:, Il])
    crit = float(xp.real(
        xp.sum(Ms ** 2) - xp.sum(Rs ** 2)
    ))

    for _ in range(n_iter_max):
        B = Rs @ Rs.mT
        C1 = xp.zeros((n, n), dtype=X.dtype, device=xpd(X))
        for i in range(n):
            C1[:, i] = xp.sum(Ms[:, i:n_matrices_x_n:n] * Rs, axis=1)

        diag_b = xp.linalg.diagonal(B)
        D0 = B * B.mT - xp.linalg.outer(diag_b, diag_b)
        A0 = (
            C1 * B
            - diag_b[:, np.newaxis] * C1.mT
        ) / (D0 + xp.eye(n, dtype=X.dtype, device=xpd(X)))
        diag0, diag1 = diag_indices(n, xp=xp, like=X)
        A0[diag0, diag1] += 1
        V = xp.linalg.solve(A0, V)

        Raux = V @ M[:, 0:n] @ V.mT
        aux = 1. / xp.sqrt(xp.abs(xp.linalg.diagonal(Raux)))
        V = aux[:, np.newaxis] * V

        for k in range(n_matrices):
            ini = k * n
            Il = list(range(ini, ini + n))
            Ms[:, Il] = V @ M[:, Il] @ V.mT
            Rs[:, k] = xp.linalg.diagonal(Ms[:, Il])
        crit_new = float(xp.real(
            xp.sum(Ms ** 2) - xp.sum(Rs ** 2)
        ))

        if abs(crit_new - crit) < eps:
            break
        crit = crit_new
    else:
        warnings.warn("Convergence not reached")

    D = rearrange(
        Ms, 'n1 (matrices n2) -> matrices n1 n2', matrices=n_matrices,
    )
    return (
        xp.asarray(V, dtype=X.dtype, device=xpd(X)),
        xp.asarray(D, dtype=X.dtype, device=xpd(X)),
    )


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
