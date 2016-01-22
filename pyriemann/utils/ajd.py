"""Aproximate joint diagonalization algorithm."""
import numpy as np


def rjd(X, threshold=1e-8, n_iter_max=1000):
    """Approximate joint diagonalization based on jacobi angle.

    This is a direct implementation of the Cardoso AJD algorithm [1] used in
    JADE. The code is a translation of the matlab code provided in the author
    website.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_channels)
        A set of covariance matrices to diagonalize
    threshold : float (default 1e-8)
        The number of standard deviation to reject artifacts.
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
    wedge

    References
    ----------
    [1] Cardoso, Jean-Francois, and Antoine Souloumiac. Jacobi angles for
    simultaneous diagonalization. SIAM journal on matrix analysis and
    applications 17.1 (1996): 161-164.


    """

    # reshape input matrix
    A = np.concatenate(X, 0).T

    # init variables
    m, nm = A.shape
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
                theta = 0.5 * np.arctan2(toff, ton +
                                         np.sqrt(ton * ton + toff * toff))
                c = np.cos(theta)
                s = np.sin(theta)
                encore = encore | (np.abs(s) > threshold)
                if (np.abs(s) > threshold):
                    tmp = A[:, Ip].copy()
                    A[:, Ip] = c * A[:, Ip] + s * A[:, Iq]
                    A[:, Iq] = c * A[:, Iq] - s * tmp

                    tmp = A[p, :].copy()
                    A[p, :] = c * A[p, :] + s * A[q, :]
                    A[q, :] = c * A[q, :] - s * tmp

                    tmp = V[:, p].copy()
                    V[:, p] = c * V[:, p] + s * V[:, q]
                    V[:, q] = c * V[:, q] - s * tmp

    D = np.reshape(A, (m, nm/m, m)).transpose(1, 0, 2)
    return V, D
