import warnings

import numpy as np
from scipy.linalg import block_diag
from sklearn.covariance import oas, ledoit_wolf, fast_mcd, empirical_covariance

from .test import is_square

# Mapping different estimator on the sklearn toolbox


def _lwf(X):
    """Wrapper for sklearn ledoit wolf covariance estimator"""
    C, _ = ledoit_wolf(X.T)
    return C


def _mcd(X):
    """Wrapper for sklearn mcd covariance estimator"""
    _, C, _, _ = fast_mcd(X.T)
    return C


def _oas(X):
    """Wrapper for sklearn oas covariance estimator"""
    C, _ = oas(X.T)
    return C


def _scm(X):
    """Wrapper for sklearn sample covariance estimator"""
    return empirical_covariance(X.T)


def _sch(X):
    r"""Schaefer-Strimmer covariance estimator.

    Shrinkage estimator using method from [1]_:

    .. math::
            \hat{\Sigma} = (1 - \gamma)\Sigma_{scm} + \gamma T

    where :math:`T` is the diagonal target matrix:

    .. math::
        T_{i,j} = \{ \Sigma_{scm}^{ii} \text{if} i = j, 0 \text{otherwise} \}

    Note that the optimal :math:`\gamma` is estimated by the authors' method.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Multi-channel time-series.

    Returns
    -------
    cov : ndarray, shape (n_channels, n_channels)
        Schaefer-Strimmer shrinkage covariance matrix.

    Notes
    -----
    .. versionadded:: 0.3

    References
    ----------
    .. [1] `A shrinkage approach to large-scale covariance estimation and
        implications for functional genomics
        <https://pubmed.ncbi.nlm.nih.gov/16646851/>`_
        J. Schafer, and K. Strimmer. Statistical Applications in Genetics and
        Molecular Biology, Volume 4, Issue 1, November 2005.
    """
    n_times = X.shape[1]
    X_c = (X.T - X.T.mean(axis=0)).T
    C_scm = 1. / n_times * X_c @ X_c.T

    # Compute optimal gamma, the weigthing between SCM and shrinkage estimator
    R = (n_times / ((n_times - 1.) * np.outer(X.std(axis=1), X.std(axis=1))))
    R *= C_scm
    var_R = (X_c ** 2) @ (X_c ** 2).T - 2 * C_scm * (X_c @ X_c.T)
    var_R += n_times * C_scm ** 2
    Xvar = np.outer(X.var(axis=1), X.var(axis=1))
    var_R *= n_times / ((n_times - 1) ** 3 * Xvar)
    R -= np.diag(np.diag(R))
    var_R -= np.diag(np.diag(var_R))
    gamma = max(0, min(1, var_R.sum() / (R ** 2).sum()))

    sigma = (1. - gamma) * (n_times / (n_times - 1.)) * C_scm
    shrinkage = gamma * (n_times / (n_times - 1.)) * np.diag(np.diag(C_scm))
    return sigma + shrinkage


def _check_est(est):
    """Check if a given estimator is valid."""

    # Check estimator exist and return the correct function
    estimators = {
        'corr': np.corrcoef,
        'cov': np.cov,
        'lwf': _lwf,
        'mcd': _mcd,
        'oas': _oas,
        'sch': _sch,
        'scm': _scm,
    }

    if callable(est):
        # All good (cross your fingers)
        pass
    elif est in estimators.keys():
        # Map the corresponding estimator
        est = estimators[est]
    else:
        # raise an error
        raise ValueError(
            """%s is not an valid estimator ! Valid estimators are : %s or a
             callable function""" % (est, (' , ').join(estimators.keys())))
    return est


def covariances(X, estimator='cov'):
    """Estimation of covariance matrix.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_times)
        Multi-channel time-series.
    estimator : {'corr', 'cov', 'lwf', 'mcd', 'oas', 'sch', 'scm'}, \
            default='scm'
        Covariance matrix estimator [1]_:

        * 'corr' for correlation coefficient matrix [2]_,
        * 'cov' for numpy based covariance matrix [3]_,
        * 'lwf' for shrunk Ledoit-Wolf covariance matrix [4]_,
        * 'mcd' for minimum covariance determinant matrix [5]_,
        * 'oas' for oracle approximating shrunk covariance matrix [6]_,
        * 'sch' for Schaefer-Strimmer covariance matrix [7]_,
        * 'scm' for sample covariance matrix [8]_,
        * or a callable function.

        For regularization, consider 'lwf' or 'oas'.

    Returns
    -------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Covariance matrices.

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/covariance.html
    .. [2] https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
    .. [3] https://numpy.org/doc/stable/reference/generated/numpy.cov.html
    .. [4] https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html
    .. [5] https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html
    .. [6] https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html
    .. [7] http://doi.org/10.2202/1544-6115.1175
    .. [8] https://scikit-learn.org/stable/modules/generated/sklearn.covariance.empirical_covariance.html
    """  # noqa
    est = _check_est(estimator)
    n_matrices, n_channels, n_times = X.shape
    covmats = np.empty((n_matrices, n_channels, n_channels))
    for i in range(n_matrices):
        covmats[i] = est(X[i])
    return covmats


def covariances_EP(X, P, estimator='cov'):
    """Special form covariance matrix, concatenating a prototype P.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_times)
        Multi-channel time-series.
    P : ndarray, shape (n_channels_proto, n_times)
        Multi-channel prototype.
    estimator : {'cov', 'scm', 'lwf', 'oas', 'mcd', 'sch', 'corr'}, \
            default='cov'
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.

    Returns
    -------
    covmats : ndarray, shape (n_matrices, n_channels + n_channels_proto, \
            n_channels + n_channels_proto)
        Covariance matrices.
    """
    est = _check_est(estimator)
    n_matrices, n_channels, n_times = X.shape
    n_channels_proto, n_times_p = P.shape
    if n_times_p != n_times:
        raise ValueError(
            f"X and P do not have the same n_times: {n_times} and {n_times_p}")
    covmats = np.empty((n_matrices, n_channels + n_channels_proto,
                        n_channels + n_channels_proto))
    for i in range(n_matrices):
        covmats[i] = est(np.concatenate((P, X[i]), axis=0))
    return covmats


def covariances_X(X, estimator='scm', alpha=0.2):
    """Special form covariance matrix, embedding input X.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_times)
        Multi-channel time-series.
    estimator : {'cov', 'scm', 'lwf', 'oas', 'mcd', 'sch', 'corr'}, \
            default='scm'
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.
    alpha : float, default=0.2
        Regularization parameter (strictly positive).

    Returns
    -------
    covmats : ndarray, shape (n_matrices, n_channels + n_times, n_channels + \
            n_times)
        Covariance matrices.

    References
    ----------
    .. [1] `A special form of SPD covariance matrix for interpretation and
        visualization of data manipulated with Riemannian geometry
        <https://hal.archives-ouvertes.fr/hal-01103344/>`_
        M. Congedo and A. Barachant, MaxEnt - 34th International Workshop on
        Bayesian Inference and Maximun Entropy Methods in Science and
        Engineering (MaxEnt'14), Sep 2014, Amboise, France. pp.495
    """
    if alpha <= 0:
        raise ValueError(
            'Parameter alpha must be strictly positive (Got %d)' % alpha)
    est = _check_est(estimator)
    n_matrices, n_channels, n_times = X.shape

    Hchannels = np.eye(n_channels) \
        - np.outer(np.ones(n_channels), np.ones(n_channels)) / n_channels
    Htimes = np.eye(n_times) \
        - np.outer(np.ones(n_times), np.ones(n_times)) / n_times
    X = Hchannels @ X @ Htimes  # Eq(8), double centering

    covmats = np.empty(
        (n_matrices, n_channels + n_times, n_channels + n_times))
    for i in range(n_matrices):
        Y = np.concatenate((
            np.concatenate((X[i], alpha * np.eye(n_channels)), axis=1),  # top
            np.concatenate((alpha * np.eye(n_times), X[i].T), axis=1)  # bottom
        ), axis=0)  # Eq(9)
        covmats[i] = est(Y)
    return covmats / (2 * alpha)  # Eq(10)


def block_covariances(X, blocks, estimator='cov'):
    """Compute block diagonal covariance.

    Calculates block diagonal matrices where each block is a covariance
    matrix of a subset of channels.
    Block sizes are passed as a list of integers and can vary. The sum
    of block sizes must equal the number of channels in X.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_times)
        Multi-channel time-series.
    blocks: list of int
        List of block sizes.
    estimator : {'cov', 'scm', 'lwf', 'oas', 'mcd', 'sch', 'corr'}, \
        default='scm'
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.

    Returns
    -------
    C : ndarray, shape (n_matrices, n_channels, n_channels)
        Block diagonal covariance matrices.
    """
    est = _check_est(estimator)
    n_matrices, n_channels, n_times = X.shape

    if np.sum(blocks) != n_channels:
        raise ValueError('Sum of individual block sizes '
                         'must match number of channels of X.')

    covmats = np.empty((n_matrices, n_channels, n_channels))
    for i in range(n_matrices):
        blockcov, idx_start = [], 0
        for j in blocks:
            blockcov.append(est(X[i, idx_start:idx_start+j, :]))
            idx_start += j
        covmats[i] = block_diag(*tuple(blockcov))

    return covmats


def eegtocov(sig, window=128, overlapp=0.5, padding=True, estimator='cov'):
    """Convert EEG signal to covariance using sliding window."""
    est = _check_est(estimator)
    X = []
    if padding:
        padd = np.zeros((int(window / 2), sig.shape[1]))
        sig = np.concatenate((padd, sig, padd), axis=0)

    n_times, n_channels = sig.shape
    jump = int(window * overlapp)
    ix = 0
    while (ix + window < n_times):
        X.append(est(sig[ix:ix + window, :].T))
        ix = ix + jump

    return np.array(X)


###############################################################################


def cross_spectrum(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None):
    """Compute the complex cross-spectral matrices of a real signal X.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Multi-channel time-series.
    window : int, default=128
        The length of the FFT window used for spectral estimation.
    overlap : float, default=0.75
        The percentage of overlap between window.
    fmin : float | None, default=None
        The minimal frequency to be returned.
    fmax : float | None, default=None
        The maximal frequency to be returned.
    fs : float | None, default=None
        The sampling frequency of the signal.

    Returns
    -------
    S : ndarray, shape (n_channels, n_channels, n_freqs)
        Cross-spectral matrices, for each frequency bin.
    freqs : ndarray, shape (n_freqs,)
        The frequencies associated to cross-spectra.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Cross-spectrum
    """
    window = int(window)
    if window < 1:
        raise ValueError('Value window must be a positive integer')
    if not 0 < overlap < 1:
        raise ValueError(
            'Value overlap must be included in (0, 1) (Got %d)' % overlap)

    n_channels, n_times = X.shape
    n_freqs = int(window / 2) + 1  # X real signal => compute half-spectrum
    step = int((1.0 - overlap) * window)
    n_windows = int((n_times - window) / step + 1)
    win = np.hanning(window)

    # FFT calculation on all windows
    shape = (n_channels, n_windows, window)
    strides = X.strides[:-1]+(step*X.strides[-1], X.strides[-1])
    Xs = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    fdata = np.fft.rfft(Xs * win, n=window).transpose(1, 0, 2)

    # adjust frequency range to specified range
    if fs is not None:
        if fmin is None:
            fmin = 0
        if fmax is None:
            fmax = fs / 2
        if fmax <= fmin:
            raise ValueError('Parameter fmax must be superior to fmin')
        if 2.0 * fmax > fs:  # check Nyquist-Shannon
            raise ValueError('Parameter fmax must be inferior to fs/2')
        f = np.arange(0, n_freqs, dtype=int) * float(fs / window)
        fix = (f >= fmin) & (f <= fmax)
        fdata = fdata[:, :, fix]
        freqs = f[fix]
    else:
        if fmin is not None:
            warnings.warn('Parameter fmin not used because fs is None')
        if fmax is not None:
            warnings.warn('Parameter fmax not used because fs is None')
        freqs = None

    n_freqs = fdata.shape[2]
    S = np.zeros((n_channels, n_channels, n_freqs), dtype=complex)
    for i in range(n_freqs):
        S[:, :, i] = fdata[:, :, i].conj().T @ fdata[:, :, i]
    S /= n_windows * np.linalg.norm(win)**2

    # normalization to respect Parseval's theorem with the half-spectrum
    # excepted DC bin (always), and Nyquist bin (when window is even)
    if window % 2:
        S[..., 1:] *= 2
    else:
        S[..., 1:-1] *= 2

    return S, freqs


def cospectrum(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None):
    """Compute co-spectral matrices, the real part of cross-spectra.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Multi-channel time-series.
    window : int, default=128
        The length of the FFT window used for spectral estimation.
    overlap : float, default=0.75
        The percentage of overlap between window.
    fmin : float | None, default=None
        The minimal frequency to be returned.
    fmax : float | None, default=None
        The maximal frequency to be returned.
    fs : float | None, default=None
        The sampling frequency of the signal.

    Returns
    -------
    S : ndarray, shape (n_channels, n_channels, n_freqs)
        Co-spectral matrices, for each frequency bin.
    freqs : ndarray, shape (n_freqs,)
        The frequencies associated to cospectra.
    """
    S, freqs = cross_spectrum(
        X=X,
        window=window,
        overlap=overlap,
        fmin=fmin,
        fmax=fmax,
        fs=fs)

    return S.real, freqs


def coherence(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None,
              coh='ordinary'):
    """Compute squared coherence.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Multi-channel time-series.
    window : int, default=128
        The length of the FFT window used for spectral estimation.
    overlap : float, default=0.75
        The percentage of overlap between window.
    fmin : float | None, default=None
        The minimal frequency to be returned.
    fmax : float | None, default=None
        The maximal frequency to be returned.
    fs : float | None, default=None
        The sampling frequency of the signal.
    coh : {'ordinary', 'instantaneous', 'lagged', 'imaginary'}, \
            default='ordinary'
        The coherence type, see :class:`pyriemann.estimation.Coherences`.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels, n_freqs)
        Squared coherence matrices, for each frequency bin.
    freqs : ndarray, shape (n_freqs,)
        The frequencies associated to coherence.
    """
    S, freqs = cross_spectrum(
        X,
        window=window,
        overlap=overlap,
        fmin=fmin,
        fmax=fmax,
        fs=fs)
    S2 = np.abs(S)**2  # squared cross-spectral modulus

    C = np.zeros_like(S2)
    f_inds = np.arange(0, C.shape[-1], dtype=int)

    # lagged coh not defined for DC and Nyquist bins, because S is real
    if coh == 'lagged':
        if freqs is None:
            f_inds = np.arange(1, C.shape[-1] - 1, dtype=int)
            warnings.warn('DC and Nyquist bins are not defined for lagged-'
                          'coherence: filled with zeros')
        else:
            f_inds_ = f_inds[(freqs > 0) & (freqs < fs / 2)]
            if not np.array_equal(f_inds_, f_inds):
                warnings.warn('DC and Nyquist bins are not defined for lagged-'
                              'coherence: filled with zeros')
            f_inds = f_inds_

    for f in f_inds:
        psd = np.sqrt(np.diag(S2[..., f]))
        psd_prod = np.outer(psd, psd)
        if coh == 'ordinary':
            C[..., f] = S2[..., f] / psd_prod
        elif coh == 'instantaneous':
            C[..., f] = (S[..., f].real)**2 / psd_prod
        elif coh == 'lagged':
            np.fill_diagonal(S[..., f].real, 0.)  # prevent div by zero on diag
            C[..., f] = (S[..., f].imag)**2 / (psd_prod - (S[..., f].real)**2)
        elif coh == 'imaginary':
            C[..., f] = (S[..., f].imag)**2 / psd_prod
        else:
            raise ValueError("'%s' is not a supported coherence" % coh)

    return C, freqs


###############################################################################


def normalize(X, norm):
    """Normalize a set of square matrices, using corr, trace or determinant.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray. Matrices must be
        invertible for determinant-normalization.

    norm : {"corr", "trace", "determinant"}
        The type of normalization:

        * 'corr': normalized matrices are correlation matrices, with values in
          [-1, 1] and diagonal values equal to 1;
        * 'trace': trace of normalized matrices is 1;
        * 'determinant': determinant of normalized matrices is +/- 1.

    Returns
    -------
    Xn : ndarray, shape (..., n, n)
        The set of normalized matrices, same dimensions as X.
    """
    if not is_square(X):
        raise ValueError('Matrices must be square')

    if norm == "corr":
        stddev = np.sqrt(np.abs(np.diagonal(X, axis1=-2, axis2=-1)))
        denom = np.expand_dims(stddev, axis=-2) * stddev[..., np.newaxis]
    elif norm == "trace":
        denom = np.trace(X, axis1=-2, axis2=-1)
    elif norm == "determinant":
        denom = np.abs(np.linalg.det(X)) ** (1 / X.shape[-1])
    else:
        raise ValueError("'%s' is not a supported normalization" % norm)

    denom = np.expand_dims(denom, axis=tuple(range(denom.ndim, X.ndim)))
    Xn = X / denom

    if norm == "corr":
        np.clip(Xn, -1, 1, out=Xn)

    return Xn


def get_nondiag_weight(X):
    """Compute non-diagonality weights of a set of square matrices.

    Compute non-diagonality weights of a set of square matrices, following
    Eq(B.1) in [1]_.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    weights : ndarray, shape (...,)
        The non-diagonality weights for matrices.

    References
    ----------
    .. [1] `On the blind source separation of human electroencephalogram by
        approximate joint diagonalization of second order statistics
        <https://hal.archives-ouvertes.fr/hal-00343628>`_
        M. Congedo, C. Gouy-Pailler, C. Jutten. Clinical Neurophysiology,
        Elsevier, 2008, 119 (12), pp.2677-2686.
    """
    if not is_square(X):
        raise ValueError('Matrices must be square')

    X2 = X**2
    # sum of squared diagonal elements
    denom = np.trace(X2, axis1=-2, axis2=-1)
    # sum of squared off-diagonal elements
    num = np.sum(X2, axis=(-2, -1)) - denom
    weights = (1.0 / (X.shape[-1] - 1)) * (num / denom)
    return weights
