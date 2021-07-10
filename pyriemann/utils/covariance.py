import warnings

import numpy as np

from sklearn.covariance import oas, ledoit_wolf, fast_mcd, empirical_covariance

# Mapping different estimator on the sklearn toolbox


def _lwf(X):
    """Wrapper for sklearn ledoit wolf covariance estimator"""
    C, _ = ledoit_wolf(X.T)
    return C


def _oas(X):
    """Wrapper for sklearn oas covariance estimator"""
    C, _ = oas(X.T)
    return C


def _scm(X):
    """Wrapper for sklearn sample covariance estimator"""
    return empirical_covariance(X.T)


def _mcd(X):
    """Wrapper for sklearn mcd covariance estimator"""
    _, C, _, _ = fast_mcd(X.T)
    return C


def _check_est(est):
    """Check if a given estimator is valid"""

    # Check estimator exist and return the correct function
    estimators = {
        'cov': np.cov,
        'scm': _scm,
        'lwf': _lwf,
        'oas': _oas,
        'mcd': _mcd,
        'corr': np.corrcoef
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
    X : ndarray, shape (n_trials, n_channels, n_samples)
        ndarray of trials.

    estimator : {'cov', 'scm', 'lwf', 'oas', 'mcd', 'corr'} (default: 'scm')
        covariance matrix estimator:

        * 'cov' for numpy based covariance matrix, https://numpy.org/doc/stable/reference/generated/numpy.cov.html
        * 'scm' for sample covariance matrix, https://scikit-learn.org/stable/modules/generated/sklearn.covariance.empirical_covariance.html
        * 'lwf' for shrunk Ledoit-Wolf covariance matrix, https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html
        * 'oas' for oracle approximating shrunk covariance matrix, https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html
        * 'mcd' for minimum covariance determinant matrix, https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html
        * 'corr' for correlation coefficient matrix, https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html

    Returns
    -------
    covmats : ndarray, shape (n_trials, n_channels, n_channels)
        ndarray of covariance matrices.

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/covariance.html
    """  # noqa
    est = _check_est(estimator)
    Nt, Ne, Ns = X.shape
    covmats = np.zeros((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i, :, :] = est(X[i, :, :])
    return covmats


def covariances_EP(X, P, estimator='cov'):
    """Special form covariance matrix."""
    est = _check_est(estimator)
    Nt, Ne, Ns = X.shape
    Np, Ns = P.shape
    covmats = np.zeros((Nt, Ne + Np, Ne + Np))
    for i in range(Nt):
        covmats[i, :, :] = est(np.concatenate((P, X[i, :, :]), axis=0))
    return covmats


def eegtocov(sig, window=128, overlapp=0.5, padding=True, estimator='cov'):
    """Convert EEG signal to covariance using sliding window"""
    est = _check_est(estimator)
    X = []
    if padding:
        padd = np.zeros((int(window / 2), sig.shape[1]))
        sig = np.concatenate((padd, sig, padd), axis=0)

    Ns, Ne = sig.shape
    jump = int(window * overlapp)
    ix = 0
    while (ix + window < Ns):
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
    window : int (default 128)
        The length of the FFT window used for spectral estimation.
    overlap : float (default 0.75)
        The percentage of overlap between window.
    fmin : float | None, (default None)
        The minimal frequency to be returned.
    fmax : float | None, (default None)
        The maximal frequency to be returned.
    fs : float | None, (default None)
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
    X : ndarray, shape (n_channels, n_samples)
        Multi-channel time-series.
    window : int (default 128)
        The length of the FFT window used for spectral estimation.
    overlap : float (default 0.75)
        The percentage of overlap between window.
    fmin : float | None, (default None)
        The minimal frequency to be returned.
    fmax : float | None, (default None)
        The maximal frequency to be returned.
    fs : float | None, (default None)
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


def coherence(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None):
    """Compute coherence.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_samples)
        Multi-channel time-series.
    window : int (default 128)
        The length of the FFT window used for spectral estimation.
    overlap : float (default 0.75)
        The percentage of overlap between window.
    fmin : float | None, (default None)
        The minimal frequency to be returned.
    fmax : float | None, (default None)
        The maximal frequency to be returned.
    fs : float | None, (default None)
        The sampling frequency of the signal.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels, n_freqs)
        Coherence matrices, for each frequency bin.
    freqs : ndarray, shape (n_freqs,)
        The frequencies associated to coherence.
    """
    S, freqs = cross_spectrum(X, window, overlap, fmin, fmax, fs)
    S2 = np.abs(S)**2  # squared cross-spectral modulus
    C = np.zeros_like(S2)
    for f in range(S2.shape[-1]):
        psd = np.sqrt(np.diag(S2[..., f]))
        C[..., f] = S2[..., f] / np.outer(psd, psd)
    return C, freqs


###############################################################################


def normalize(X, norm):
    """Normalize a set of square matrices, using trace or determinant.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray. Matrices must be
        invertible for determinant-normalization.

    norm : {"trace", "determinant"}
        The type of normalization.

    Returns
    -------
    Xn : ndarray, shape (..., n, n)
        The set of normalized matrices, same dimensions as X.
    """
    if X.ndim < 2:
        raise ValueError('Input must have at least 2 dimensions')
    if X.shape[-2] != X.shape[-1]:
        raise ValueError('Matrices must be square')

    if norm == "trace":
        denom = np.trace(X, axis1=-2, axis2=-1)
    elif norm == "determinant":
        denom = np.abs(np.linalg.det(X)) ** (1 / X.shape[-1])
    else:
        raise ValueError("'%s' is not a supported normalization" % norm)

    while denom.ndim != X.ndim:
        denom = denom[..., np.newaxis]
    Xn = X / denom
    return Xn


def get_nondiag_weight(X):
    """Compute non-diagonality weights of a set of square matrices, following
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
    .. [1] M. Congedo, C. Gouy-Pailler, C. Jutten, "On the blind source
        separation of human electroencephalogram by approximate joint
        diagonalization of second order statistics", Clin Neurophysiol, 2008
    """
    if X.ndim < 2:
        raise ValueError('Input must have at least 2 dimensions')
    if X.shape[-2] != X.shape[-1]:
        raise ValueError('Matrices must be square')

    X2 = X**2
    # sum of squared diagonal elements
    denom = np.trace(X2, axis1=-2, axis2=-1)
    # sum of squared off-diagonal elements
    num = np.sum(X2, axis=(-2, -1)) - denom
    weights = (1.0 / (X.shape[-1] - 1)) * (num / denom)
    return weights
