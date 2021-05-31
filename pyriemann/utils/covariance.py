import numpy
from sklearn.covariance import oas, ledoit_wolf, fast_mcd, empirical_covariance
import warnings

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
        'cov': numpy.cov,
        'scm': _scm,
        'lwf': _lwf,
        'oas': _oas,
        'mcd': _mcd,
        'corr': numpy.corrcoef
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
    """
    est = _check_est(estimator)
    Nt, Ne, Ns = X.shape
    covmats = numpy.zeros((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i, :, :] = est(X[i, :, :])
    return covmats


def covariances_EP(X, P, estimator='cov'):
    """Special form covariance matrix."""
    est = _check_est(estimator)
    Nt, Ne, Ns = X.shape
    Np, Ns = P.shape
    covmats = numpy.zeros((Nt, Ne + Np, Ne + Np))
    for i in range(Nt):
        covmats[i, :, :] = est(numpy.concatenate((P, X[i, :, :]), axis=0))
    return covmats


def eegtocov(sig, window=128, overlapp=0.5, padding=True, estimator='cov'):
    """Convert EEG signal to covariance using sliding window"""
    est = _check_est(estimator)
    X = []
    if padding:
        padd = numpy.zeros((int(window / 2), sig.shape[1]))
        sig = numpy.concatenate((padd, sig, padd), axis=0)

    Ns, Ne = sig.shape
    jump = int(window * overlapp)
    ix = 0
    while (ix + window < Ns):
        X.append(est(sig[ix:ix + window, :].T))
        ix = ix + jump

    return numpy.array(X)


###############################################################################


def cross_spectrum(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None):
    """Compute the complex cross-spectral matrices of a real signal X.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_samples)
        ndarray of trials.
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
    S : ndarray, shape (n_trials, n_channels, n_channels, n_freqs)
        ndarray of cross-spectral matrices for each trials and for each
        frequency bin.
    freqs : ndarray, shape (n_freqs,)
        The frequencies associated to cospectra.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Cross-spectrum
    """
    Ne, Ns = X.shape
    number_freqs = int(window / 2)

    step = int((1.0 - overlap) * window)
    step = max(1, step)

    number_windows = int((Ns - window) / step + 1)
    # pre-allocation of memory
    fdata = numpy.zeros((number_windows, Ne, number_freqs), dtype=complex)
    win = numpy.hanning(window)

    # Loop on all frequencies
    for window_ix in range(int(number_windows)):

        # time markers to select the data
        # marker of the beginning of the time window
        t1 = int(window_ix * step)
        # marker of the end of the time window
        t2 = int(t1 + window)
        # select current window and apodize it
        cdata = X[:, t1:t2] * win

        # FFT calculation
        fdata[window_ix, :, :] = numpy.fft.fft(
            cdata, n=window, axis=1)[:, 0:number_freqs]

    # adjust frequency range to specified range (in case it is a parameter)
    if fs is not None:
        if fmin is None:
            fmin = 0
        if fmax is None:
            fmax = fs / 2
        if fmax <= fmin:
            raise ValueError('Parameter fmax must be superior to fmin')
        if 2.0 * fmax > fs: # check Nyquist-Shannon
            raise ValueError('Parameter fmax must be inferior to fs/2')
        f = numpy.arange(0, 1, 1.0 / number_freqs) * (fs / 2.0)
        Fix = (f >= fmin) & (f <= fmax)
        fdata = fdata[:, :, Fix]
        freqs = f[Fix]
    else:
        if fmin is not None:
            warnings.warn('Parameter fmin not used because fs is None')
        if fmax is not None:
            warnings.warn('Parameter fmax not used because fs is None')
        freqs = None

    Nf = fdata.shape[2]
    S = numpy.zeros((Ne, Ne, Nf), dtype=complex)
    for i in range(Nf):
        S[:, :, i] = numpy.dot(fdata[:, :, i].conj().T, fdata[:, :, i])
    S /= number_windows * numpy.linalg.norm(win)**2

    return S, freqs


def cospectrum(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None):
    """Compute co-spectral matrices, the real part of cross-spectra.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_samples)
        ndarray of trials.
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
    S : ndarray, shape (n_trials, n_channels, n_channels, n_freqs)
        ndarray of co-spectral matrices for each trials and for each
        frequency bin.
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
    X : ndarray, shape (n_trials, n_channels, n_samples)
        ndarray of trials.
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
    C : ndarray, shape (n_trials, n_channels, n_channels, n_freqs)
        ndarray of coherence matrices for each trials and for each
        frequency bin.
    """
    S, _ = cross_spectrum(X, window, overlap, fmin, fmax, fs)
    S2 = numpy.abs(S)**2 # squared cross-spectral modulus
    C = numpy.zeros_like(S2)
    for f in range(S2.shape[-1]):
        psd = numpy.sqrt(numpy.diag(S2[..., f]))
        C[..., f] = S2[..., f] / numpy.outer(psd, psd)
    return C


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
        denom = numpy.trace(X, axis1=-2, axis2=-1)
    elif norm  == "determinant":
        denom = numpy.abs(numpy.linalg.det(X)) ** (1 / X.shape[-1])
    else:
        raise ValueError("'%s' is not a supported normalization" % norm)

    while denom.ndim != X.ndim:
        denom = denom[..., numpy.newaxis]
    Xn = X / denom
    return Xn


def get_nondiag_weight(X):
    """Compute non-diagonality weights for square matrices, Eq(B.1) in [1]_.

    Parameters
    ----------
    X : ndarray, shape (n_mats, n_channels, n_channels)
        The set of square matrices.

    Returns
    -------
    weights : ndarray, shape (n_mats,)
        The non-diagonality weights for matrices.

    References
    ----------
    .. [1] M. Congedo, C. Gouy-Pailler, C. Jutten, "On the blind source
        separation of human electroencephalogram by approximate joint
        diagonalization of second order statistics", Clin Neurophysiol, 2008
    """
    if X.ndim != 3:
        raise ValueError('Input must have 3 dimensions')
    if X.shape[-2] != X.shape[-1]:
        raise ValueError('Matrices must be square')

    # sum of squared diagonal elements
    denom = numpy.trace(X**2, axis1=-2, axis2=-1)
    # sum of squared off-diagonal elements
    num = numpy.sum(X**2, axis=(-2, -1)) - denom
    weights = ( 1.0 / (X.shape[-1] - 1) ) * (num / denom)
    return weights
