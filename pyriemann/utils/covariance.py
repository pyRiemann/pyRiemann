import numpy
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
    """Estimation of covariance matrix."""
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


def coherence(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None):
    """Compute coherence."""
    cosp = cospectrum(X, window, overlap, fmin, fmax, fs)
    coh = numpy.zeros_like(cosp)
    for f in range(cosp.shape[-1]):
        psd = numpy.sqrt(numpy.diag(cosp[..., f]))
        coh[..., f] = cosp[..., f] / numpy.outer(psd, psd)
    return coh


def cospectrum(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None):
    """Compute Cospectrum."""
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

    # Adjust Frequency range to specified range (in case it is a parameter)
    if fmin is not None:
        f = numpy.arange(0, 1, 1.0 / number_freqs) * (fs / 2.0)
        Fix = (f >= fmin) & (f <= fmax)
        fdata = fdata[:, :, Fix]

    # fdata = fdata.real
    Nf = fdata.shape[2]
    S = numpy.zeros((Ne, Ne, Nf), dtype=complex)
    normval = numpy.linalg.norm(win)**2
    for i in range(Nf):
        S[:, :, i] = numpy.dot(fdata[:, :, i].conj().T, fdata[:, :, i]) / (
            number_windows * normval)

    return numpy.abs(S)**2
