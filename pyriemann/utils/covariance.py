import numpy as np
from numpy import array, arange, zeros, concatenate, hanning
from numpy.fft import fft
from sklearn.covariance import oas, ledoit_wolf, fast_mcd, empirical_covariance
from matplotlib import mlab

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
            '%s is not an valid estimator ! Valid estimators are : %s or a callable function' %
            (est,
             (' , ').join(
                 estimators.keys())))
    return est


def covariances(X, estimator='cov'):
    est = _check_est(estimator)
    Nt, Ne, Ns = X.shape
    covmats = zeros((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i, :, :] = est(X[i, :, :])
    return covmats


def covariances_EP(X, P, estimator='cov'):
    est = _check_est(estimator)
    Nt, Ne, Ns = X.shape
    Np, Ns = P.shape
    covmats = zeros((Nt, Ne + Np, Ne + Np))
    for i in range(Nt):
        covmats[i, :, :] = est(concatenate((P, X[i, :, :]), axis=0))
    return covmats


def eegtocov(sig, window=128, overlapp=0.5, padding=True, estimator='cov'):
    est = _check_est(estimator)
    X = []
    if padding:
        padd = zeros((int(window / 2), sig.shape[1]))
        sig = concatenate((padd, sig, padd), axis=0)

    Ns, Ne = sig.shape
    jump = int(window * overlapp)
    ix = 0
    while (ix + window < Ns):
        X.append(est(sig[ix:ix + window, :].T))
        ix = ix + jump

    return array(X)


def coherence(X, nfft=256, fs=2, noverlap=0):
    """Compute coherence."""
    n_chan = X.shape[0]
    ij = []
    for i in range(n_chan):
        for j in range(i+1, n_chan):
            ij.append((i, j))
    Cxy, Phase, freqs = mlab.cohere_pairs(X, ij, NFFT=nfft, Fs=fs,
                                          noverlap=noverlap)
    coh = zeros((n_chan, n_chan, len(freqs)))
    for i in range(n_chan):
        coh[i, i] = 1
        for j in range(i+1, n_chan):
            coh[i, j] = coh[j, i] = Cxy[(i, j)]
    return coh


def cospectrum(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None,
               phase_correction=False):
    Ne, Ns = X.shape
    number_freqs = int(window / 2)

    step = int((1.0 - overlap) * window)
    step = max(1, step)

    number_windows = (Ns - window) / step + 1
    # pre-allocation of memory
    fdata = zeros((number_windows, Ne, number_freqs), dtype=complex)
    win = hanning(window)

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
        fdata[window_ix, :, :] = fft(cdata, n=window, axis=1)[:, 0:number_freqs]

    # Adjust Frequency range to specified range (in case it is a parameter)
    if fmin is not None:
        f = arange(0, 1, 1.0 / number_freqs) * (fs / 2.0)
        Fix = (f >= fmin) & (f <= fmax)
        fdata = fdata[:, :, Fix]

    # fdata = fdata.real
    Nf = fdata.shape[2]
    S = zeros((Ne, Ne, Nf), dtype=complex)

    for i in range(Nf):
        S[:, :, i] = fdata[:,:,i].conj().T.dot(fdata[:,:,i]) / number_windows

    return S
