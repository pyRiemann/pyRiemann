from numpy.testing import assert_array_almost_equal, assert_array_equal
import numpy as np
from scipy.signal import coherence as coh_sp
import pytest

from pyriemann.utils.covariance import (covariances, covariances_EP, eegtocov,
                                        cross_spectrum, cospectrum, coherence)


def test_covariances():
    """Test covariance for multiple estimator"""
    x = np.random.randn(2, 3, 100)
    cov = covariances(x)
    cov = covariances(x, estimator='oas')
    cov = covariances(x, estimator='lwf')
    cov = covariances(x, estimator='scm')
    cov = covariances(x, estimator='corr')
    cov = covariances(x, estimator='mcd')
    cov = covariances(x, estimator=np.cov)
    
    with pytest.raises(ValueError):
        covariances(x, estimator='truc')


def test_covariances_EP():
    """Test covariance_EP for multiple estimator"""
    x = np.random.randn(2, 3, 100)
    p = np.random.randn(3, 100)
    cov = covariances_EP(x, p)
    cov = covariances_EP(x, p, estimator='oas')
    cov = covariances_EP(x, p, estimator='lwf')
    cov = covariances_EP(x, p, estimator='scm')
    cov = covariances_EP(x, p, estimator='corr')
    cov = covariances_EP(x, p, estimator='mcd')


def test_covariances_eegtocov():
    """Test eegtocov"""
    x = np.random.randn(1000, 3)
    cov = eegtocov(x)
    assert cov.shape[1] == 3


def test_covariances_cross_spectrum():
    """Test cross_spectrum"""
    x = np.random.randn(3, 1000)
    cross_spectrum(x)
    cross_spectrum(x, fs=128, fmin=2, fmax=40)


def test_covariances_cospectrum():
    """Test cospectrum"""
    x = np.random.randn(3, 1000)
    cospectrum(x)
    cospectrum(x, fs=128, fmin=2, fmax=40)


def test_covariances_coherence():
    """Test coherence"""
    x = np.random.randn(2, 2048)
    coh = coherence(x, fs=128, window=256)

    _, coh2 = coh_sp(
        x[0],
        x[1],
        fs=128,
        nperseg=256,
        noverlap=int(0.75 * 256),
        window='hanning',
        detrend=False)
    assert_array_almost_equal(coh[0, 1], coh2[:-1], 0.1)
