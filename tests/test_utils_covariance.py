from numpy.testing import assert_array_almost_equal, assert_array_equal
import numpy as np
from scipy.signal import coherence as coh_sp
import pytest

from pyriemann.utils.covariance import (covariances, covariances_EP, eegtocov,
                                        cross_spectrum, cospectrum, coherence,
                                        normalize_trace)


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
    """Test cross_spectrum, symmetric real part and skew-symmetric imag part"""
    x = np.random.randn(3, 1000)
    c, f = cross_spectrum(x)
    assert_array_almost_equal(c.real, np.transpose(c.real, (1, 0, 2)), 6)
    assert_array_almost_equal(c.imag, -np.transpose(c.imag, (1, 0, 2)), 6)
    c, f = cross_spectrum(x, fs=128, fmin=2, fmax=40)
    assert_array_almost_equal(c.real, np.transpose(c.real, (1, 0, 2)), 6)
    assert_array_almost_equal(c.imag, -np.transpose(c.imag, (1, 0, 2)), 6)
    assert_raises(ValueError, cross_spectrum, x, fmin=10, fmax=5, fs=32)
    assert_raises(ValueError, cross_spectrum, x, fmin=5, fmax=10, fs=16)


def test_covariances_cospectrum():
    """Test cospectrum, symmetric real part"""
    x = np.random.randn(3, 1000)
    c, f = cospectrum(x)
    assert_array_almost_equal(c, np.transpose(c, (1, 0, 2)), 6)
    c, f = cospectrum(x, fs=128, fmin=2, fmax=40)
    assert_array_almost_equal(c, np.transpose(c, (1, 0, 2)), 6)


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
    assert_array_almost_equal(coh[0, 1], coh2[:-1], 2)


def test_normalize_trace():
    """Test normalize_trace"""
    n_channels = 3
    x = np.random.randn(1000, n_channels)
    cov = eegtocov(x)
    covn = normalize_trace(cov)
    assert_array_equal(np.ones(covn.shape[0]),
                       np.trace(covn, axis1=-2, axis2=-1))

    assert_raises(ValueError, normalize_trace,
        np.random.randn(10, n_channels, n_channels + 2)) # not square
