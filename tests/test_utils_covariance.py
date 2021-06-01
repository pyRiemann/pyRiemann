from numpy.testing import assert_array_almost_equal, assert_array_equal
import numpy as np
from scipy.signal import coherence as coh_sp
import pytest

from pyriemann.utils.covariance import (covariances, covariances_EP, eegtocov,
                                        cross_spectrum, cospectrum, coherence,
                                        normalize, get_nondiag_weight)


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
    x = np.random.randn(3, 1000)
    cross_spectrum(x)
    cross_spectrum(x, fs=128, fmin=2, fmax=40)

    with pytest.raises(ValueError): # fmin > fmax
        cross_spectrum(x, fs=128, fmin=20, fmax=10)
    with pytest.raises(ValueError): # fmax > fs/2
        cross_spectrum(x, fs=128, fmin=20, fmax=65)
    with pytest.warns(UserWarning): # fs is None
        cross_spectrum(x, fmin=12)
    with pytest.warns(UserWarning): # fs is None
        cross_spectrum(x, fmax=12)

    c, _ = cross_spectrum(x, fs=128, window=256, fmin=3, fmax=51)
    # test if real part is symmetric
    assert_array_almost_equal(c.real, np.transpose(c.real, (1, 0, 2)), 6)
    # test if imag part is skew-symmetric
    assert_array_almost_equal(c.imag, -np.transpose(c.imag, (1, 0, 2)), 6)


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
    assert_array_almost_equal(coh[0, 1], coh2[:-1], 1)


def test_normalize():
    """Test normalize"""
    n_channels = 3
    cov = eegtocov(np.random.randn(1000, n_channels))

    covn = normalize(cov, "trace")
    assert_array_almost_equal(np.ones(covn.shape[0]),
                              np.trace(covn, axis1=-2, axis2=-1))
    covn = normalize(cov, "determinant")
    assert_array_almost_equal(np.ones(covn.shape[0]), np.linalg.det(covn))

    with pytest.raises(ValueError): # not square
        normalize(np.random.randn(10, n_channels, n_channels + 2), "trace")
    with pytest.raises(ValueError): # invalid normalization type
        normalize(cov, "abc")


def test_get_nondiag_weight():
    """Test get_nondiag_weight"""
    n_trials, n_channels = 20, 3
    mats = np.random.randn(n_trials, n_channels, n_channels)
    w = get_nondiag_weight(mats)
    assert len(w) == n_trials

    # 2x2 constant matrices => non-diag weights equal to 1
    mats = np.random.randn(n_trials, 1, 1) * np.ones((n_trials, 2, 2))
    w = get_nondiag_weight(mats)
    assert_array_almost_equal(w, np.ones(n_trials))

    # diagonal matrices => non-diag weights equal to 0
    mats = np.random.randn(n_trials, 1, 1) * ([np.eye(n_channels)] * n_trials)
    w = get_nondiag_weight(mats)
    assert_array_almost_equal(w, np.zeros(n_trials))

    with pytest.raises(ValueError): # not 3 dims
        get_nondiag_weight(np.random.randn(10, 10, n_channels, n_channels))
    with pytest.raises(ValueError): # not square
        get_nondiag_weight(np.random.randn(10, n_channels, n_channels + 2))
