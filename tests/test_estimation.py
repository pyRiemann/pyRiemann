import numpy as np
from pyriemann.estimation import (Covariances, ERPCovariances,
                                  XdawnCovariances, CospCovariances,
                                  HankelCovariances, Coherences, Shrinkage)
import pytest


def test_covariances():
    """Test Covariances"""
    x = np.random.randn(2, 3, 100)
    cov = Covariances()
    cov.fit(x)
    cov.fit_transform(x)
    assert cov.get_params() == dict(estimator='scm')


def test_hankel_covariances():
    """Test Hankel Covariances"""
    x = np.random.randn(2, 3, 100)
    cov = HankelCovariances()
    cov.fit(x)
    cov.fit_transform(x)
    assert cov.get_params() == dict(estimator='scm', delays=4)

    cov = HankelCovariances(delays=[1, 2])
    cov.fit(x)
    cov.fit_transform(x)


def test_erp_covariances():
    """Test fit ERPCovariances"""
    x = np.random.randn(10, 3, 100)
    labels = np.array([0, 1]).repeat(5)
    cov = ERPCovariances()
    cov.fit_transform(x, labels)
    cov = ERPCovariances(classes=[0])
    cov.fit_transform(x, labels)
    # assert raise svd
    with pytest.raises(TypeError):
        ERPCovariances(svd='42')
    cov = ERPCovariances(svd=2)
    assert cov.get_params() == dict(classes=None, estimator='scm', svd=2)
    cov.fit_transform(x, labels)


def test_xdawn_covariances():
    """Test fit XdawnCovariances"""
    x = np.random.randn(10, 3, 100)
    labels = np.array([0, 1]).repeat(5)
    cov = XdawnCovariances()
    cov.fit_transform(x, labels)
    assert cov.get_params() == dict(nfilter=4, applyfilters=True,
                                    classes=None, estimator='scm',
                                    xdawn_estimator='scm',
                                    baseline_cov=None)


def test_cosp_covariances():
    """Test fit CospCovariances"""
    x = np.random.randn(2, 3, 1000)
    cov = CospCovariances()
    cov.fit(x)
    cov.fit_transform(x)
    assert cov.get_params() == dict(window=128, overlap=0.75, fmin=None,
                                    fmax=None, fs=None)

@pytest.mark.parametrize('coh', ['ordinary', 'instantaneous', 'lagged', 'imaginary'])
def test_coherences(coh):
    """Test fit Coherences"""
    rs = np.random.RandomState(42)
    n_trials, n_channels, n_times = 10, 3, 1000
    x = rs.randn(n_trials, n_channels, n_times)

    cov = Coherences(coh=coh)
    cov.fit(x)
    cov.fit_transform(x)
    assert cov.get_params() == dict(window=128, overlap=0.75, fmin=None,
                                    fmax=None, fs=None, coh=coh)


def test_shrinkage():
    """Test Shrinkage"""
    x = np.random.randn(2, 3, 100)
    cov = Covariances()
    covs = cov.fit_transform(x)
    sh = Shrinkage()
    sh.fit(covs)
    sh.transform(covs)
    assert sh.get_params() == dict(shrinkage=0.1)
