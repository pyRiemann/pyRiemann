import numpy as np
from pyriemann.estimation import (Covariances, ERPCovariances,
                                  XdawnCovariances, CospCovariances,
                                  HankelCovariances)
from nose.tools import assert_raises, assert_equal


def test_covariances():
    """Test Covariances"""
    x = np.random.randn(2, 3, 100)
    cov = Covariances()
    cov.fit(x)
    cov.fit_transform(x)
    assert_equal(cov.get_params(), dict(estimator='scm'))


def test_Hankelcovariances():
    """Test Hankel Covariances"""
    x = np.random.randn(2, 3, 100)
    cov = HankelCovariances()
    cov.fit(x)
    cov.fit_transform(x)
    assert_equal(cov.get_params(), dict(estimator='scm', delays=4))

    cov = HankelCovariances(delays=[1, 2])
    cov.fit(x)
    cov.fit_transform(x)


def test_ERPcovariances():
    """Test fit ERPCovariances"""
    x = np.random.randn(10, 3, 100)
    labels = np.array([0, 1]).repeat(5)
    cov = ERPCovariances()
    cov.fit_transform(x, labels)
    cov = ERPCovariances(classes=[0])
    cov.fit_transform(x, labels)
    # assert raise svd
    assert_raises(TypeError, ERPCovariances, svd='42')
    cov = ERPCovariances(svd=2)
    assert_equal(cov.get_params(), dict(classes=None, estimator='scm',
                                        svd=2))
    cov.fit_transform(x, labels)



def test_Xdawncovariances():
    """Test fit ERPCovariances"""
    x = np.random.randn(10, 3, 100)
    labels = np.array([0, 1]).repeat(5)
    cov = XdawnCovariances()
    cov.fit_transform(x, labels)
    assert_equal(cov.get_params(), dict(nfilter=4, applyfilters=True,
                                        classes=None, estimator='scm',
                                        xdawn_estimator='scm',
                                        baseline_cov=None))


def test_Cospcovariances():
    """Test fit CospCovariances"""
    x = np.random.randn(2, 3, 1000)
    cov = CospCovariances()
    cov.fit(x)
    cov.fit_transform(x)
    assert_equal(cov.get_params(), dict(window=128, overlap=0.75, fmin=None,
                                        fmax=None, fs=None))
