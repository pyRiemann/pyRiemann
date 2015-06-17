import numpy as np
from pyriemann.estimation import Covariances, ERPCovariances, XdawnCovariances, CospCovariances
from nose.tools import assert_raises

def test_covariances():
    """Test fit Covariances"""
    x = np.random.randn(2,3,100)
    cov = Covariances()
    cov.fit(x)
    cov.fit_transform(x)

def test_ERPcovariances():
    """Test fit ERPCovariances"""
    x = np.random.randn(10,3,100)
    labels = np.array([0,1]).repeat(5)
    cov = ERPCovariances()
    cov.fit_transform(x,labels)
    cov = ERPCovariances(classes=[0])
    cov.fit_transform(x,labels)
    # assert raise svd
    assert_raises(TypeError,ERPCovariances,svd='42')
    cov = ERPCovariances(svd=1)


def test_Xdawncovariances():
    """Test fit ERPCovariances"""
    x = np.random.randn(10,3,100)
    labels = np.array([0,1]).repeat(5)
    cov = XdawnCovariances()
    cov.fit_transform(x,labels)

def test_Cospcovariances():
    """Test fit CospCovariances"""
    x = np.random.randn(2,3,1000)
    cov = CospCovariances()
    cov.fit(x)
    cov.fit_transform(x)
