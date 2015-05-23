from numpy.testing import assert_array_almost_equal,assert_array_equal
from nose.tools import assert_equal
import numpy as np

from pyriemann.utils.covariance import (covariances, covariances_EP,
                                        eegtocov, cospectrum)

def test_covariances():
    """Test covariance for multiple estimator"""
    x = np.random.randn(2, 3, 100)
    cov = covariances(x)
    cov = covariances(x, estimator='oas')
    cov = covariances(x, estimator='lwf')
    cov = covariances(x, estimator='scm')
    cov = covariances(x, estimator='corr')
    cov = covariances(x, estimator='mcd')

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
    x = np.random.randn(1000,3)
    cov = eegtocov(x)
    assert_equal(cov.shape[1],3)
    
def test_covariances_eegtocov():
    """Test eegtocov"""
    x = np.random.randn(3, 1000)
    cov = cospectrum(x)
    
    
    