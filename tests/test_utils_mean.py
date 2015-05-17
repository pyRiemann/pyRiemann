from numpy.testing import assert_array_almost_equal,assert_array_equal
import numpy as np

from pyriemann.utils.mean import (mean_riemann,mean_euclid,mean_logeuclid,mean_logdet,mean_ale,mean_covariance)

def generate_cov(Nt,Ne):
    """Generate a set of cavariances matrices for test purpose"""
    diags = 1.0+0.1*np.random.randn(Nt,Ne)
    covmats = np.empty((Nt,Ne,Ne))
    for i in range(Nt):
        covmats[i] = np.diag(diags[i])
    return covmats
    
def test_riemann_mean():
    """Test the riemannian mean"""
    covmats = generate_cov(100,3)
    C = mean_riemann(covmats)

def test_logeuclid_mean():
    """Test the logeuclidean mean"""
    covmats = generate_cov(100,3)
    C = mean_logeuclid(covmats)
    
def test_euclid_mean():
    """Test the euclidean mean"""
    covmats = generate_cov(100,3)
    C = mean_euclid(covmats)
    assert_array_almost_equal(C,covmats.mean(axis=0))

def test_logdet_mean():
    """Test the logdet mean"""
    covmats = generate_cov(100,3)
    C = mean_logdet(covmats)

#def test_ald_mean():
#    """Test the Ale mean"""
#    covmats = generate_cov(100,3)
#    C = mean_ale(covmats)
    
def test_mean_covariance_riemann():
    """Test mean_covariance for riemannian metric"""
    covmats = generate_cov(100,3)
    C = mean_covariance(covmats,metric='riemann')
    Ctrue = mean_riemann(covmats)
    assert_array_equal(C,Ctrue)

def test_mean_covariance_logdet():
    """Test mean_covariance for logdet metric"""
    covmats = generate_cov(100,3)
    C = mean_covariance(covmats,metric='logdet')
    Ctrue = mean_logdet(covmats)
    assert_array_equal(C,Ctrue)
    
def test_mean_covariance_logeuclid():
    """Test mean_covariance for logeuclid metric"""
    covmats = generate_cov(100,3)
    C = mean_covariance(covmats,metric='logeuclid')
    Ctrue = mean_logeuclid(covmats)
    assert_array_equal(C,Ctrue)
    
def test_mean_covariance_euclid():
    """Test mean_covariance for euclidean metric"""
    covmats = generate_cov(100,3)
    C = mean_covariance(covmats,metric='euclid')
    Ctrue = mean_euclid(covmats)
    assert_array_equal(C,Ctrue)