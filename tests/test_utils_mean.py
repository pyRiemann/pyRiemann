from numpy.testing import assert_array_almost_equal, assert_array_equal
import numpy as np

from pyriemann.utils.mean import (mean_riemann, mean_euclid, mean_logeuclid,
                                  mean_logdet, mean_ale, mean_identity,
                                  mean_covariance, mean_kullback_sym,
                                  mean_harmonic, mean_wasserstein)


def generate_cov(Nt, Ne):
    """Generate a set of cavariances matrices for test purpose"""
    rs = np.random.RandomState(1234)
    diags = 2.0 + 0.1 * rs.randn(Nt, Ne)
    A = 2*rs.rand(Ne, Ne) - 1
    A /= np.atleast_2d(np.sqrt(np.sum(A**2, 1))).T
    covmats = np.empty((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.dot(np.dot(A, np.diag(diags[i])), A.T)
    return covmats, diags, A


def test_riemann_mean():
    """Test the riemannian mean"""
    covmats, diags, A = generate_cov(100, 3)
    Ctrue = np.exp(np.log(diags).mean(0))
    Ctrue = np.dot(np.dot(A, np.diag(Ctrue)), A.T)
    C = mean_riemann(covmats)
    assert_array_almost_equal(C, Ctrue)


def test_riemann_mean_with_init():
    """Test the riemannian mean with init"""
    covmats, diags, A = generate_cov(100, 3)
    C = mean_riemann(covmats, init=covmats[0])
    Ctrue = np.exp(np.log(diags).mean(0))
    Ctrue = np.dot(np.dot(A, np.diag(Ctrue)), A.T)
    assert_array_almost_equal(C, Ctrue)


def test_logeuclid_mean():
    """Test the logeuclidean mean"""
    covmats, diags, A = generate_cov(100, 3)
    C = mean_logeuclid(covmats)


def test_euclid_mean():
    """Test the euclidean mean"""
    covmats, diags, A = generate_cov(100, 3)
    C = mean_euclid(covmats)
    assert_array_almost_equal(C, covmats.mean(axis=0))


def test_identity_mean():
    """Test the logdet mean"""
    covmats, diags, A = generate_cov(100, 3)
    C = mean_identity(covmats)
    assert_array_equal(C, np.eye(3))


def test_logdet_mean():
    """Test the logdet mean"""
    covmats, diags, A = generate_cov(100, 3)
    C = mean_logdet(covmats)


def test_logdet_mean_with_init():
    """Test the logdet mean"""
    covmats, diags, A = generate_cov(100, 3)
    C = mean_logdet(covmats, init=covmats[0])


def test_ald_mean():
    """Test the Ale mean"""
    covmats, diags, A = generate_cov(100, 3)
    C = mean_ale(covmats)


def test_kullback_mean():
    """Test the kullback mean"""
    covmats, diags, A = generate_cov(100, 3)
    C = mean_kullback_sym(covmats)


def test_harmonic_mean():
    """Test the harmonic mean"""
    covmats, diags, A = generate_cov(100, 3)
    C = mean_harmonic(covmats)


def test_wasserstein_mean():
    """Test the wasserstein mean"""
    covmats, diags, A = generate_cov(100, 3)
    C = mean_wasserstein(covmats)


def test_mean_covariance_riemann():
    """Test mean_covariance for riemannian metric"""
    covmats, diags, A = generate_cov(100, 3)
    C = mean_covariance(covmats, metric='riemann')
    Ctrue = mean_riemann(covmats)
    assert_array_equal(C, Ctrue)


def test_mean_covariance_logdet():
    """Test mean_covariance for logdet metric"""
    covmats, diags, A = generate_cov(100, 3)
    C = mean_covariance(covmats, metric='logdet')
    Ctrue = mean_logdet(covmats)
    assert_array_equal(C, Ctrue)


def test_mean_covariance_logeuclid():
    """Test mean_covariance for logeuclid metric"""
    covmats, diags, A = generate_cov(100, 3)
    C = mean_covariance(covmats, metric='logeuclid')
    Ctrue = mean_logeuclid(covmats)
    assert_array_equal(C, Ctrue)


def test_mean_covariance_euclid():
    """Test mean_covariance for euclidean metric"""
    covmats, diags, A = generate_cov(100, 3)
    C = mean_covariance(covmats, metric='euclid')
    Ctrue = mean_euclid(covmats)
    assert_array_equal(C, Ctrue)
