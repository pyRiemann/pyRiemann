from numpy.testing import assert_array_almost_equal, assert_array_equal
import numpy as np

from pyriemann.utils.mean import (mean_riemann, mean_euclid, mean_logeuclid,
                                  mean_logdet, mean_ale, mean_identity,
                                  mean_covariance, mean_kullback_sym,
                                  mean_harmonic, mean_wasserstein, mean_alm)


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
    covmats, _, A = generate_cov(100, 3)
    C = mean_logeuclid(covmats)
    assert C.shape == (3, 3)


def test_euclid_mean():
    """Test the euclidean mean"""
    covmats, _, _ = generate_cov(100, 3)
    C = mean_euclid(covmats)
    assert C.shape == (3, 3)
    assert_array_almost_equal(C, covmats.mean(axis=0))


def test_identity_mean():
    """Test the logdet mean"""
    covmats, _, _ = generate_cov(100, 3)
    C = mean_identity(covmats)
    assert C.shape == (3, 3)
    assert_array_equal(C, np.eye(3))


def test_logdet_mean():
    """Test the logdet mean"""
    covmats, _, _ = generate_cov(100, 3)
    C = mean_logdet(covmats)
    assert C.shape == (3, 3)


def test_logdet_mean_with_init():
    """Test the logdet mean"""
    covmats, _, _ = generate_cov(100, 3)
    C = mean_logdet(covmats, init=covmats[0])
    assert C.shape == (3, 3)


def test_ald_mean():
    """Test the Ale mean"""
    covmats, _, _ = generate_cov(100, 3)
    C = mean_ale(covmats)
    assert C.shape == (3, 3)


def test_kullback_mean():
    """Test the kullback mean"""
    covmats, _, _ = generate_cov(100, 3)
    C = mean_kullback_sym(covmats)
    assert C.shape == (3, 3)


def test_harmonic_mean():
    """Test the harmonic mean"""
    covmats, _, _ = generate_cov(100, 3)
    C = mean_harmonic(covmats)
    assert C.shape == (3, 3)


def test_wasserstein_mean():
    """Test the wasserstein mean"""
    covmats, _, _ = generate_cov(100, 3)
    C = mean_wasserstein(covmats)
    assert C.shape == (3, 3)


def test_mean_covariance_riemann():
    """Test mean_covariance for riemannian metric"""
    covmats, _, _ = generate_cov(100, 3)
    C = mean_covariance(covmats, metric='riemann')
    Ctrue = mean_riemann(covmats)
    assert_array_equal(C, Ctrue)


def test_mean_covariance_logdet():
    """Test mean_covariance for logdet metric"""
    covmats, _, _ = generate_cov(100, 3)
    C = mean_covariance(covmats, metric='logdet')
    Ctrue = mean_logdet(covmats)
    assert_array_equal(C, Ctrue)


def test_mean_covariance_logeuclid():
    """Test mean_covariance for logeuclid metric"""
    covmats, _, _ = generate_cov(100, 3)
    C = mean_covariance(covmats, metric='logeuclid')
    Ctrue = mean_logeuclid(covmats)
    assert_array_equal(C, Ctrue)


def test_mean_covariance_euclid():
    """Test mean_covariance for euclidean metric"""
    covmats, diags, A = generate_cov(100, 3)
    C = mean_covariance(covmats, metric='euclid')
    Ctrue = mean_euclid(covmats)
    assert_array_equal(C, Ctrue)


def test_alm_mean():
    """Test the ALM mean"""
    covmats, _, _ = generate_cov(3, 3)
    C_alm = mean_alm(covmats)
    C_riem = mean_riemann(covmats)
    assert_array_almost_equal(C_alm, C_riem)

    covmats, _, _ = generate_cov(10, 8)
    mean_alm(covmats, maxiter=1, verbose=True) # maxiter reached

    covmats, _, _ = generate_cov(2, 8)
    mean_alm(covmats) # Nt=2


def test_mean_covariance_alm():
    """Test mean_covariance for ALM"""
    covmats, _, _ = generate_cov(3, 3)
    C = mean_covariance(covmats, metric='alm')
    Ctrue = mean_alm(covmats)
    assert_array_equal(C, Ctrue)


