import numpy as np
import pytest
from pytest import approx
from pyriemann.utils.mean import (
    mean_riemann,
    mean_euclid,
    mean_logeuclid,
    mean_logdet,
    mean_ale,
    mean_identity,
    mean_covariance,
    mean_kullback_sym,
    mean_harmonic,
    mean_wasserstein,
    mean_alm,
)
from pyriemann.utils.geodesic import geodesic_riemann


def generate_cov(n_trials, n_channels):
    """Generate a set of cavariances matrices for test purpose"""
    rs = np.random.RandomState(1234)
    diags = 2.0 + 0.1 * rs.randn(n_trials, n_channels)
    A = 2 * rs.rand(n_channels, n_channels) - 1
    A /= np.linalg.norm(A, axis=1)[:, np.newaxis]
    covmats = np.empty((n_trials, n_channels, n_channels))
    for i in range(n_trials):
        covmats[i] = A @ np.diag(diags[i]) @ A.T
    return covmats, diags, A


@pytest.mark.parametrize(
    "mean",
    [
        mean_riemann,
        mean_logeuclid,
        mean_euclid,
        mean_identity,
        mean_logdet,
        mean_ale,
        mean_kullback_sym,
        mean_harmonic,
        mean_wasserstein,
    ],
)
def test_mean_shape(mean):
    """Test the shape of mean"""
    n_trials, n_channels = 5, 3
    covmats, _, A = generate_cov(n_trials, n_channels)
    C = mean(covmats)
    assert C.shape == (n_channels , n_channels)


@pytest.mark.parametrize("mean", [mean_riemann, mean_logdet])
def test_mean_shape_with_init(mean):
    """Test the shape of mean with init"""
    n_trials, n_channels = 5, 3
    covmats, _, A = generate_cov(n_trials, n_channels)
    C = mean(covmats, init=covmats[0])
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize("init", [True, False])
def test_riemann_mean(init):
    """Test the riemannian mean"""
    n_trials, n_channels = 100, 3
    covmats, diags, A = generate_cov(n_trials, n_channels)
    if init:
        C = mean_riemann(covmats, init=covmats[0])
    else:
        C = mean_riemann(covmats)
    Ctrue = np.exp(np.log(diags).mean(0))
    Ctrue = A @ np.diag(Ctrue) @ A.T
    assert C == approx(Ctrue)


def test_euclid_mean():
    """Test the euclidean mean"""
    n_trials, n_channels = 100, 3
    covmats, _, _ = generate_cov(n_trials, n_channels)
    C = mean_euclid(covmats)
    assert C == approx(covmats.mean(axis=0))


def test_identity_mean():
    """Test the identity mean"""
    n_trials, n_channels = 100, 3
    covmats, _, _ = generate_cov(n_trials, n_channels)
    C = mean_identity(covmats)
    assert np.all(C == np.eye(n_channels))


def test_alm_mean():
    """Test the ALM mean"""
    n_trials, n_channels = 5, 3
    covmats, _, _ = generate_cov(n_trials, n_channels)
    C_alm = mean_alm(covmats)
    C_riem = mean_riemann(covmats)
    assert C_alm == approx(C_riem)


def test_alm_mean_maxiter():
    """Test the ALM mean with max iteration"""
    n_trials, n_channels = 5, 3
    covmats, _, _ = generate_cov(n_trials, n_channels)
    C = mean_alm(covmats, maxiter=1, verbose=True)  # maxiter reached
    assert C.shape == (3, 3)


def test_alm_mean_2trials():
    """Test the ALM mean with 2 trials"""
    n_trials, n_channels = 2, 3
    covmats, _, _ = generate_cov(n_trials, n_channels)
    C = mean_alm(covmats)  # n_trials=2
    assert np.all(C == geodesic_riemann(covmats[0], covmats[1], alpha=0.5))


@pytest.mark.parametrize(
    "metric,mean",
    [
        ("riemann", mean_riemann),
        ("logdet", mean_logdet),
        ("logeuclid", mean_logeuclid),
        ("euclid", mean_euclid),
        ("alm", mean_alm),
    ],
)
def test_mean_covariance_metric(metric, mean):
    """Test mean_covariance for metric"""
    n_trials, n_channels = 5, 3
    covmats, _, _ = generate_cov(n_trials, n_channels)
    C = mean_covariance(covmats, metric=metric)
    Ctrue = mean(covmats)
    assert np.all(C == Ctrue)
