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
def test_mean_shape(mean, get_covmats):
    """Test the shape of mean"""
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)
    C = mean(covmats)
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize("mean", [mean_riemann, mean_logdet])
def test_mean_shape_with_init(mean, get_covmats):
    """Test the shape of mean with init"""
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)
    C = mean(covmats, init=covmats[0])
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize("init", [True, False])
def test_riemann_mean(init, get_covmats_params):
    """Test the riemannian mean"""
    n_trials, n_channels = 100, 3
    covmats, diags, A = get_covmats_params(n_trials, n_channels)
    if init:
        C = mean_riemann(covmats, init=covmats[0])
    else:
        C = mean_riemann(covmats)
    Ctrue = np.exp(np.log(diags).mean(0))
    Ctrue = A @ np.diag(Ctrue) @ A.T
    assert C == approx(Ctrue)


def test_euclid_mean(get_covmats):
    """Test the euclidean mean"""
    n_trials, n_channels = 100, 3
    covmats = get_covmats(n_trials, n_channels)
    C = mean_euclid(covmats)
    assert C == approx(covmats.mean(axis=0))


def test_identity_mean(get_covmats):
    """Test the identity mean"""
    n_trials, n_channels = 100, 3
    covmats = get_covmats(n_trials, n_channels)
    C = mean_identity(covmats)
    assert np.all(C == np.eye(n_channels))


def test_alm_mean(get_covmats):
    """Test the ALM mean"""
    n_trials, n_channels = 3, 3
    covmats = get_covmats(n_trials, n_channels)
    C_alm = mean_alm(covmats)
    C_riem = mean_riemann(covmats)
    assert C_alm == approx(C_riem)


def test_alm_mean_maxiter(get_covmats):
    """Test the ALM mean with max iteration"""
    n_trials, n_channels = 3, 3
    covmats = get_covmats(n_trials, n_channels)
    C = mean_alm(covmats, maxiter=1)
    assert C.shape == (n_channels, n_channels)


def test_alm_mean_2trials(get_covmats):
    """Test the ALM mean with 2 trials"""
    n_trials, n_channels = 2, 3
    covmats = get_covmats(n_trials, n_channels)
    C = mean_alm(covmats)
    assert np.all(C == geodesic_riemann(covmats[0], covmats[1], alpha=0.5))


@pytest.mark.parametrize(
    "metric, mean",
    [
        ("riemann", mean_riemann),
        ("logdet", mean_logdet),
        ("logeuclid", mean_logeuclid),
        ("euclid", mean_euclid),
        ("alm", mean_alm),
        ("identity", mean_identity),
        ("wasserstein", mean_wasserstein),
        ("ale", mean_ale),
        ("harmonic", mean_harmonic),
        ("kullback_sym", mean_kullback_sym),
    ],
)
def test_mean_covariance_metric(metric, mean, get_covmats):
    """Test mean_covariance for metric"""
    n_trials, n_channels = 3, 3
    covmats = get_covmats(n_trials, n_channels)
    C = mean_covariance(covmats, metric=metric)
    Ctrue = mean(covmats)
    assert np.all(C == Ctrue)
