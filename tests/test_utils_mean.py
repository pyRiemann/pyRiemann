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
    mean_power,
    maskedmean_riemann,
    nanmean_riemann,
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
        nanmean_riemann,
        mean_power,
    ],
)
def test_mean_shape(mean, get_covmats):
    """Test the shape of mean"""
    n_matrices, n_channels = 5, 3
    covmats = get_covmats(n_matrices, n_channels)
    if mean == mean_power:
        C = mean(covmats, 0.42)
    else:
        C = mean(covmats)
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize("mean", [mean_riemann, mean_logdet])
def test_mean_shape_with_init(mean, get_covmats):
    """Test the shape of mean with init"""
    n_matrices, n_channels = 5, 3
    covmats = get_covmats(n_matrices, n_channels)
    if mean == mean_power:
        C = mean(covmats, 0.42, init=covmats[0])
    else:
        C = mean(covmats, init=covmats[0])
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize("init", [True, False])
def test_riemann_mean(init, get_covmats_params):
    """Test the riemannian mean"""
    n_matrices, n_channels = 100, 3
    covmats, diags, A = get_covmats_params(n_matrices, n_channels)
    if init:
        C = mean_riemann(covmats, init=covmats[0])
    else:
        C = mean_riemann(covmats)
    Ctrue = np.exp(np.log(diags).mean(0))
    Ctrue = A @ np.diag(Ctrue) @ A.T
    assert C == approx(Ctrue)


def test_euclid_mean(get_covmats):
    """Test the euclidean mean"""
    n_matrices, n_channels = 100, 3
    covmats = get_covmats(n_matrices, n_channels)
    C = mean_euclid(covmats)
    assert C == approx(covmats.mean(axis=0))


def test_identity_mean(get_covmats):
    """Test the identity mean"""
    n_matrices, n_channels = 100, 3
    covmats = get_covmats(n_matrices, n_channels)
    C = mean_identity(covmats)
    assert np.all(C == np.eye(n_channels))


def test_alm_mean(get_covmats):
    """Test the ALM mean"""
    n_matrices, n_channels = 3, 3
    covmats = get_covmats(n_matrices, n_channels)
    C_alm = mean_alm(covmats)
    C_riem = mean_riemann(covmats)
    assert C_alm == approx(C_riem)


def test_alm_mean_maxiter(get_covmats):
    """Test the ALM mean with max iteration"""
    n_matrices, n_channels = 3, 3
    covmats = get_covmats(n_matrices, n_channels)
    C = mean_alm(covmats, maxiter=1)
    assert C.shape == (n_channels, n_channels)


def test_alm_mean_2matrices(get_covmats):
    """Test the ALM mean with 2 matrices"""
    n_matrices, n_channels = 2, 3
    covmats = get_covmats(n_matrices, n_channels)
    C = mean_alm(covmats)
    assert np.all(C == geodesic_riemann(covmats[0], covmats[1], alpha=0.5))


def test_power_mean(get_covmats):
    """Test the power mean"""
    n_matrices, n_channels = 3, 3
    covmats = get_covmats(n_matrices, n_channels)
    C_power_1 = mean_power(covmats, 1)
    C_power_0 = mean_power(covmats, 0)
    C_power_m1 = mean_power(covmats, -1)
    C_arithm = mean_euclid(covmats)
    C_geom = mean_riemann(covmats)
    C_harm = mean_harmonic(covmats)
    assert C_power_1 == approx(C_arithm)
    assert C_power_0 == approx(C_geom)
    assert C_power_m1 == approx(C_harm)


@pytest.mark.parametrize("init", [True, False])
def test_riemann_mean_masked_shape(init, get_covmats, get_masks):
    """Test the masked riemann mean"""
    n_matrices, n_channels = 10, 4
    covmats = get_covmats(n_matrices, n_channels)
    masks = get_masks(n_matrices, n_channels)
    if init:
        C = maskedmean_riemann(covmats, masks, init=covmats[0])
    else:
        C = maskedmean_riemann(covmats, masks)
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize("init", [True, False])
def test_riemann_mean_nan_shape(init, get_covmats, rndstate):
    """Test the riemann nan mean shape"""
    n_matrices, n_channels = 50, 6
    covmats = get_covmats(n_matrices, n_channels)
    emean = np.mean(covmats, axis=0)
    for i in range(n_matrices):
        corrup_channels = rndstate.choice(
            np.arange(0, n_channels), size=n_channels // 2, replace=False)
        for j in corrup_channels:
            covmats[i, j] = np.nan
            covmats[i, :, j] = np.nan
    if init:
        C = nanmean_riemann(covmats, init=emean)
    else:
        C = nanmean_riemann(covmats)
    assert C.shape == (n_channels, n_channels)


def test_riemann_mean_nan_errors(get_covmats):
    """Test the riemann nan mean errors"""
    n_matrices, n_channels = 10, 4
    covmats = get_covmats(n_matrices, n_channels)

    with pytest.raises(ValueError):  # not symmetric NaN values
        covmats_ = covmats.copy()
        covmats_[0, 0] = np.nan  # corrup only a row, not its corresp column
        nanmean_riemann(covmats_)
    with pytest.raises(ValueError):  # not rows and columns NaN values
        covmats_ = covmats.copy()
        covmats_[1, 0, 1] = np.nan  # corrup an off-diagonal value
        nanmean_riemann(covmats_)


def test_power_mean_errors(get_covmats):
    """Test the power mean errors"""
    n_matrices, n_channels = 3, 2
    covmats = get_covmats(n_matrices, n_channels)

    with pytest.raises(ValueError):  # exponent is not a scalar
        mean_power(covmats, [1])
    with pytest.raises(ValueError):  # exponent is not in [-1,1]
        mean_power(covmats, 3)


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
    n_matrices, n_channels = 3, 3
    covmats = get_covmats(n_matrices, n_channels)
    C = mean_covariance(covmats, metric=metric)
    Ctrue = mean(covmats)
    assert np.all(C == Ctrue)
