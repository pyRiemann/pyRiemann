import numpy as np
from scipy.stats import gmean
import pytest
from pytest import approx

from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.mean import (
    mean_covariance,
    mean_ale,
    mean_alm,
    mean_euclid,
    mean_harmonic,
    mean_identity,
    mean_kullback_sym,
    mean_logdet,
    mean_logeuclid,
    mean_power,
    mean_riemann,
    mean_wasserstein,
    maskedmean_riemann,
    nanmean_riemann,
)


@pytest.mark.parametrize(
    "mean",
    [
        mean_ale,
        mean_euclid,
        mean_harmonic,
        mean_identity,
        mean_kullback_sym,
        mean_logdet,
        mean_logeuclid,
        mean_power,
        mean_riemann,
        mean_wasserstein,
        nanmean_riemann,
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


@pytest.mark.parametrize(
    "mean", [mean_logdet, mean_riemann, mean_wasserstein, nanmean_riemann]
)
def test_mean_shape_with_init(mean, get_covmats):
    """Test the shape of mean with init"""
    n_matrices, n_channels = 5, 3
    covmats = get_covmats(n_matrices, n_channels)
    C = mean(covmats, init=covmats[0])
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize(
    "mean",
    [
        mean_euclid,
        mean_harmonic,
        mean_kullback_sym,
        mean_logdet,
        mean_logeuclid,
        mean_riemann,
        mean_wasserstein,
        nanmean_riemann,
    ],
)
def test_mean_weight_zero(mean, get_covmats):
    """Setting one weight to almost 0 it's almost like not passing the mat"""
    n_matrices, n_channels, w_val = 5, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    w = w_val * np.ones(n_matrices)
    C = mean(covmats[1:], sample_weight=w[1:])
    w[0] = 1e-12
    Cw = mean(covmats, sample_weight=w)
    assert C == approx(Cw, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize(
    "mean",
    [
        mean_euclid,
        mean_harmonic,
        mean_kullback_sym,
        mean_logdet,
        mean_logeuclid,
        mean_riemann,
        mean_wasserstein,
        nanmean_riemann,
    ],
)
def test_mean_weight_len_error(mean, get_covmats):
    n_matrices, n_channels = 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    with pytest.raises(ValueError):
        mean(covmats, sample_weight=np.ones(n_matrices + 1))


@pytest.mark.parametrize(
    "mean", [
        mean_ale,
        mean_alm,
        mean_logdet,
        mean_power,
        mean_riemann,
        mean_wasserstein,
        nanmean_riemann
    ]
)
def test_mean_warning_convergence(mean, get_covmats):
    """Test warning for convergence not reached """
    n_matrices, n_channels = 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    with pytest.warns(UserWarning):
        if mean == mean_power:
            mean(covmats, 0.3, maxiter=0)
        else:
            mean(covmats, maxiter=0)


@pytest.mark.parametrize(
    "mean",
    [
        mean_ale,
        mean_euclid,
        mean_harmonic,
        mean_identity,
        mean_kullback_sym,
        mean_logdet,
        mean_logeuclid,
        mean_power,
        mean_riemann,
        mean_wasserstein,
    ],
)
def test_mean_of_means(mean, get_covmats):
    """Test mean of submeans equal to grand mean"""
    n_matrices, n_channels = 10, 3
    covmats = get_covmats(n_matrices, n_channels)
    if mean == mean_power:
        C = mean(covmats, 0.42)
        C1 = mean(covmats[:n_matrices//2], 0.42)
        C2 = mean(covmats[n_matrices//2:], 0.42)
        C3 = mean(np.array([C1, C2]), 0.42)
    else:
        C = mean(covmats)
        C1 = mean(covmats[:n_matrices//2])
        C2 = mean(covmats[n_matrices//2:])
        C3 = mean(np.array([C1, C2]))
    assert C3 == approx(C, 6)


def test_alm_mean(get_covmats):
    """Test the ALM mean"""
    n_matrices, n_channels = 3, 3
    covmats = get_covmats(n_matrices, n_channels)
    C_alm = mean_alm(covmats)
    assert C_alm.shape == (n_channels, n_channels)
    C_riem = mean_riemann(covmats)
    assert C_alm == approx(C_riem)


def test_alm_mean_2matrices(get_covmats):
    """Test the ALM mean with 2 matrices"""
    n_matrices, n_channels = 2, 3
    covmats = get_covmats(n_matrices, n_channels)
    C = mean_alm(covmats)
    assert np.all(C == geodesic_riemann(covmats[0], covmats[1], alpha=0.5))


def test_euclid_mean(get_covmats):
    """Test the euclidean mean"""
    n_matrices, n_channels = 10, 3
    covmats = get_covmats(n_matrices, n_channels)
    C = mean_euclid(covmats)
    assert C == approx(covmats.mean(axis=0))


def test_identity_mean(get_covmats):
    """Test the identity mean"""
    n_matrices, n_channels = 10, 3
    covmats = get_covmats(n_matrices, n_channels)
    C = mean_identity(covmats)
    assert np.all(C == np.eye(n_channels))


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


def test_power_mean_errors(get_covmats):
    """Test the power mean errors"""
    n_matrices, n_channels = 3, 2
    covmats = get_covmats(n_matrices, n_channels)

    with pytest.raises(ValueError):  # exponent is not a scalar
        mean_power(covmats, [1])
    with pytest.raises(ValueError):  # exponent is not in [-1,1]
        mean_power(covmats, 3)


@pytest.mark.parametrize("init", [True, False])
def test_riemann_mean(init, get_covmats_params):
    """Test the riemannian mean"""
    n_matrices, n_channels = 100, 3
    covmats, evals, evecs = get_covmats_params(n_matrices, n_channels)
    if init:
        C = mean_riemann(covmats, init=covmats[0])
    else:
        C = mean_riemann(covmats)
    Ctrue = np.exp(np.log(evals).mean(0))
    Ctrue = evecs @ np.diag(Ctrue) @ evecs.T
    assert C == approx(Ctrue)


def test_riemann_mean_properties(get_covmats):
    n_matrices, n_channels = 5, 3
    covmats = get_covmats(n_matrices, n_channels)
    C = mean_riemann(covmats)

    # congruence-invariance, P2 in [Moakher2005] or P6 in [Nakamura2009]
    W = np.random.normal(size=(n_channels, n_channels))  # must be invertible
    assert W @ C @ W.T == approx(mean_riemann(W @ covmats @ W.T))

    # self-duality, P3 in [Moakher2005] or P8 in [Nakamura2009]
    assert C == approx(np.linalg.inv(mean_riemann(np.linalg.inv(covmats))))

    # determinant identity, P9 in [Nakamura2009]
    assert np.linalg.det(C) == approx(gmean(np.linalg.det(covmats)))


@pytest.mark.parametrize("init", [True, False])
def test_riemann_mean_masked_shape(init, get_covmats, get_masks):
    """Test the masked riemann mean"""
    n_matrices, n_channels = 5, 3
    covmats = get_covmats(n_matrices, n_channels)
    masks = get_masks(n_matrices, n_channels)
    if init:
        C = maskedmean_riemann(covmats, masks, tol=10e-3, init=covmats[0])
    else:
        C = maskedmean_riemann(covmats, masks, tol=10e-3)
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize("init", [True, False])
def test_riemann_mean_nan_shape(init, get_covmats, rndstate):
    """Test the riemann nan mean shape"""
    n_matrices, n_channels = 10, 6
    covmats = get_covmats(n_matrices, n_channels)
    emean = np.mean(covmats, axis=0)
    for i in range(n_matrices):
        corrup_channels = rndstate.choice(
            np.arange(0, n_channels), size=n_channels // 3, replace=False)
        for j in corrup_channels:
            covmats[i, j] = np.nan
            covmats[i, :, j] = np.nan
    if init:
        C = nanmean_riemann(covmats, tol=10e-3, init=emean)
    else:
        C = nanmean_riemann(covmats, tol=10e-3)
    assert C.shape == (n_channels, n_channels)


def test_riemann_mean_nan_errors(get_covmats):
    """Test the riemann nan mean errors"""
    n_matrices, n_channels = 5, 4
    covmats = get_covmats(n_matrices, n_channels)

    with pytest.raises(ValueError):  # not symmetric NaN values
        covmats_ = covmats.copy()
        covmats_[0, 0] = np.nan  # corrup only a row, not its corresp column
        nanmean_riemann(covmats_)
    with pytest.raises(ValueError):  # not rows and columns NaN values
        covmats_ = covmats.copy()
        covmats_[1, 0, 1] = np.nan  # corrup an off-diagonal value
        nanmean_riemann(covmats_)


def callable_np_average(X, sample_weight=None):
    return np.average(X, axis=0, weights=sample_weight)


@pytest.mark.parametrize(
    "metric, mean",
    [
        ("ale", mean_ale),
        ("alm", mean_alm),
        ("euclid", mean_euclid),
        ("harmonic", mean_harmonic),
        ("identity", mean_identity),
        ("kullback_sym", mean_kullback_sym),
        ("logdet", mean_logdet),
        ("logeuclid", mean_logeuclid),
        ("riemann", mean_riemann),
        ("wasserstein", mean_wasserstein),
        (callable_np_average, mean_euclid),
    ],
)
def test_mean_covariance_metric(metric, mean, get_covmats):
    """Test mean_covariance for metric"""
    n_matrices, n_channels = 3, 3
    covmats = get_covmats(n_matrices, n_channels)
    C = mean_covariance(covmats, metric=metric)
    Ctrue = mean(covmats)
    assert np.all(C == Ctrue)
