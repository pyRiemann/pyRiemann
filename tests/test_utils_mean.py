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


@pytest.mark.parametrize("kind", ["spd", "hpd"])
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
def test_mean_shape(kind, mean, get_mats):
    """Test the shape of mean"""
    n_matrices, n_channels = 5, 3
    mats = get_mats(n_matrices, n_channels, kind)
    if mean is mean_ale and kind == "hpd":
        pytest.skip()
    if mean == mean_power:
        C = mean(mats, 0.42)
    else:
        C = mean(mats)
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize(
    "mean", [mean_logdet, mean_riemann, mean_wasserstein, nanmean_riemann]
)
def test_mean_shape_with_init(kind, mean, get_mats):
    """Test the shape of mean with init"""
    n_matrices, n_channels = 5, 3
    mats = get_mats(n_matrices, n_channels, kind)
    C = mean(mats, init=mats[0])
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
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
def test_mean_weight_zero(kind, mean, get_mats):
    """Setting one weight to almost 0 it's almost like not passing the mat"""
    n_matrices, n_channels = 5, 3
    mats = get_mats(n_matrices, n_channels, kind)
    w = 2.3 * np.ones(n_matrices)
    C = mean(mats[1:], sample_weight=w[1:])
    w[0] = 1e-12
    Cw = mean(mats, sample_weight=w)
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
    mats = get_covmats(n_matrices, n_channels)
    with pytest.raises(ValueError):
        mean(mats, sample_weight=np.ones(n_matrices + 1))


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
    mats = get_covmats(n_matrices, n_channels)
    with pytest.warns(UserWarning):
        if mean == mean_power:
            mean(mats, 0.3, maxiter=0)
        else:
            mean(mats, maxiter=0)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
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
def test_mean_of_means(kind, mean, get_mats):
    """Test mean of submeans equal to grand mean"""
    n_matrices, n_channels = 10, 3
    mats = get_mats(n_matrices, n_channels, kind)
    if mean is mean_ale and kind == "hpd":
        pytest.skip()
    if mean == mean_power:
        p = -0.42
        C = mean(mats, p)
        C1 = mean(mats[:n_matrices//2], p)
        C2 = mean(mats[n_matrices//2:], p)
        C3 = mean(np.array([C1, C2]), p)
    else:
        C = mean(mats)
        C1 = mean(mats[:n_matrices//2])
        C2 = mean(mats[n_matrices//2:])
        C3 = mean(np.array([C1, C2]))
    assert C3 == approx(C, 6)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_mean_alm(kind, get_mats):
    """Test the ALM mean"""
    n_matrices, n_channels = 3, 3
    mats = get_mats(n_matrices, n_channels, kind)
    C_alm = mean_alm(mats)
    assert C_alm.shape == (n_channels, n_channels)
    C_riem = mean_riemann(mats)
    assert C_alm == approx(C_riem, abs=1e-6, rel=1e-3)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_mean_alm_2matrices(kind, get_mats):
    """Test the ALM mean with 2 matrices"""
    n_matrices, n_channels = 2, 3
    mats = get_mats(n_matrices, n_channels, kind)
    C = mean_alm(mats)
    assert np.all(C == geodesic_riemann(mats[0], mats[1], alpha=0.5))


@pytest.mark.parametrize("complex_valued", [True, False])
def test_mean_euclid(rndstate, complex_valued):
    """Test the Euclidean mean for generic matrices"""
    n_matrices, n_dim0, n_dim1 = 10, 3, 4
    mats = rndstate.randn(n_matrices, n_dim0, n_dim1)
    if complex_valued:
        mats = mats + 1j * rndstate.randn(n_matrices, n_dim0, n_dim1)
    assert mean_euclid(mats) == approx(mats.mean(axis=0))


def test_mean_identity(get_covmats):
    """Test the identity mean"""
    n_matrices, n_channels = 2, 3
    mats = get_covmats(n_matrices, n_channels)
    C = mean_identity(mats)
    assert np.all(C == np.eye(n_channels))


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_mean_power(kind, get_mats):
    """Test the power mean"""
    n_matrices, n_channels = 3, 3
    mats = get_mats(n_matrices, n_channels, kind)
    assert mean_power(mats, 1) == approx(mean_euclid(mats))
    assert mean_power(mats, 0) == approx(mean_riemann(mats))
    assert mean_power(mats, -1) == approx(mean_harmonic(mats))


def test_mean_power_errors(get_covmats):
    """Test the power mean errors"""
    n_matrices, n_channels = 3, 2
    mats = get_covmats(n_matrices, n_channels)

    with pytest.raises(ValueError):  # exponent is not a scalar
        mean_power(mats, [1])
    with pytest.raises(ValueError):  # exponent is not in [-1,1]
        mean_power(mats, 3)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("init", [True, False])
def test_mean_riemann(kind, init, get_mats_params):
    """Test the Riemannian mean with same eigen vectors"""
    n_matrices, n_channels = 10, 3
    mats, eigvals, eigvecs = get_mats_params(n_matrices, n_channels, kind)
    if init:
        C = mean_riemann(mats, init=mats[0])
    else:
        C = mean_riemann(mats)
    eigval = np.exp(np.mean(np.log(eigvals), axis=0))
    Ctrue = eigvecs @ np.diag(eigval) @ eigvecs.conj().T
    assert C == approx(Ctrue)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_mean_riemann_properties(kind, get_mats):
    n_matrices, n_channels = 5, 3
    mats = get_mats(n_matrices, n_channels, kind)
    C = mean_riemann(mats)

    # congruence-invariance, P2 in [Moakher2005] or P6 in [Nakamura2009]
    W = np.random.normal(size=(n_channels, n_channels))  # must be invertible
    assert W @ C @ W.T == approx(mean_riemann(W @ mats @ W.T))

    # self-duality, P3 in [Moakher2005] or P8 in [Nakamura2009]
    assert C == approx(np.linalg.inv(mean_riemann(np.linalg.inv(mats))))

    # determinant identity, P9 in [Nakamura2009]
    assert np.linalg.det(C) == approx(gmean(np.linalg.det(mats)))


@pytest.mark.parametrize("init", [True, False])
def test_mean_masked_riemann_shape(init, get_covmats, get_masks):
    """Test the masked Riemannian mean"""
    n_matrices, n_channels = 5, 3
    mats = get_covmats(n_matrices, n_channels)
    masks = get_masks(n_matrices, n_channels)
    if init:
        C = maskedmean_riemann(mats, masks, tol=10e-3, init=mats[0])
    else:
        C = maskedmean_riemann(mats, masks, tol=10e-3)
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize("init", [True, False])
def test_mean_nan_riemann_shape(init, get_covmats, rndstate):
    """Test the Riemannian NaN-mean"""
    n_matrices, n_channels = 10, 6
    mats = get_covmats(n_matrices, n_channels)
    emean = np.mean(mats, axis=0)
    for i in range(n_matrices):
        corrup_channels = rndstate.choice(
            np.arange(0, n_channels), size=n_channels // 3, replace=False)
        for j in corrup_channels:
            mats[i, j] = np.nan
            mats[i, :, j] = np.nan
    if init:
        C = nanmean_riemann(mats, tol=10e-3, init=emean)
    else:
        C = nanmean_riemann(mats, tol=10e-3)
    assert C.shape == (n_channels, n_channels)


def test_mean_nan_riemann_errors(get_covmats):
    """Test the Riemannian NaN-mean errors"""
    n_matrices, n_channels = 5, 4
    mats = get_covmats(n_matrices, n_channels)

    with pytest.raises(ValueError):  # not symmetric NaN values
        mats_ = mats.copy()
        mats_[0, 0] = np.nan  # corrup only a row, not its corresp column
        nanmean_riemann(mats_)
    with pytest.raises(ValueError):  # not rows and columns NaN values
        mats_ = mats.copy()
        mats_[1, 0, 1] = np.nan  # corrup an off-diagonal value
        nanmean_riemann(mats_)


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
    mats = get_covmats(n_matrices, n_channels)
    C = mean_covariance(mats, metric=metric)
    Ctrue = mean(mats)
    assert np.all(C == Ctrue)


def test_mean_covariance_args(get_covmats):
    """Test mean_covariance with different arguments"""
    n_matrices, n_channels = 3, 3
    mats = get_covmats(n_matrices, n_channels)
    mean_covariance(mats, metric='ale', maxiter=5)
    mean_covariance(mats, metric='logdet', tol=10e-3)
    mean_covariance(mats, metric='riemann', init=np.eye(n_channels))
