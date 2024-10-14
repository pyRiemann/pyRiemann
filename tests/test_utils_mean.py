import numpy as np
import pytest
from pytest import approx
from scipy.stats import gmean

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
    mean_logchol,
    mean_logeuclid,
    mean_power,
    mean_poweuclid,
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
def test_mean(kind, mean, get_mats):
    """Test the shape of mean"""
    n_matrices, n_channels = 5, 3
    mats = get_mats(n_matrices, n_channels, kind)
    if mean == mean_power:
        M = mean(mats, 0.42)
    else:
        M = mean(mats)
    assert M.shape == (n_channels, n_channels)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize(
    "mean",
    [
        mean_ale,
        mean_logdet,
        mean_power,
        mean_riemann,
        mean_wasserstein,
        nanmean_riemann,
    ]
)
def test_mean_init(kind, mean, get_mats):
    """Test the shape of mean with init"""
    n_matrices, n_channels = 4, 3
    mats = get_mats(n_matrices, n_channels, kind)

    init = mats[0]
    if mean == mean_power:
        M = mean(mats, 0.123, init=init)
    else:
        M = mean(mats, init=init)
    assert M.shape == (n_channels, n_channels)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize(
    "mean",
    [
        mean_euclid,
        mean_harmonic,
        mean_kullback_sym,
        mean_logdet,
        mean_logchol,
        mean_logeuclid,
        mean_riemann,
        mean_wasserstein,
        nanmean_riemann,
    ],
)
def test_mean_weight_zero(kind, mean, get_mats, get_weights):
    """Setting one weight to almost 0 it's almost like not passing the mat"""
    n_matrices, n_channels = 5, 3
    mats = get_mats(n_matrices, n_channels, kind)
    weights = get_weights(n_matrices)

    M = mean(mats[1:], sample_weight=weights[1:])
    weights[0] = 1e-12
    Mw = mean(mats, sample_weight=weights)
    assert M == approx(Mw, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize(
    "mean",
    [
        mean_euclid,
        mean_harmonic,
        mean_kullback_sym,
        mean_logdet,
        mean_logchol,
        mean_logeuclid,
        mean_riemann,
        mean_wasserstein,
        nanmean_riemann,
    ],
)
def test_mean_weight_error(mean, get_mats, get_weights):
    n_matrices, n_channels = 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    weights = get_weights(n_matrices + 1)
    with pytest.raises(ValueError):
        mean(mats, sample_weight=weights)


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
def test_mean_warning_convergence(mean, get_mats):
    """Test warning for convergence not reached """
    n_matrices, n_channels = 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
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
        mean_logchol,
        mean_logdet,
        mean_logeuclid,
        mean_power,
        mean_poweuclid,
        mean_riemann,
        mean_wasserstein,
    ],
)
def test_mean_of_means(kind, mean, get_mats):
    """Test mean of submeans equal to grand mean"""
    n_matrices, n_channels = 10, 3
    mats = get_mats(n_matrices, n_channels, kind)
    if mean in [mean_power, mean_poweuclid]:
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


@pytest.mark.parametrize(
    "mean",
    [
        mean_ale,
        mean_alm,
        mean_euclid,
        mean_harmonic,
        mean_kullback_sym,
        mean_logchol,
        mean_logdet,
        mean_logeuclid,
        mean_power,
        mean_poweuclid,
        mean_riemann,
        mean_wasserstein,
        nanmean_riemann,
    ],
)
def test_mean_of_single_matrix(mean, get_mats):
    """Test the mean of a single matrix"""
    n_channels = 3
    mats = get_mats(1, n_channels, "spd")
    if mean in [mean_power, mean_poweuclid]:
        M = mean(mats, 0.42)
    else:
        M = mean(mats)
    assert M == approx(mats[0])


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


def test_mean_identity(get_mats):
    """Test the identity mean"""
    n_matrices, n_channels = 2, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    C = mean_identity(mats)
    assert np.all(C == np.eye(n_channels))


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_mean_power(kind, get_mats, get_weights):
    """Test the power mean"""
    n_matrices, n_channels = 3, 3
    mats = get_mats(n_matrices, n_channels, kind)
    assert mean_power(mats, 1) == approx(mean_euclid(mats))
    assert mean_power(mats, 0) == approx(mean_riemann(mats))
    assert mean_power(mats, -1) == approx(mean_harmonic(mats))

    weights = get_weights(n_matrices)
    mean_power(mats, 0.42, sample_weight=weights)


def test_mean_power_errors(get_mats):
    n_matrices, n_channels = 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")

    with pytest.raises(ValueError):  # exponent is not a scalar
        mean_power(mats, [1])
    with pytest.raises(ValueError):  # exponent is not in [-1,1]
        mean_power(mats, 3)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_mean_poweuclid(kind, get_mats, get_weights):
    n_matrices, n_channels = 10, 4
    mats = get_mats(n_matrices, n_channels, kind)
    assert mean_poweuclid(mats, 1) == approx(mean_euclid(mats))
    assert mean_poweuclid(mats, 0) == approx(mean_logeuclid(mats))
    assert mean_poweuclid(mats, -1) == approx(mean_harmonic(mats))

    weights = get_weights(n_matrices)
    mean_poweuclid(mats, 0.42, sample_weight=weights)


def test_mean_poweuclid_error(get_mats):
    n_matrices, n_channels = 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")

    with pytest.raises(ValueError):  # exponent is not a scalar
        mean_poweuclid(mats, [1])


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
def test_mean_masked_riemann(init, get_mats, get_masks):
    """Test the masked Riemannian mean"""
    n_matrices, n_channels = 5, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    masks = get_masks(n_matrices, n_channels)
    if init:
        C = maskedmean_riemann(mats, masks, tol=10e-3, init=mats[0])
    else:
        C = maskedmean_riemann(mats, masks, tol=10e-3)
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize("init", [True, False])
def test_mean_nan_riemann(init, get_mats, rndstate):
    """Test the Riemannian NaN-mean"""
    n_matrices, n_channels = 10, 6
    mats = get_mats(n_matrices, n_channels, "spd")
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


def test_mean_nan_riemann_errors(get_mats):
    """Test the Riemannian NaN-mean errors"""
    n_matrices, n_channels = 5, 4
    mats = get_mats(n_matrices, n_channels, "spd")

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
        ("logchol", mean_logchol),
        ("logdet", mean_logdet),
        ("logeuclid", mean_logeuclid),
        ("power", mean_power),
        ("poweuclid", mean_poweuclid),
        ("riemann", mean_riemann),
        ("wasserstein", mean_wasserstein),
        (callable_np_average, mean_euclid),
    ],
)
def test_mean_covariance_metric(metric, mean, get_mats):
    """Test mean_covariance for metric"""
    n_matrices, n_channels = 3, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    if metric in ["power", "poweuclid"]:
        p = 0.1
        assert mean(mats, p) == approx(mean_covariance(mats, p, metric=metric))
    else:
        assert mean(mats) == approx(mean_covariance(mats, metric=metric))


def test_mean_covariance_arguments(get_mats):
    """Test mean_covariance with different args and kwargs"""
    n_matrices, n_channels = 3, 3
    mats = get_mats(n_matrices, n_channels, "spd")

    mean_covariance(mats)
    mean_covariance(mats, 0.2, metric="power", zeta=10e-3)
    mean_covariance(mats, 0.3, metric="poweuclid", sample_weight=None)

    mean_covariance(mats, metric="ale", maxiter=5)
    mean_covariance(mats, metric="logdet", tol=10e-3)
    mean_covariance(mats, metric="riemann", init=np.eye(n_channels))


def test_mean_covariance_deprecation(get_mats):
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mean_covariance(get_mats(3, 2, "spd"), "euclid")
        assert len(w) >= 1
        assert issubclass(w[-1].category, DeprecationWarning)
