import numpy as np
import pytest
from pytest import approx
from scipy.stats import gmean, hmean

from pyriemann.utils.base import invsqrtm, logm, sqrtm
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.mean import (
    mean_covariance,
    mean_ale,
    mean_alm,
    mean_chol,
    mean_euclid,
    mean_harmonic,
    mean_kullback_sym,
    mean_logdet,
    mean_logchol,
    mean_logeuclid,
    mean_power,
    mean_poweuclid,
    mean_riemann,
    mean_thompson,
    mean_wasserstein,
    maskedmean_riemann,
    nanmean_riemann,
)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize(
    "mean",
    [
        mean_ale,
        mean_chol,
        mean_euclid,
        mean_harmonic,
        mean_kullback_sym,
        mean_logdet,
        mean_logeuclid,
        mean_power,
        mean_riemann,
        mean_thompson,
        mean_wasserstein,
        nanmean_riemann,
    ],
)
def test_mean(kind, mean, get_mats):
    """Test the shape of mean"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, kind)
    if mean == mean_power:
        M = mean(X, 0.42)
    else:
        M = mean(X)
    assert M.shape == (n_channels, n_channels)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize(
    "mean",
    [
        mean_ale,
        mean_logdet,
        mean_power,
        mean_riemann,
        mean_thompson,
        mean_wasserstein,
        nanmean_riemann,
    ]
)
def test_mean_init(kind, mean, get_mats):
    """Test the shape of mean with init"""
    n_matrices, n_channels = 4, 3
    X = get_mats(n_matrices, n_channels, kind)

    init = X[0]
    if mean == mean_power:
        M = mean(X, 0.123, init=init)
    else:
        M = mean(X, init=init)
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
    X = get_mats(n_matrices, n_channels, kind)
    weights = get_weights(n_matrices)

    M = mean(X[1:], sample_weight=weights[1:])
    weights[0] = 1e-12
    Mw = mean(X, sample_weight=weights)
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
    X = get_mats(n_matrices, n_channels, "spd")
    weights = get_weights(n_matrices + 1)
    with pytest.raises(ValueError):
        mean(X, sample_weight=weights)


@pytest.mark.parametrize(
    "mean", [
        mean_ale,
        mean_alm,
        mean_logdet,
        mean_power,
        mean_riemann,
        mean_thompson,
        mean_wasserstein,
        nanmean_riemann
    ]
)
def test_mean_warning_convergence(mean, get_mats):
    """Test warning for convergence not reached """
    n_matrices, n_channels = 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    with pytest.warns(UserWarning):
        if mean == mean_power:
            mean(X, 0.3, maxiter=0)
        else:
            mean(X, maxiter=0)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize(
    "mean",
    [
        mean_ale,
        mean_chol,
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
    ],
)
def test_mean_of_means(kind, mean, get_mats):
    """Test mean of submeans equal to grand mean"""
    n_matrices, n_channels = 10, 3
    X = get_mats(n_matrices, n_channels, kind)
    if mean in [mean_power, mean_poweuclid]:
        p = -0.42
        M = mean(X, p)
        M1 = mean(X[:n_matrices//2], p)
        M2 = mean(X[n_matrices//2:], p)
        M3 = mean(np.array([M1, M2]), p)
    else:
        M = mean(X)
        M1 = mean(X[:n_matrices//2])
        M2 = mean(X[n_matrices//2:])
        M3 = mean(np.array([M1, M2]))
    assert M3 == approx(M, 6)


@pytest.mark.parametrize(
    "mean",
    [
        mean_ale,
        mean_alm,
        mean_chol,
        mean_euclid,
        mean_harmonic,
        mean_kullback_sym,
        mean_logchol,
        mean_logdet,
        mean_logeuclid,
        mean_power,
        mean_poweuclid,
        mean_riemann,
        mean_thompson,
        mean_wasserstein,
        nanmean_riemann,
    ],
)
def test_mean_of_single_matrix(mean, get_mats):
    """Test the mean of a single matrix"""
    n_channels = 3
    X = get_mats(1, n_channels, "spd")
    if mean in [mean_power, mean_poweuclid]:
        M = mean(X, 0.42)
    else:
        M = mean(X)
    assert M == approx(X[0])


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("mean", [
    mean_logeuclid,
    mean_riemann,
    # mean_thompson,  # Th 6.16 (4) in [Mostajeran2024], KO
])
def test_mean_property_joint_homogeneity(kind, mean, get_mats, rndstate):
    """Test joint homogeneity"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, kind)

    # P2 in [Nakamura2009]
    a = rndstate.uniform(low=1.0, high=2.0, size=n_matrices)
    assert mean(a[:, np.newaxis, np.newaxis] * X) == approx(gmean(a) * mean(X))

    # P2' in [Nakamura2009]
    a = rndstate.uniform(0.01, 5.0)
    assert mean(a * X) == approx(a * mean(X))


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("mean", [
    mean_logchol,  # Corollary 13 in [Lin2019]
    mean_logeuclid,
    mean_riemann,
])
def test_mean_property_determinant_identity(kind, mean, get_mats, rndstate):
    """Test determinant identity, P9 in [Nakamura2009]"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, kind)
    assert np.linalg.det(mean(X)) == approx(gmean(np.linalg.det(X)))


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("mean", [
    mean_logeuclid,  # Th 3.13 in [Arsigny2007]
    mean_riemann,  # P8 in [Nakamura2009]
])
def test_mean_property_invariance_inversion(kind, mean, get_mats):
    """Test invariance under inversion, also called self-duality"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, kind)
    assert mean(X) == approx(np.linalg.inv(mean(np.linalg.inv(X))))


@pytest.mark.parametrize("kind, kindQ", [("spd", "orth"), ("hpd", "unit")])
@pytest.mark.parametrize("mean", [
    mean_logeuclid,  # Th 3.13 in [Arsigny2007]
    mean_riemann,
    mean_thompson,
])
def test_mean_property_invariance_similarity(kind, kindQ, mean,
                                             get_mats, rndstate):
    """Test invariance by similarity, ie a scale and a rotation"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, kind)
    Q = get_mats(1, n_channels, kindQ)[0]
    Qh = Q.conj().T
    scale = rndstate.uniform(0.01, 10.0)
    assert scale * Q @ mean(X) @ Qh == approx(mean(scale * Q @ X @ Qh))


@pytest.mark.parametrize("kind, kindW", [("spd", "inv"), ("hpd", "cinv")])
@pytest.mark.parametrize("mean", [
    mean_riemann,  # P6 in [Nakamura2009]
    mean_thompson,  # Th 6.16 (3) in [Mostajeran2024]
])
def test_mean_property_invariance_congruence(kind, kindW, mean, get_mats):
    """Test invariance under congruence, ie an invertible transform"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, kind)
    W = get_mats(1, n_channels, kindW)[0]
    Wh = W.conj().T
    assert W @ mean(X) @ Wh == approx(mean(W @ X @ Wh))


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_mean_alm(kind, get_mats):
    n_matrices, n_channels = 3, 3
    X = get_mats(n_matrices, n_channels, kind)
    M = mean_alm(X)
    assert M.shape == (n_channels, n_channels)
    assert M == approx(mean_riemann(X), abs=1e-6, rel=1e-3)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_mean_alm_2matrices(kind, get_mats):
    n_matrices, n_channels = 2, 3
    X = get_mats(n_matrices, n_channels, kind)
    assert mean_alm(X) == approx(geodesic_riemann(X[0], X[1], alpha=0.5))


@pytest.mark.parametrize("n_dim1, n_dim2", [(4, 5), (5, 4)])
@pytest.mark.parametrize("kind", ["real", "comp"])
def test_mean_euclid(n_dim1, n_dim2, kind, get_mats):
    """Euclidean mean for non-square matrices"""
    n_matrices = 10
    X = get_mats(n_matrices, [n_dim1, n_dim2], kind)
    assert mean_euclid(X) == approx(np.mean(X, axis=0))


@pytest.mark.parametrize("kind", ["inv", "cinv"])
def test_mean_harmonic(kind, get_mats):
    """harmonic mean of invertible matrices"""
    n_matrices, n_channels = 4, 5
    X = get_mats(n_matrices, n_channels, kind)
    mean_harmonic(X)


@pytest.mark.parametrize("n_values", [3, 5, 7])
def test_mean_harmonic_scalars(n_values, rndstate):
    """Compare harmonic mean to scipy.hmean for scalars"""
    values = rndstate.uniform(0.1, 10, size=n_values)
    sp_hmean = hmean(values)
    py_hmean = mean_harmonic(values[..., np.newaxis, np.newaxis])[0, 0]
    assert sp_hmean == approx(py_hmean)


@pytest.mark.parametrize("n_values", [4, 6, 8])
def test_mean_logeuclid_scalars(n_values, rndstate):
    """Compare log-Euclidean mean to scipy.gmean for scalars"""
    values = rndstate.uniform(0.1, 10, size=n_values)
    sp_gmean = gmean(values)
    py_lemean = mean_logeuclid(values[..., np.newaxis, np.newaxis])[0, 0]
    assert sp_gmean == approx(py_lemean)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_mean_power(kind, get_mats, get_weights):
    n_matrices, n_channels = 3, 3
    X = get_mats(n_matrices, n_channels, kind)
    assert mean_power(X, 1) == approx(mean_euclid(X))
    assert mean_power(X, 0) == approx(mean_riemann(X))
    assert mean_power(X, -1) == approx(mean_harmonic(X))

    weights = get_weights(n_matrices)
    mean_power(X, 0.42, sample_weight=weights)


def test_mean_power_errors(get_mats):
    n_matrices, n_channels = 3, 2
    X = get_mats(n_matrices, n_channels, "spd")

    with pytest.raises(ValueError):  # exponent is not a scalar
        mean_power(X, [1])
    with pytest.raises(ValueError):  # exponent is not in [-1,1]
        mean_power(X, 3)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_mean_poweuclid(kind, get_mats, get_weights):
    n_matrices, n_channels = 10, 4
    X = get_mats(n_matrices, n_channels, kind)
    assert mean_poweuclid(X, 1) == approx(mean_euclid(X))
    assert mean_poweuclid(X, 0) == approx(mean_logeuclid(X))
    assert mean_poweuclid(X, -1) == approx(mean_harmonic(X))

    weights = get_weights(n_matrices)
    mean_poweuclid(X, 0.42, sample_weight=weights)


def test_mean_poweuclid_error(get_mats):
    n_matrices, n_channels = 3, 2
    X = get_mats(n_matrices, n_channels, "spd")

    with pytest.raises(ValueError):  # exponent is not a scalar
        mean_poweuclid(X, [1])


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_mean_riemann_solution(kind, get_mats):
    """AIR mean is solution to the nonlinear matrix equations"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, kind)
    M = mean_riemann(X, tol=10e-16, maxiter=500)
    Zero = np.zeros((n_channels, n_channels))

    Mm12 = invsqrtm(M)
    assert np.sum(logm(Mm12 @ X @ Mm12), axis=0) == approx(Zero)

    # Eq(1.2) in [Lim2012]
    M12 = sqrtm(M)
    assert np.sum(logm(M12 @ np.linalg.inv(X) @ M12), axis=0) == approx(Zero)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_mean_riemann_same_eigenvecs(kind, get_mats_params):
    """Test the Riemannian mean with same eigen vectors"""
    n_matrices, n_channels = 10, 3
    X, eigvals, eigvecs = get_mats_params(n_matrices, n_channels, kind)
    M = mean_riemann(X)
    eigval = np.exp(np.mean(np.log(eigvals), axis=0))
    Mtrue = eigvecs @ np.diag(eigval) @ eigvecs.conj().T
    assert M == approx(Mtrue)


@pytest.mark.parametrize("init", [True, False])
def test_mean_masked_riemann(init, get_mats, get_masks):
    """Test the masked Riemannian mean"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    masks = get_masks(n_matrices, n_channels)
    if init:
        M = maskedmean_riemann(X, masks, tol=10e-3, init=X[0])
    else:
        M = maskedmean_riemann(X, masks, tol=10e-3)
    assert M.shape == (n_channels, n_channels)


@pytest.mark.parametrize("init", [True, False])
def test_mean_nan_riemann(init, get_mats, rndstate):
    """Test the Riemannian NaN-mean"""
    n_matrices, n_channels = 10, 6
    X = get_mats(n_matrices, n_channels, "spd")
    emean = np.mean(X, axis=0)
    for i in range(n_matrices):
        corrup_channels = rndstate.choice(
            np.arange(0, n_channels), size=n_channels // 3, replace=False)
        for j in corrup_channels:
            X[i, j] = np.nan
            X[i, :, j] = np.nan
    if init:
        M = nanmean_riemann(X, tol=10e-3, init=emean)
    else:
        M = nanmean_riemann(X, tol=10e-3)
    assert M.shape == (n_channels, n_channels)


def test_mean_nan_riemann_errors(get_mats):
    """Test the Riemannian NaN-mean errors"""
    n_matrices, n_channels = 5, 4
    X = get_mats(n_matrices, n_channels, "spd")

    with pytest.raises(ValueError):  # not symmetric NaN values
        X_ = X.copy()
        X_[0, 0] = np.nan  # corrup only a row, not its corresp column
        nanmean_riemann(X_)
    with pytest.raises(ValueError):  # not rows and columns NaN values
        X_ = X.copy()
        X_[1, 0, 1] = np.nan  # corrup an off-diagonal value
        nanmean_riemann(X_)


def callable_np_average(X, sample_weight=None):
    return np.average(X, axis=0, weights=sample_weight)


@pytest.mark.parametrize(
    "metric, mean",
    [
        ("ale", mean_ale),
        ("alm", mean_alm),
        ("chol", mean_chol),
        ("euclid", mean_euclid),
        ("harmonic", mean_harmonic),
        ("kullback_sym", mean_kullback_sym),
        ("logchol", mean_logchol),
        ("logdet", mean_logdet),
        ("logeuclid", mean_logeuclid),
        ("power", mean_power),
        ("poweuclid", mean_poweuclid),
        ("riemann", mean_riemann),
        ("thompson", mean_thompson),
        ("wasserstein", mean_wasserstein),
        (callable_np_average, mean_euclid),
    ],
)
def test_mean_covariance_metric(metric, mean, get_mats):
    """Test mean_covariance for metric"""
    n_matrices, n_channels = 3, 3
    X = get_mats(n_matrices, n_channels, "spd")
    if metric in ["power", "poweuclid"]:
        p = 0.1
        assert mean(X, p) == approx(mean_covariance(X, p, metric=metric))
    else:
        assert mean(X) == approx(mean_covariance(X, metric=metric))


def test_mean_covariance_arguments(get_mats):
    """Test mean_covariance with different args and kwargs"""
    n_matrices, n_channels = 3, 2
    X = get_mats(n_matrices, n_channels, "spd")

    mean_covariance(X)
    mean_covariance(X, 0.2, metric="power", zeta=10e-3)
    mean_covariance(X, 0.3, metric="poweuclid", sample_weight=None)

    mean_covariance(X, metric="ale", maxiter=5)
    mean_covariance(X, metric="logdet", tol=10e-3)
    mean_covariance(X, metric="riemann", init=np.eye(n_channels))
