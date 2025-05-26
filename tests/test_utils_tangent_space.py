import numpy as np
import pytest
from pytest import approx

from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.tangentspace import (
    exp_map,
    exp_map_euclid,
    exp_map_logchol,
    exp_map_logeuclid,
    exp_map_riemann,
    exp_map_wasserstein,
    log_map,
    log_map_euclid,
    log_map_logchol,
    log_map_logeuclid,
    log_map_riemann,
    log_map_wasserstein,
    upper,
    unupper,
    tangent_space,
    untangent_space,
    transport,
)


@pytest.mark.parametrize(
    "fun_map", [
        exp_map_euclid,
        exp_map_logchol,
        exp_map_logeuclid,
        exp_map_riemann,
        exp_map_wasserstein,
        log_map_euclid,
        log_map_logchol,
        log_map_logeuclid,
        log_map_riemann,
        log_map_wasserstein
    ]
)
def test_maps_ndarray(fun_map, get_mats):
    """Test log and exp maps"""
    n_matrices, n_channels = 6, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    Xt = fun_map(mats, np.eye(n_channels))
    assert Xt.shape == (n_matrices, n_channels, n_channels)

    n_sets = 2
    mats_4d = np.asarray([mats for _ in range(n_sets)])
    Xt = fun_map(mats_4d, np.eye(n_channels))
    assert Xt.shape == (n_sets, n_matrices, n_channels, n_channels)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", [
    "euclid", "logchol", "logeuclid", "riemann", "wasserstein"
])
def test_map_log_exp(kind, metric, get_mats):
    """Test log then exp maps should be identity"""
    n_matrices, n_channels = 9, 2
    mats = get_mats(n_matrices, n_channels, kind)
    X, C = mats[:-1], mats[-1]
    assert exp_map(log_map(X, C, metric=metric), C, metric=metric) == approx(X)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize(
    "log_map_, exp_map_", zip(
        [
            log_map_euclid,
            log_map_logchol,
            log_map_logeuclid,
            log_map_riemann,
            log_map_wasserstein,
        ],
        [
            exp_map_euclid,
            exp_map_logchol,
            exp_map_logeuclid,
            exp_map_riemann,
            exp_map_wasserstein,
        ]
    )
)
def test_maps_log_exp(kind, log_map_, exp_map_, get_mats):
    """Test log then exp maps should be identity"""
    n_matrices, n_channels = 10, 3
    mats = get_mats(n_matrices, n_channels, kind)
    X, C = mats[:-1], mats[-1]
    assert exp_map_(log_map_(X, C), C) == approx(X)


@pytest.mark.parametrize("complex_valued", [True, False])
def test_map_euclid(rndstate, complex_valued):
    """Test Euclidean map for generic matrices"""
    n_matrices, n_dim0, n_dim1 = 5, 3, 4
    mats = rndstate.randn(n_matrices, n_dim0, n_dim1)
    if complex_valued:
        mats = mats + 1j * rndstate.randn(n_matrices, n_dim0, n_dim1)
    X, C = mats[:n_matrices - 1], mats[-1]
    assert exp_map_euclid(log_map_euclid(X, C), C) == approx(X)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_upper_and_unupper(kind, get_mats):
    """Test upper then unupper should be identity"""
    n_matrices, n_channels = 7, 3
    mats = get_mats(n_matrices, n_channels, kind)
    assert unupper(upper(mats)) == approx(mats)

    n_sets = 2
    mats_4d = np.asarray([mats for _ in range(n_sets)])
    assert unupper(upper(mats_4d)) == approx(mats_4d)


@pytest.mark.parametrize("metric", [
    "euclid", "logchol", "logeuclid", "riemann"
])
def test_tangent_space_ndarray(metric, get_mats):
    """Test tangent space projection"""
    n_matrices, n_channels = 6, 3
    n_ts = (n_channels * (n_channels + 1)) // 2
    X = get_mats(n_matrices, n_channels, "spd")
    Xts = tangent_space(X, np.eye(n_channels), metric=metric)
    assert Xts.shape == (n_matrices, n_ts)

    n_sets = 2
    X_4d = np.asarray([X for _ in range(n_sets)])
    Xts = tangent_space(X_4d, np.eye(n_channels), metric=metric)
    assert Xts.shape == (n_sets, n_matrices, n_ts)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_tangent_space_riemann_properties(kind, get_mats):
    n_matrices, n_channels = 2, 3
    mats = get_mats(n_matrices, n_channels, kind)
    A, B = mats[0], mats[1]

    # equivalent definitions of Riemannian distance, Eq(7) in [Barachant2012]
    dist = distance_riemann(A, B)
    s = tangent_space(A, B, metric="riemann")
    assert dist == approx(np.linalg.norm(s))


@pytest.mark.parametrize("metric", [
    "euclid", "logchol", "logeuclid", "riemann", "wasserstein"
])
def test_untangent_space_ndarray(metric, rndstate):
    """Test untangent space projection"""
    n_matrices, n_channels = 10, 3
    n_ts = (n_channels * (n_channels + 1)) // 2
    T = rndstate.randn(n_matrices, n_ts)
    X = untangent_space(T, np.eye(n_channels), metric=metric)
    assert X.shape == (n_matrices, n_channels, n_channels)

    n_sets = 2
    T_4d = np.asarray([T for _ in range(n_sets)])
    X = untangent_space(T_4d, np.eye(n_channels), metric=metric)
    assert X.shape == (n_sets, n_matrices, n_channels, n_channels)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", [
    "euclid", "logchol", "logeuclid", "riemann", "wasserstein"
])
def test_tangent_and_untangent_space(kind, metric, get_mats):
    """Test tangent space projection then back-projection should be identity"""
    n_matrices, n_channels = 10, 3
    mats = get_mats(n_matrices, n_channels, kind)
    X, C = mats[:n_matrices - 1], mats[-1]
    X_t = tangent_space(X, C, metric=metric)
    X_ut = untangent_space(X_t, C, metric=metric)
    assert X_ut == approx(X)


@pytest.mark.parametrize("kind", ["sym", "herm"])
def test_transport(get_mats, kind):
    n_matrices, n_channels = 10, 3
    X = get_mats(n_matrices, n_channels, kind)
    X_tr = transport(X, np.eye(n_channels), np.eye(n_channels))
    assert X == approx(X_tr)
