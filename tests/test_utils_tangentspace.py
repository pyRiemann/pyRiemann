import numpy as np
import pytest
from pytest import approx

from pyriemann.spatialfilters import Whitening
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
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
    transport_euclid,
    transport_logchol,
    transport_logeuclid,
    transport_riemann,
)

metrics = ["euclid", "logchol", "logeuclid", "riemann", "wasserstein"]


@pytest.mark.parametrize(
    "fmap",
    [
        exp_map_euclid,
        exp_map_logchol,
        exp_map_logeuclid,
        exp_map_riemann,
        exp_map_wasserstein,
        log_map_euclid,
        log_map_logchol,
        log_map_logeuclid,
        log_map_riemann,
        log_map_wasserstein,
    ],
)
def test_maps_ndarray(fmap, get_mats):
    """Test log and exp maps"""
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    Xt = fmap(X, np.eye(n_channels))
    assert Xt.shape == (n_matrices, n_channels, n_channels)

    n_sets = 2
    X_4d = np.asarray([X for _ in range(n_sets)])
    Xt = fmap(X_4d, np.eye(n_channels))
    assert Xt.shape == (n_sets, n_matrices, n_channels, n_channels)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", metrics)
def test_map_log_exp(kind, metric, get_mats):
    """Test log then exp maps should be identity"""
    n_matrices, n_channels = 9, 2
    mats = get_mats(n_matrices, n_channels, kind)
    X, C = mats[:-1], mats[-1]
    assert exp_map(log_map(X, C, metric=metric), C, metric=metric) == approx(X)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize(
    "log_map_, exp_map_",
    zip(
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
        ],
    ),
)
def test_maps_log_exp(kind, log_map_, exp_map_, get_mats):
    """Test log then exp maps should be identity"""
    n_matrices, n_channels = 10, 3
    mats = get_mats(n_matrices, n_channels, kind)
    X, C = mats[:-1], mats[-1]
    assert exp_map_(log_map_(X, C), C) == approx(X)


@pytest.mark.parametrize("n_dim1, n_dim2", [(4, 5), (5, 4)])
@pytest.mark.parametrize("kind", ["real", "comp"])
def test_map_euclid(n_dim1, n_dim2, kind, get_mats):
    """Euclidean map for non-square matrices"""
    n_matrices = 7
    mats = get_mats(n_matrices, [n_dim1, n_dim2], kind)
    X, C = mats[: n_matrices - 1], mats[-1]
    assert exp_map_euclid(log_map_euclid(X, C), C) == approx(X)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_upper_and_unupper(kind, get_mats):
    """Test upper then unupper should be identity"""
    n_matrices, n_channels = 7, 3
    X = get_mats(n_matrices, n_channels, kind)
    assert unupper(upper(X)) == approx(X)

    n_sets = 2
    X_4d = np.asarray([X for _ in range(n_sets)])
    assert unupper(upper(X_4d)) == approx(X_4d)


@pytest.mark.parametrize("metric", metrics)
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
    n_channels = 3
    A, B = get_mats(2, n_channels, kind)

    # equivalent definitions of Riemannian distance, Eq(7) in [Barachant2012]
    dist = distance_riemann(A, B)
    s = tangent_space(A, B, metric="riemann")
    assert dist == approx(np.linalg.norm(s))


@pytest.mark.parametrize("metric", metrics)
def test_untangent_space_ndarray(metric, get_mats):
    """Test untangent space projection"""
    n_matrices, n_channels = 10, 3
    n_ts = (n_channels * (n_channels + 1)) // 2
    T = get_mats(n_matrices, [n_ts], "real")
    X = untangent_space(T, np.eye(n_channels), metric=metric)
    assert X.shape == (n_matrices, n_channels, n_channels)

    n_sets = 2
    T_4d = np.asarray([T for _ in range(n_sets)])
    X = untangent_space(T_4d, np.eye(n_channels), metric=metric)
    assert X.shape == (n_sets, n_matrices, n_channels, n_channels)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", metrics)
def test_tangent_and_untangent_space(kind, metric, get_mats):
    """Tangent space projection then back-projection should be identity"""
    n_matrices, n_channels = 10, 3
    mats = get_mats(n_matrices, n_channels, kind)
    X, C = mats[: n_matrices - 1], mats[-1]
    X_t = tangent_space(X, C, metric=metric)
    X_ut = untangent_space(X_t, C, metric=metric)
    assert X_ut == approx(X)


@pytest.mark.parametrize(
    "ftransport",
    [
        transport_euclid,
        transport_logchol,
        transport_logeuclid,
        transport_riemann,
    ],
)
def test_transport_ndarray(ftransport, get_mats):
    n_matrices, n_channels = 7, 3
    X = get_mats(n_matrices, n_channels, "herm")
    A, B = get_mats(2, n_channels, "hpd")

    X_tr = ftransport(X, A, B)
    assert X_tr.shape == X.shape

    n_sets = 2
    X_4d = np.asarray([X for _ in range(n_sets)])
    X_tr = ftransport(X_4d, A, B)
    assert X_tr.shape == X_4d.shape


@pytest.mark.parametrize(
    "ftransport",
    [
        transport_logchol,
        transport_logeuclid,
        transport_riemann,
    ],
)
def test_transport_properties(ftransport, get_mats):
    n_matrices, n_channels = 10, 3
    X = get_mats(n_matrices, n_channels, "sym")
    A, B = get_mats(2, n_channels, "spd")

    # trivial transport
    assert ftransport(X, A, A) == approx(X)

    # reversibility
    assert ftransport(ftransport(X, A, B), B, A) == approx(X)


def test_transport_riemann_property_linearity(get_mats):
    n_matrices, n_channels = 7, 4
    X = get_mats(n_matrices, n_channels, "sym")
    Y = get_mats(n_matrices, n_channels, "sym")
    A, B = get_mats(2, n_channels, "spd")

    Xt, Yt = transport_riemann(X, A, B), transport_riemann(Y, A, B)
    assert transport_riemann(X + Y, A, B) == approx(Xt + Yt)


def test_transport_riemann_vs_whitening(get_mats):
    """AIR PT from mean to identity should be equivalent to a whitening"""
    n_matrices, n_channels = 15, 2
    X = get_mats(n_matrices, n_channels, "spd")

    Xw = Whitening(dim_red=None, metric="riemann").fit_transform(X)

    M = mean_riemann(X)
    T = log_map_riemann(X, M, C12=True)
    Tt = transport(T, M, np.eye(n_channels), metric="riemann")
    Xt = exp_map_riemann(Tt, np.eye(n_channels), Cm12=True)
    assert Xw == approx(Xt)
