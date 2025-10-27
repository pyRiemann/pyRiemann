import numpy as np
import pytest
from pytest import approx

from pyriemann.utils.geodesic import (
    geodesic,
    geodesic_euclid,
    geodesic_logchol,
    geodesic_logeuclid,
    geodesic_riemann,
    geodesic_wasserstein
)
from pyriemann.utils.mean import (
    mean_euclid,
    mean_logchol,
    mean_logeuclid,
    mean_riemann,
    mean_wasserstein,
)


def get_geod_func():
    geod_func = [
        geodesic_euclid,
        geodesic_logchol,
        geodesic_logeuclid,
        geodesic_riemann,
        geodesic_wasserstein
    ]
    for gf in geod_func:
        yield gf


def get_geod_name():
    geod_name = [
        "euclid",
        "logchol",
        "logeuclid",
        "riemann",
        "wasserstein"
    ]
    for gn in geod_name:
        yield gn


def assert_geodesic_0(gfun, A, B):
    assert gfun(A, B, 0) == approx(A)


def assert_geodesic_1(gfun, A, B):
    assert gfun(A, B, 1) == approx(B)


def assert_geodesic_middle(gfun, A, B, M):
    assert gfun(A, B, 0.5) == approx(M)


@pytest.mark.parametrize("gfun", get_geod_func())
def test_geodesic_eye(gfun):
    n_channels = 3
    eye = np.eye(n_channels)
    if gfun is geodesic_euclid:
        a, b, m = 1, 2, 1.5
    elif gfun is geodesic_wasserstein:
        a, b, m = 0.5, 2, 1.125
    else:
        a, b, m = 0.5, 2, 1
    assert_geodesic_0(gfun, a * eye, b * eye)
    assert_geodesic_1(gfun, a * eye, b * eye)
    assert_geodesic_middle(gfun, a * eye, b * eye, m * eye)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("gfun", get_geod_func())
def test_geodesic_random(kind, gfun, get_mats):
    n_matrices, n_channels = 2, 5
    X = get_mats(n_matrices, n_channels, kind)
    A, B = X[0], X[1]
    if gfun is geodesic_euclid:
        M = mean_euclid(X)
    elif gfun is geodesic_logchol:
        M = mean_logchol(X)
    elif gfun is geodesic_logeuclid:
        M = mean_logeuclid(X)
    elif gfun is geodesic_riemann:
        M = mean_riemann(X)
    elif gfun is geodesic_wasserstein:
        M = mean_wasserstein(X)
    assert_geodesic_0(gfun, A, B)
    assert_geodesic_1(gfun, A, B)
    assert_geodesic_middle(gfun, A, B, M)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("gfun", get_geod_func())
def test_geodesic_properties(kind, gfun, get_mats, rndstate):
    n_channels = 4
    A, B = get_mats(2, n_channels, kind)
    alpha = rndstate.uniform(0.01, 0.99)

    # WG3 in [Nakamura2009]
    assert gfun(A, B, alpha) == approx(gfun(B, A, 1 - alpha))

    # WG11 in [Nakamura2009]
    beta = rndstate.uniform(0.01, 0.99)
    assert gfun(A, gfun(A, B, beta), alpha) == approx(gfun(A, B, alpha * beta))


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("gfun", [
    geodesic_logeuclid,
    geodesic_riemann,
])
def test_geodesic_property_invariance_inversion(kind, gfun,
                                                get_mats, rndstate):
    """Test invariance under inversion, also called self-duality """
    n_channels = 4
    A, B = get_mats(2, n_channels, kind)
    alpha = rndstate.uniform(0.01, 0.99)
    G = gfun(A, B, alpha)
    Ginv = np.linalg.inv(gfun(np.linalg.inv(A), np.linalg.inv(B), alpha))
    assert G == approx(Ginv)


@pytest.mark.parametrize("kind, kindW", [("spd", "inv"), ("hpd", "cinv")])
@pytest.mark.parametrize("gfun", [
    geodesic_riemann,
])
def test_geodesic_property_invariance_congruence(kind, kindW, gfun,
                                                 get_mats, rndstate):
    """Test invariance under congruence, ie an invertible transform"""
    n_channels = 3
    A, B = get_mats(2, n_channels, kind)
    alpha = rndstate.uniform(0.01, 0.99)
    W = get_mats(1, n_channels, kindW)[0]
    WAW, WBW = W @ A @ W.conj().T, W @ B @ W.conj().T
    assert W @ gfun(A, B, alpha) @ W.conj().T == approx(gfun(WAW, WBW, alpha))


@pytest.mark.parametrize("kind", ["real", "comp"])
def test_geodesic_euclid(kind, get_mats):
    """Euclidean geodesic for non-square matrices"""
    n_dim1, n_dim2 = 3, 4
    A, B = get_mats(2, [n_dim1, n_dim2], kind)
    assert geodesic_euclid(A, B).shape == A.shape


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_geodesic_riemann(kind, get_mats, rndstate):
    n_channels = 4
    A, B = get_mats(2, n_channels, kind)
    alpha = rndstate.uniform(0.01, 0.99)
    G = geodesic_riemann(A, B, alpha)

    # WG9 in [Nakamura2009]
    det = (np.linalg.det(A) ** (1 - alpha)) * (np.linalg.det(B) ** alpha)
    assert np.linalg.det(G) == approx(det)


@pytest.mark.parametrize("metric", get_geod_name())
def test_geodesic_wrapper_ndarray(metric, get_mats):
    n_matrices, n_channels = 5, 3
    A = get_mats(n_matrices, n_channels, "spd")
    B = get_mats(n_matrices, n_channels, "spd")
    assert geodesic(A[0], B[0], .3, metric=metric).shape == A[0].shape
    assert geodesic(A, B, .2, metric=metric).shape == A.shape  # 3D arrays

    n_sets = 4
    C = np.asarray([A for _ in range(n_sets)])
    D = np.asarray([B for _ in range(n_sets)])
    assert geodesic(C, D, .7, metric=metric).shape == C.shape  # 4D arrays


@pytest.mark.parametrize("metric", get_geod_name())
def test_geodesic_wrapper_eye(metric):
    n_channels = 3
    eye = np.eye(n_channels)
    a = 0.5
    if metric == "euclid":
        b = 1.5
    elif metric == "wasserstein":
        b = (9/2 - np.sqrt(8))
    else:
        b = 2.0 * eye
    assert geodesic(a * eye, b * eye, 0.5, metric=metric) == approx(eye)


@pytest.mark.parametrize("metric, gfun", zip(get_geod_name(), get_geod_func()))
def test_geodesic_wrapper_random(metric, gfun, get_mats):
    n_channels = 5
    X = get_mats(2, n_channels, "spd")
    if gfun is geodesic_euclid:
        M = mean_euclid(X)
    elif gfun is geodesic_logchol:
        M = mean_logchol(X)
    elif gfun is geodesic_logeuclid:
        M = mean_logeuclid(X)
    elif gfun is geodesic_riemann:
        M = mean_riemann(X)
    elif gfun is geodesic_wasserstein:
        M = mean_wasserstein(X)
    assert geodesic(X[0], X[1], 0.5, metric=metric) == approx(M)
