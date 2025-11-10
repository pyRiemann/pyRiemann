import numpy as np
import pytest
from pytest import approx

from pyriemann.utils.geodesic import (
    geodesic,
    geodesic_chol,
    geodesic_euclid,
    geodesic_logchol,
    geodesic_logeuclid,
    geodesic_riemann,
    geodesic_wasserstein
)
from pyriemann.utils.mean import mean_covariance


metrics = [
    "chol",
    "euclid",
    "logchol",
    "logeuclid",
    "riemann",
    "wasserstein"
]


def assert_geodesics(metric, A, B, M):
    assert geodesic(A, B, 0, metric=metric) == approx(A)
    assert geodesic(A, B, 1, metric=metric) == approx(B)
    assert geodesic(A, B, 0.5, metric=metric) == approx(M)


@pytest.mark.parametrize(
    "geo",
    [
        geodesic_chol,
        geodesic_euclid,
        geodesic_logchol,
        geodesic_logeuclid,
        geodesic_riemann,
        geodesic_wasserstein,
    ],
)
def test_geodesic_ndarray(geo, get_mats):
    n_matrices, n_channels = 5, 3
    A = get_mats(n_matrices, n_channels, "spd")
    B = get_mats(n_matrices, n_channels, "spd")

    assert geo(A[0], B[0], .3).shape == A[0].shape  # 2D arrays

    assert geo(A, B, .2).shape == A.shape  # 3D arrays

    n_sets = 4
    C = np.asarray([A for _ in range(n_sets)])
    D = np.asarray([B for _ in range(n_sets)])
    assert geo(C, D, .7).shape == C.shape  # 4D arrays


@pytest.mark.parametrize("metric", metrics)
def test_geodesic_eye(metric):
    n_channels = 3
    eye = np.eye(n_channels)
    a = 0.5
    if metric == "chol":
        b = (2 - np.sqrt(0.5)) ** 2
    elif metric == "euclid":
        b = 1.5
    elif metric == "wasserstein":
        b = (9/2 - np.sqrt(8))
    else:
        b = 2
    assert_geodesics(metric, a * eye, b * eye, eye)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", metrics)
def test_geodesic_random(kind, metric, get_mats):
    n_matrices, n_channels = 2, 5
    X = get_mats(n_matrices, n_channels, kind)
    A, B = X[0], X[1]
    M = mean_covariance(X, metric=metric)
    assert_geodesics(metric, A, B, M)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", metrics)
def test_geodesic_properties(kind, metric, get_mats, rndstate):
    n_channels = 4
    A, B = get_mats(2, n_channels, kind)

    # WG3 in [Nakamura2009]
    alpha = rndstate.uniform(0.01, 0.99)
    G1 = geodesic(A, B, alpha, metric=metric)
    G2 = geodesic(B, A, 1 - alpha, metric=metric)
    assert G1 == approx(G2)

    # WG11 in [Nakamura2009]
    beta = rndstate.uniform(0.01, 0.99)
    G1 = geodesic(A, geodesic(A, B, beta, metric=metric), alpha, metric=metric)
    G2 = geodesic(A, B, alpha * beta, metric=metric)
    assert G1 == approx(G2)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("gfun", [
    geodesic_logeuclid,
    geodesic_riemann,
])
def test_geodesic_property_joint_homogeneity(kind, gfun, get_mats, rndstate):
    n_channels = 3
    A, B = get_mats(2, n_channels, kind)
    alpha, s1, s2 = rndstate.uniform(0.01, 0.99, size=3)
    s = (s1 ** (1 - alpha)) * (s2 ** alpha)
    assert gfun(s1 * A, s2 * B, alpha) == approx(s * gfun(A, B, alpha))


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


@pytest.mark.parametrize("n_dim1, n_dim2", [(4, 5), (5, 4)])
@pytest.mark.parametrize("kind", ["real", "comp"])
def test_geodesic_euclid(n_dim1, n_dim2, kind, get_mats):
    """Euclidean geodesic for non-square matrices"""
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
