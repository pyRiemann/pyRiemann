import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from pytest import approx
from scipy.spatial.distance import euclidean, mahalanobis

from conftest import get_distances
from pyriemann.utils.distance import (
    distance_chol,
    distance_euclid,
    distance_harmonic,
    distance_kullback,
    distance_kullback_right,
    distance_kullback_sym,
    distance_logchol,
    distance_logdet,
    distance_logeuclid,
    distance_poweuclid,
    distance_riemann,
    distance_wasserstein,
    distance,
    pairwise_distance,
    distance_mahalanobis,
)
from pyriemann.utils.base import logm
from pyriemann.utils.geodesic import geodesic
from pyriemann.utils.test import is_sym


def get_dist_func():
    dist_func = [
        distance_chol,
        distance_euclid,
        distance_harmonic,
        distance_kullback,
        distance_kullback_right,
        distance_kullback_sym,
        distance_logchol,
        distance_logdet,
        distance_logeuclid,
        distance_riemann,
        distance_wasserstein,
    ]
    for df in dist_func:
        yield df


def callable_sp_euclidean(A, B, squared=False):
    return euclidean(A.flatten(), B.flatten())


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize(
    "metric, dist",
    [
        ("chol", distance_chol),
        ("euclid", distance_euclid),
        ("harmonic", distance_harmonic),
        ("kullback", distance_kullback),
        ("kullback_right", distance_kullback_right),
        ("kullback_sym", distance_kullback_sym),
        ("logchol", distance_logchol),
        ("logdet", distance_logdet),
        ("logeuclid", distance_logeuclid),
        ("riemann", distance_riemann),
        ("wasserstein", distance_wasserstein),
        (callable_sp_euclidean, distance_euclid),
    ],
)
def test_distances_metric(kind, metric, dist, get_mats):
    """Test distance for metric"""
    n_matrices, n_channels = 2, 3
    mats = get_mats(n_matrices, n_channels, kind)
    A, B = mats[0], mats[1]
    d = distance(A, B, metric=metric)
    assert d == approx(dist(A, B))
    assert np.isreal(d)


def test_distances_metric_error(get_mats):
    n_matrices, n_channels = 2, 2
    A = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):
        distance(A[0], A[1], metric="universe")
    with pytest.raises(ValueError):
        distance(A[0], A[1], metric=42)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("dist", get_dist_func())
def test_distances_squared(kind, dist, get_mats):
    n_matrices, n_channels = 2, 5
    mats = get_mats(n_matrices, n_channels, kind)
    A, B = mats[0], mats[1]
    assert dist(A, B, squared=True) == approx(dist(A, B) ** 2)


@pytest.mark.parametrize("dist", get_dist_func())
def test_distances_all_error(dist, get_mats):
    n_matrices, n_channels = 3, 3
    A = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):
        dist(A, A[0])


@pytest.mark.parametrize("dist", get_dist_func())
def test_distances_all_ndarray(dist, get_mats):
    n_matrices, n_channels = 5, 3
    A = get_mats(n_matrices, n_channels, "spd")
    B = get_mats(n_matrices, n_channels, "spd")
    assert isinstance(dist(A[0], B[0]), float)  # 2D arrays
    assert dist(A, B).shape == (n_matrices,)  # 3D arrays

    n_sets = 5
    C = np.asarray([A for _ in range(n_sets)])
    D = np.asarray([B for _ in range(n_sets)])
    assert dist(C, D).shape == (n_sets, n_matrices,)  # 4D arrays


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("dist", get_dist_func())
def test_distances_all_geodesic(kind, dist, get_mats):
    n_matrices, n_channels = 2, 6
    mats = get_mats(n_matrices, n_channels, kind)
    A, C = mats[0], mats[1]
    B = geodesic(A, C, alpha=0.5)
    assert dist(A, B) < dist(A, C)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("dist", get_dist_func())
def test_distances_all_separability(kind, dist, get_mats):
    n_matrices, n_channels = 1, 6
    mats = get_mats(n_matrices, n_channels, kind)
    assert dist(mats[0], mats[0]) == approx(0, abs=2e-7)
    assert dist(np.eye(n_channels), np.eye(n_channels)) == approx(0)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize(
    "dist", [
        distance_chol,
        distance_euclid,
        distance_harmonic,
        distance_kullback_sym,
        distance_logchol,
        distance_logdet,
        distance_logeuclid,
        distance_riemann,
        distance_wasserstein,
    ]
)
def test_distances_all_symmetry(kind, dist, get_mats):
    n_matrices, n_channels = 2, 5
    mats = get_mats(n_matrices, n_channels, kind)
    A, B = mats[0], mats[1]
    assert dist(A, B) == approx(dist(B, A))


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("dist", get_dist_func())
def test_distances_all_triangle_inequality(kind, dist, get_mats):
    n_matrices, n_channels = 3, 4
    mats = get_mats(n_matrices, n_channels, kind)
    A, B, C = mats[0], mats[1], mats[2]
    assert dist(A, B) <= dist(A, C) + dist(C, B)


@pytest.mark.parametrize("complex_valued", [True, False])
def test_distance_euclid(rndstate, complex_valued):
    """Test Euclidean distance for generic matrices"""
    n_matrices, n_dim0, n_dim1 = 2, 3, 4
    mats = rndstate.randn(n_matrices, n_dim0, n_dim1)
    if complex_valued:
        mats = mats + 1j * rndstate.randn(n_matrices, n_dim0, n_dim1)
    A, B = mats[0], mats[1]
    distance_euclid(A, B)


@pytest.mark.parametrize("kind", ["real", "comp"])
def test_distance_harmonic(kind, get_mats):
    """Test harmonic distance for invertible matrices"""
    n_matrices, n_channels = 2, 5
    mats = get_mats(n_matrices, n_channels, kind)
    A, B = mats[0], mats[1]
    distance_harmonic(A, B)


def test_distance_kullback_implementation(get_mats):
    n_matrices, n_channels = 2, 6
    mats = get_mats(n_matrices, n_channels, "spd")
    A, B = mats[0], mats[1]
    d = 0.5*(np.trace(np.linalg.inv(B) @ A) - n_channels
             + np.log(np.linalg.det(B) / np.linalg.det(A)))
    assert distance_kullback(A, B) == approx(d)


def test_distance_logdet_implementation(get_mats):
    n_matrices, n_channels = 2, 6
    mats = get_mats(n_matrices, n_channels, "spd")
    A, B = mats[0], mats[1]
    d = np.sqrt(np.log(np.linalg.det((A + B) / 2.0))
                - 0.5 * np.log(np.linalg.det(A)*np.linalg.det(B)))
    assert distance_logdet(A, B) == approx(d)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_distance_poweuclid(kind, get_mats):
    n_matrices, n_channels = 2, 4
    mats = get_mats(n_matrices, n_channels, kind)
    A, B = mats[0], mats[1]
    assert distance_poweuclid(A, B, 1) == approx(distance_euclid(A, B))
    assert distance_poweuclid(A, B, 0) == approx(distance_logeuclid(A, B))
    assert distance_poweuclid(A, B, -1) == approx(distance_harmonic(A, B))
    distance_poweuclid(A, B, 0.42)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_distance_riemann_properties(kind, get_mats):
    n_channels = 6
    mats = get_mats(2, n_channels, kind)
    A, B = mats[0], mats[1]
    dist_AB = distance_riemann(A, B)

    # exponential metric increasing property, Eq(6.8) in [Bhatia2007]
    assert dist_AB >= np.linalg.norm(logm(A) - logm(B))

    # invariance under inversion
    assert dist_AB == approx(
        distance_riemann(np.linalg.inv(A), np.linalg.inv(B))
    )

    # congruence-invariance
    W = np.random.normal(size=(n_channels, n_channels))  # must be invertible
    WAW, WBW = W @ A @ W.T, W @ B @ W.T
    assert dist_AB == approx(distance_riemann(WAW, WBW))

    # proportionality, Eq(6.12) in [Bhatia2007]
    alpha = np.random.uniform()
    dist_1 = distance_riemann(A, geodesic(A, B, alpha, metric="riemann"))
    dist_2 = alpha * distance_riemann(A, B)
    assert dist_1 == approx(dist_2)


@pytest.mark.parametrize("dist, dfunc", zip(get_distances(), get_dist_func()))
def test_distance_wrapper(dist, dfunc, get_mats):
    n_matrices, n_channels = 2, 5
    mats = get_mats(n_matrices, n_channels, "spd")
    A, B = mats[0], mats[1]
    assert distance(A, B, metric=dist) == dfunc(A, B)


@pytest.mark.parametrize("dist", get_dist_func())
def test_distance_wrapper_between_set_and_matrix(dist, get_mats):
    n_matrices, n_channels = 10, 4
    mats = get_mats(n_matrices, n_channels, "spd")
    assert distance(mats, mats[-1], metric=dist).shape == (n_matrices, 1)

    n_sets = 5
    mats_4d = np.asarray([mats for _ in range(n_sets)])
    with pytest.raises(ValueError):
        distance(mats_4d, mats, metric=dist)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("dist", get_distances())
@pytest.mark.parametrize("Y", [None, True])
@pytest.mark.parametrize("squared", [False, True])
def test_pairwise_distance_matrix(kind, dist, Y, squared, get_mats):
    n_matrices_X, n_matrices_Y, n_channels = 6, 4, 5
    X = get_mats(n_matrices_X, n_channels, kind)
    if Y is None:
        n_matrices_Y = n_matrices_X
        Y_ = X
    else:
        Y = get_mats(n_matrices_Y, n_channels, kind)
        Y_ = Y

    pdist = pairwise_distance(X, Y, metric=dist, squared=squared)
    assert pdist.shape == (n_matrices_X, n_matrices_Y)

    for i in range(n_matrices_X):
        for j in range(n_matrices_Y):
            assert np.isclose(
                pdist[i, j],
                distance(X[i], Y_[j], metric=dist, squared=squared),
                atol=1e-5,
                rtol=1e-5,
            )

    if Y is None and dist not in ["kullback", "kullback_right"]:
        assert is_sym(pdist)
    else:
        assert not is_sym(pdist)


@pytest.mark.parametrize("complex_valued", [True, False])
def test_distance_mahalanobis(rndstate, complex_valued):
    n_channels, n_times = 3, 100
    X = rndstate.randn(n_channels, n_times)
    if complex_valued:
        X = X + 1j * rndstate.randn(n_channels, n_times)
    d = distance_mahalanobis(X, np.cov(X))
    assert d.shape == (n_times,)
    assert np.all(np.isreal(d))


@pytest.mark.parametrize("mean", [True, None])
def test_distance_mahalanobis_scipy(rndstate, get_mats, mean):
    """Test equivalence between pyriemann and scipy for real data"""
    n_channels, n_times = 3, 100
    X = rndstate.randn(n_channels, n_times)
    C = get_mats(1, n_channels, "spd")[0]

    Cinv = np.linalg.inv(C)
    y = np.zeros(n_channels)
    dist_sp = [mahalanobis(x, y, Cinv) for x in X.T]

    if mean:
        mean = np.zeros((n_channels, 1))
    else:
        mean = None
    dist_pr = distance_mahalanobis(X, C, mean=mean)

    assert_array_almost_equal(dist_sp, dist_pr)
