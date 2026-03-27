from functools import partial

import numpy as np
import pytest
from scipy.linalg import eigvalsh
from scipy.spatial.distance import euclidean, mahalanobis

from conftest import approx, assert_array_almost_equal
from pyriemann.utils._backend import get_namespace, xpd as device
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
    distance_thompson,
    distance_wasserstein,
    distance,
    pairwise_distance,
    distance_mahalanobis,
)
from pyriemann.utils.base import logm, invsqrtm
from pyriemann.utils.geodesic import geodesic
from pyriemann.utils.test import is_sym


dists = [
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
    distance_thompson,
    distance_wasserstein,
]


BROADCAST_DISTANCE_FUNCS = {
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
    distance_thompson,
    distance_wasserstein,
}


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
        ("thompson", distance_thompson),
        ("wasserstein", distance_wasserstein),
        pytest.param(
            callable_sp_euclidean, distance_euclid,
            marks=pytest.mark.numpy_only,
        ),
    ],
)
def test_distance_metric(kind, metric, dist, get_mats):
    n_channels = 3
    A, B = get_mats(2, n_channels, kind)
    d = distance(A, B, metric=metric)
    assert d == approx(dist(A, B))


def test_distance_metric_error(get_mats):
    n_channels = 2
    A = get_mats(2, n_channels, "spd")
    with pytest.raises(ValueError):
        distance(A[0], A[1], metric="universe")
    with pytest.raises(ValueError):
        distance(A[0], A[1], metric=42)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("dist", dists)
def test_distance_squared(kind, dist, get_mats):
    n_channels = 5
    A, B = get_mats(2, n_channels, kind)
    assert dist(A, B, squared=True) == approx(dist(A, B) ** 2)


@pytest.mark.parametrize("dist", dists)
def test_distance_between_set_and_matrix(dist, get_mats):
    n_matrices, n_channels = 10, 4
    X = get_mats(n_matrices, n_channels, "spd")
    xp = get_namespace(X)

    if dist in BROADCAST_DISTANCE_FUNCS:
        assert dist(X, X[-1]).shape == (n_matrices,)
    else:
        with pytest.raises(ValueError):
            dist(X, X[-1])

    assert distance(X, X[-1], metric=dist).shape == (n_matrices, 1)

    n_sets = 5
    X_4d = xp.stack([X] * n_sets, axis=0)
    with pytest.raises(ValueError):
        distance(X_4d, X, metric=dist)


@pytest.mark.numpy_only
@pytest.mark.parametrize("dist", dists)
def test_distance_broadcasting(dist, get_mats):
    n_dim5, n_dim4, n_matrices, n_channels = 7, 5, 3, 4
    A = get_mats([n_dim5, n_dim4, n_matrices], n_channels, "spd")
    B = get_mats([n_dim5, n_dim4, n_matrices], n_channels, "spd")

    # 2D array
    d2 = dist(A[0, 0, 0], B[0, 0, 0])
    assert isinstance(d2, float)

    # 3D array
    D3 = dist(A[0, 0], B[0, 0])
    assert D3.shape == (n_matrices,)
    assert D3[0] == d2

    # 4D array
    D4 = dist(A[0], B[0])
    assert D4.shape == (n_dim4, n_matrices)
    assert D4[0, 0] == d2

    # 5D array
    D5 = dist(A, B)
    assert D5.shape == (n_dim5, n_dim4, n_matrices)
    assert D5[0, 0, 0] == d2


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("dist", dists)
def test_distance_property_geodesic(kind, dist, get_mats):
    n_channels = 6
    A, C = get_mats(2, n_channels, kind)
    B = geodesic(A, C, alpha=0.5)
    assert dist(A, B) < dist(A, C)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("dist", dists)
def test_distance_property_separability(kind, dist, get_mats):
    n_channels = 5
    A = get_mats(1, n_channels, kind)[0]
    xp = get_namespace(A)
    assert dist(A, A) == approx(0, abs=2e-7)
    Id = xp.eye(n_channels, dtype=A.dtype, device=device(A))
    assert dist(Id, Id) == approx(0)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("dist", [
    distance_chol,
    distance_euclid,
    distance_harmonic,
    distance_kullback_sym,
    distance_logchol,
    distance_logdet,
    distance_logeuclid,
    pytest.param(partial(distance_poweuclid, p=0.5), id="distance_poweuclid"),
    distance_riemann,
    distance_thompson,
    distance_wasserstein,
])
def test_distance_property_symmetry(kind, dist, get_mats):
    n_channels = 5
    A, B = get_mats(2, n_channels, kind)
    assert dist(A, B) == approx(dist(B, A))


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("dist", dists)
def test_distance_property_triangle_inequality(kind, dist, get_mats):
    n_channels = 4
    A, B, C = get_mats(3, n_channels, kind)
    assert dist(A, B) <= dist(A, C) + dist(C, B)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("dist", [
    distance_logeuclid,  # Th 3.6 in [Arsigny2007]
    distance_riemann,  # Th 1 (v) of [Forstner2003]
    distance_thompson,  # Eq(4.7a) in [Sra2015]
])
def test_distance_property_invariance_under_inversion(kind, dist, get_mats):
    n_channels = 4
    A, B = get_mats(2, n_channels, kind)
    xp = get_namespace(A)
    assert dist(A, B) == approx(dist(xp.linalg.inv(A), xp.linalg.inv(B)))


@pytest.mark.parametrize("kind, kindQ", [("spd", "orth"), ("hpd", "unit")])
@pytest.mark.parametrize("dist", [
    distance_euclid,
    distance_harmonic,
    distance_kullback,
    distance_logeuclid,
    distance_riemann,
    distance_thompson,
    distance_wasserstein,
])
def test_distance_property_invariance_rotation(kind, kindQ, dist, get_mats):
    n_channels = 4
    A, B = get_mats(2, n_channels, kind)
    Q = get_mats(1, n_channels, kindQ)[0]
    QAQ, QBQ = Q @ A @ Q.conj().T, Q @ B @ Q.conj().T
    assert dist(A, B) == approx(dist(QAQ, QBQ))


@pytest.mark.parametrize("kind, kindQ", [("spd", "orth"), ("hpd", "unit")])
@pytest.mark.parametrize("dist", [
    distance_logeuclid,  # Prop 3.11 in [Arsigny2007]
    distance_riemann,
    distance_thompson,
])
def test_distance_property_invariance_similarity(kind, kindQ, dist,
                                                 get_mats, rndstate):
    """Test invariance by similarity, ie a scale and a rotation"""
    n_channels = 5
    A, B = get_mats(2, n_channels, kind)
    Q = get_mats(1, n_channels, kindQ)[0]
    scale = rndstate.uniform(0.01, 10.0)
    sQAQ, sQBQ = scale * Q @ A @ Q.conj().T, scale * Q @ B @ Q.conj().T
    assert dist(A, B) == approx(dist(sQAQ, sQBQ))


@pytest.mark.parametrize("kind, kindW", [("spd", "inv"), ("hpd", "cinv")])
@pytest.mark.parametrize("dist", [
    distance_kullback,
    distance_riemann,  # Th 1 (iv) of [Forstner2003]
    distance_thompson,  # Eq(4.7b) in [Sra2015]
])
def test_distance_property_invariance_congruence(kind, kindW, dist, get_mats):
    """Test invariance under congruence, ie an invertible transform"""
    n_channels = 4
    A, B = get_mats(2, n_channels, kind)
    W = get_mats(1, n_channels, kindW)[0]
    WAW, WBW = W @ A @ W.conj().T, W @ B @ W.conj().T
    assert dist(A, B) == approx(dist(WAW, WBW))


@pytest.mark.numpy_only
@pytest.mark.parametrize("n_dim1, n_dim2", [(4, 5), (5, 4)])
@pytest.mark.parametrize("kind", ["real", "comp"])
def test_distance_euclid(n_dim1, n_dim2, kind, get_mats):
    """Euclidean distance between non-square matrices"""
    A, B = get_mats(2, [n_dim1, n_dim2], kind)
    assert distance_euclid(A, B) == approx(euclidean(A.flatten(), B.flatten()))


@pytest.mark.parametrize("kind", ["inv", "cinv"])
def test_distance_harmonic(kind, get_mats):
    """harmonic distance between invertible matrices"""
    n_channels = 5
    A, B = get_mats(2, n_channels, kind)
    distance_harmonic(A, B)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_distance_kullback_implementation(kind, get_mats):
    n_channels = 6
    A, B = get_mats(2, n_channels, kind)
    xp = get_namespace(A)
    d = 0.5*(xp.linalg.trace(xp.linalg.inv(B) @ A) - n_channels
             + xp.log(xp.linalg.det(B) / xp.linalg.det(A)))
    assert distance_kullback(A, B) == approx(d)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_distance_logdet_implementation(kind, get_mats):
    n_channels = 6
    A, B = get_mats(2, n_channels, kind)
    xp = get_namespace(A)
    d = xp.sqrt(xp.log(xp.linalg.det((A + B) / 2.0))
                - 0.5 * xp.log(xp.linalg.det(A) * xp.linalg.det(B)))
    assert distance_logdet(A, B) == approx(d)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_distance_poweuclid(kind, get_mats):
    n_channels = 4
    A, B = get_mats(2, n_channels, kind)
    assert distance_poweuclid(A, B, 1) == approx(distance_euclid(A, B))
    assert distance_poweuclid(A, B, 0) == approx(distance_logeuclid(A, B))
    assert distance_poweuclid(A, B, -1) == approx(distance_harmonic(A, B))
    distance_poweuclid(A, B, 0.42)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_distance_riemann_implementations(kind, get_mats):
    n_channels = 6
    A, B = get_mats(2, n_channels, kind)
    xp = get_namespace(A)
    d = distance_riemann(A, B)

    # Eq(6.13) in [Bhatia2007]
    Bm12 = invsqrtm(B)
    d1 = xp.linalg.norm(logm(Bm12 @ A @ Bm12), ord="fro")
    assert d1 == approx(d)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_distance_riemann_properties(kind, get_mats, rndstate):
    n_channels = 6
    A, B = get_mats(2, n_channels, kind)
    xp = get_namespace(A)
    dist_AB = distance_riemann(A, B)

    # exponential metric increasing property, Eq(6.8) in [Bhatia2007]
    assert float(dist_AB) >= float(xp.linalg.norm(logm(A) - logm(B)))

    # proportionality, Eq(6.12) in [Bhatia2007]
    alpha = rndstate.uniform(0.01, 10.0)
    dist_1 = distance_riemann(A, geodesic(A, B, alpha, metric="riemann"))
    dist_2 = alpha * distance_riemann(A, B)
    assert dist_1 == approx(dist_2)


@pytest.mark.numpy_only
@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_distance_thompson_implementation(kind, get_mats):
    n_channels = 5
    A, B = get_mats(2, n_channels, kind)
    d = distance_thompson(A, B)

    # Eq(4.6) in [Sra2015]
    Bm12 = invsqrtm(B)
    assert d == approx(np.linalg.norm(logm(Bm12 @ A @ Bm12), ord=2))

    # Eq(1.2) in [Mostajeran2024]
    d2 = np.log(max(eigvalsh(A, B).max(), 1 / eigvalsh(A, B).min()))
    assert d == approx(d2)

    # Eq(1.4) in [Mostajeran2024]
    d2 = np.log(max(eigvalsh(A, B).max(), eigvalsh(B, A).max()))
    assert d == approx(d2)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", [
    "chol",
    "euclid",
    "harmonic",
    "kullback",
    "kullback_right",
    "kullback_sym",
    "logchol",
    "logdet",
    "logeuclid",
    "riemann",
    "thompson",
    "wasserstein",
])
@pytest.mark.parametrize("Y", [None, True])
@pytest.mark.parametrize("squared", [False, True])
def test_pairwise_distance(kind, metric, Y, squared, get_mats):
    n_matrices_X, n_matrices_Y, n_channels = 6, 4, 5
    X = get_mats(n_matrices_X, n_channels, kind)
    if Y is None:
        n_matrices_Y = n_matrices_X
        Y_ = X
    else:
        Y = get_mats(n_matrices_Y, n_channels, kind)
        Y_ = Y

    pdist = pairwise_distance(X, Y, metric=metric, squared=squared)
    assert pdist.shape == (n_matrices_X, n_matrices_Y)

    for i in range(n_matrices_X):
        for j in range(n_matrices_Y):
            assert np.isclose(
                float(pdist[i, j]),
                float(distance(X[i], Y_[j], metric=metric, squared=squared)),
                atol=1e-5,
                rtol=1e-5,
            )

    if Y is None and metric not in ["kullback", "kullback_right"]:
        assert is_sym(pdist)
    else:
        assert not is_sym(pdist)


@pytest.mark.numpy_only
@pytest.mark.parametrize("kind", ["real", "comp"])
def test_distance_mahalanobis(kind, get_mats):
    n_channels, n_times = 2, 50
    X = get_mats(1, [n_channels, n_times], kind)[0]
    d = distance_mahalanobis(X, np.cov(X))
    assert d.shape == (n_times,)
    assert np.all(np.isreal(d))


@pytest.mark.numpy_only
@pytest.mark.parametrize("mean", [True, None])
def test_distance_mahalanobis_scipy(mean, get_mats):
    """Test equivalence between pyriemann and scipy for real data"""
    n_channels, n_times = 3, 100
    X = get_mats(1, [n_channels, n_times], "real")[0]
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


@pytest.mark.parametrize("mean", [True, None])
def test_distance_mahalanobis_broadcasting(mean, get_mats, rndstate):
    n_dim5, n_dim4, n_dim3, n_channels, n_vectors = 2, 5, 3, 4, 10
    cov = get_mats([n_dim5, n_dim4, n_dim3], n_channels, "spd")
    X = rndstate.randn(n_dim5, n_dim4, n_dim3, n_channels, n_vectors)

    if mean is True:
        m = rndstate.randn(n_dim5, n_dim4, n_dim3, n_channels, 1)
    else:
        m = None

    # 2D array
    d2 = distance_mahalanobis(
        X[0, 0, 0], cov[0, 0, 0],
        mean=m[0, 0, 0] if m is not None else None,
    )
    assert d2.shape == (n_vectors,)

    # 3D array
    D3 = distance_mahalanobis(
        X[0, 0], cov[0, 0],
        mean=m[0, 0] if m is not None else None,
    )
    assert D3.shape == (n_dim3, n_vectors)
    assert_array_almost_equal(D3[0], d2)

    # 4D array
    D4 = distance_mahalanobis(
        X[0], cov[0],
        mean=m[0] if m is not None else None,
    )
    assert D4.shape == (n_dim4, n_dim3, n_vectors)
    assert_array_almost_equal(D4[0, 0], d2)

    # 5D array
    D5 = distance_mahalanobis(
        X, cov,
        mean=m,
    )
    assert D5.shape == (n_dim5, n_dim4, n_dim3, n_vectors)
    assert_array_almost_equal(D5[0, 0, 0], d2)
