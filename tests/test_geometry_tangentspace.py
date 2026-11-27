from array_api_compat import array_namespace as get_namespace
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
from scipy.linalg import solve_continuous_lyapunov

from conftest import approx, to_numpy
from pyriemann.geometry.distance import distance, distance_riemann
from pyriemann.geometry.geodesic import geodesic
from pyriemann.geometry.mean import mean_riemann
from pyriemann.geometry.tangentspace import (
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
    vec_upper,
    vec_herm,
    unvec_upper,
    unvec_herm,
    tangent_space,
    untangent_space,
    innerproduct,
    innerproduct_euclid,
    innerproduct_logchol,
    innerproduct_logeuclid,
    innerproduct_riemann,
    innerproduct_wasserstein,
    norm,
    transport,
    transport_euclid,
    transport_logchol,
    transport_logeuclid,
    transport_riemann,
    transport_wasserstein,
)
from pyriemann.geometry.test import is_hermitian, is_real, is_real_type
from pyriemann.spatialfilters import Whitening

metrics = ["euclid", "logchol", "logeuclid", "riemann", "wasserstein"]


@pytest.mark.parametrize(
    "fmap", [
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
def test_maps_broadcasting(fmap, get_mats):
    n_dim5, n_dim4, n_matrices, n_channels = 2, 6, 5, 3
    X = get_mats([n_dim5, n_dim4, n_matrices], n_channels, "spd")
    Cref = get_mats(1, n_channels, "spd")[0]

    # 2D array
    F2 = fmap(X[0, 0, 0], Cref)
    assert F2.shape == (n_channels, n_channels)

    # 3D array
    F3 = fmap(X[0, 0], Cref)
    assert F3.shape == (n_matrices, n_channels, n_channels)
    assert F3[0] == approx(F2)

    # 4D array
    F4 = fmap(X[0], Cref)
    assert F4.shape == (n_dim4, n_matrices, n_channels, n_channels)
    assert F4[0, 0] == approx(F2)

    # 5D array
    F5 = fmap(X, Cref)
    assert F5.shape == (n_dim5, n_dim4, n_matrices, n_channels, n_channels)
    assert F5[0, 0, 0] == approx(F2)


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
    "log_map_, exp_map_", [
        (log_map_euclid, exp_map_euclid),
        (log_map_logchol, exp_map_logchol),
        (log_map_logeuclid, exp_map_logeuclid),
        (log_map_riemann, exp_map_riemann),
        (log_map_wasserstein, exp_map_wasserstein),
    ]
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
    X, C = mats[:n_matrices - 1], mats[-1]
    assert exp_map_euclid(log_map_euclid(X, C), C) == approx(X)


@pytest.mark.parametrize("kind", ["sym", "herm"])
@pytest.mark.parametrize("n_channels", [3, 6, 9])
@pytest.mark.parametrize("vec", [vec_upper, vec_herm])
def test_vec(kind, n_channels, vec, get_mats):
    if vec == vec_herm and kind == "sym":
        return
    n_matrices = 5
    X = get_mats(n_matrices, n_channels, kind)
    T = vec(X)

    if vec == vec_herm:
        assert T.shape == (n_matrices, n_channels * n_channels)
        assert is_real_type(T)
    else:
        assert T.shape == (n_matrices, n_channels * (n_channels + 1) // 2)
        assert T.dtype == X.dtype

    # vec is an isometry preserving the Frobenius inner product Re(tr(X0^H X1))
    xp = get_namespace(X)
    prod1 = xp.real(xp.einsum("...nm,...nm->...", xp.conj(X[0]), X[1]))
    prod2 = xp.real(xp.conj(T[0]) @ T[1])
    assert prod1 == approx(prod2)


@pytest.mark.parametrize(
    "vec, unvec", [(vec_upper, unvec_upper), (vec_herm, unvec_herm)]
)
def test_vec_and_unvec_equal_identity(vec, unvec, get_mats):
    n_dim5, n_dim4, n_matrices, n_channels = 7, 6, 5, 4
    X = get_mats([n_dim5, n_dim4, n_matrices], n_channels, "herm")

    # 2D array
    U2 = unvec(vec(X[0, 0, 0]))
    assert U2 == approx(X[0, 0, 0])

    # 5D array
    U5 = unvec(vec(X))
    assert U5 == approx(X)
    assert U5[0, 0, 0] == approx(U2)


@pytest.mark.parametrize("vec", [vec_upper, vec_herm])
@pytest.mark.parametrize("metric", metrics)
def test_tangent_space(vec, metric, get_mats):
    n_matrices, n_channels = 5, 4
    X = get_mats(n_matrices, n_channels, "spd")
    Cref = get_mats(1, n_channels, "spd")[0]

    T = tangent_space(X, Cref, metric=metric, vec=vec)
    if vec == vec_upper:
        n_ts = (n_channels * (n_channels + 1)) // 2
    else:
        n_ts = n_channels * n_channels
    assert T.shape == (n_matrices, n_ts)


@pytest.mark.parametrize("metric", metrics)
def test_tangent_space_broadcasting(metric, get_mats):
    n_dim5, n_dim4, n_matrices, n_channels = 4, 6, 5, 3
    n_ts = (n_channels * (n_channels + 1)) // 2
    X = get_mats([n_dim5, n_dim4, n_matrices], n_channels, "spd")
    Cref = get_mats(1, n_channels, "spd")[0]

    # 2D array
    T2 = tangent_space(X[0, 0, 0], Cref, metric=metric)
    assert T2.shape == (n_ts,)

    # 3D array
    T3 = tangent_space(X[0, 0], Cref, metric=metric)
    assert T3.shape == (n_matrices, n_ts)
    assert T3[0] == approx(T2)

    # 4D array
    T4 = tangent_space(X[0], Cref, metric=metric)
    assert T4.shape == (n_dim4, n_matrices, n_ts)
    assert T4[0, 0] == approx(T2)

    # 5D array
    T5 = tangent_space(X, Cref, metric=metric)
    assert T5.shape == (n_dim5, n_dim4, n_matrices, n_ts)
    assert T5[0, 0, 0] == approx(T2)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_tangent_space_riemann_properties(kind, get_mats):
    n_channels = 3
    A, B = get_mats(2, n_channels, kind)

    # equivalent definitions of Riemannian distance, Eq(7) in [Barachant2012]
    dist = distance_riemann(A, B)
    s = tangent_space(A, B, metric="riemann")
    assert dist == approx(np.linalg.norm(to_numpy(s)))


@pytest.mark.parametrize("metric", metrics)
def test_untangent_space_broadcasting(metric, get_mats):
    n_dim5, n_dim4, n_matrices, n_channels = 4, 6, 10, 3
    n_ts = (n_channels * (n_channels + 1)) // 2
    X = get_mats(n_dim5, [n_dim4, n_matrices, n_ts], "real")
    Cref = get_mats(1, n_channels, "spd")[0]

    # 2D array
    U2 = untangent_space(X[0, 0, 0], Cref, metric=metric)
    assert U2.shape == (n_channels, n_channels)

    # 3D array
    U3 = untangent_space(X[0, 0], Cref, metric=metric)
    assert U3.shape == (n_matrices, n_channels, n_channels)
    assert U3[0] == approx(U2)

    # 4D array
    U4 = untangent_space(X[0], Cref, metric=metric)
    assert U4.shape == (n_dim4, n_matrices, n_channels, n_channels)
    assert U4[0, 0] == approx(U2)

    # 5D array
    U5 = untangent_space(X, Cref, metric=metric)
    assert U5.shape == (n_dim5, n_dim4, n_matrices, n_channels, n_channels)
    assert U5[0, 0, 0] == approx(U2)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", metrics)
def test_tangent_and_untangent_space(kind, metric, get_mats):
    """Tangent space projection then back-projection should be identity"""
    n_matrices, n_channels = 10, 3
    mats = get_mats(n_matrices, n_channels, kind)
    X, C = mats[:n_matrices - 1], mats[-1]
    X_t = tangent_space(X, C, metric=metric)
    X_ut = untangent_space(X_t, C, metric=metric)
    assert X_ut == approx(X)


###############################################################################

metrics = ["euclid", "logchol", "logeuclid", "riemann", "wasserstein"]


@pytest.mark.parametrize("metric", metrics)
def test_innerproduct_build(metric, get_mats):
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "sym")
    Y = get_mats(n_matrices, n_channels, "sym")
    Cref = get_mats(1, n_channels, "spd")[0]
    K = innerproduct(X, Y, Cref, metric=metric)
    assert_array_equal(K, globals()[f"innerproduct_{metric}"](X, Y, Cref))


def test_innerproduct_metric_string_error(get_mats):
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "sym")
    with pytest.raises(ValueError):
        innerproduct(X, X, X[0], metric="foo")


@pytest.mark.parametrize("metric", metrics)
def test_innerproduct_input_dimension_error(metric, get_mats):
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "sym")
    Y = get_mats(n_matrices, n_channels + 1, "sym")
    Cref = get_mats(1, n_channels + 2, "spd")[0]
    with pytest.raises((ValueError, RuntimeError)):
        innerproduct(X, Y, Cref, metric=metric)


@pytest.mark.parametrize("kindX, kindC", [("sym", "spd"), ("herm", "hpd")])
@pytest.mark.parametrize("metric", metrics)
def test_innerproduct_x_x(kindX, kindC, metric, get_mats):
    """Test innerproduct for X = Y"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, kindX)
    Cref = get_mats(1, n_channels, kindC)[0]
    G = innerproduct(X, X, Cref, metric=metric)
    assert G.shape == (n_matrices,)
    assert is_real(G)

    G1 = innerproduct(X, None, Cref, metric=metric)
    assert_array_equal(G, G1)


@pytest.mark.parametrize("kindX, kindC", [("sym", "spd"), ("herm", "hpd")])
@pytest.mark.parametrize("metric", metrics)
def test_innerproduct_x_y(kindX, kindC, metric, get_mats):
    """Test innerproduct for different X and Y"""
    n_matrices, n_channels = 4, 3
    X = get_mats(n_matrices, n_channels, kindX)
    Y = get_mats(n_matrices, n_channels, kindX)
    Cref = get_mats(1, n_channels, kindC)[0]
    G = innerproduct(X, Y, Cref, metric=metric)
    assert G.shape == (n_matrices,)
    assert is_real(G)


@pytest.mark.parametrize(
    "finnerproduct",
    [
        innerproduct_euclid,
        innerproduct_logchol,
        innerproduct_logeuclid,
        innerproduct_riemann,
        innerproduct_wasserstein,
    ],
)
def test_innerproduct_broadcasting(finnerproduct, backend, get_mats):
    n_matrices, n_channels = 5, 3
    A = get_mats(n_matrices, n_channels, "sym")
    B = get_mats(n_matrices, n_channels, "sym")
    Cref = get_mats(1, n_channels, "spd")[0]
    xp = get_namespace(A)

    if backend == "numpy":
        assert isinstance(finnerproduct(A[0], B[0], Cref), float)  # 2D arrays

    assert finnerproduct(A, B, Cref).shape == (n_matrices,)  # 3D arrays

    n_sets = 4
    C = xp.stack([A for _ in range(n_sets)])
    D = xp.stack([B for _ in range(n_sets)])
    assert finnerproduct(C, D, Cref).shape == (n_sets, n_matrices)  # 4D arrays


@pytest.mark.parametrize("kindX, kindC", [("sym", "spd"), ("herm", "hpd")])
@pytest.mark.parametrize("metric", metrics)
def test_innerproduct_property_distance(kindX, kindC, metric, get_mats):
    """Test d(C,exp_C(X))^2 = g_C(X,X) locally"""
    n_matrices, n_channels = 5, 3
    X = 0.1 * get_mats(n_matrices, n_channels, kindX)
    Cref = get_mats(1, n_channels, kindC)[0]
    G = innerproduct(X, X, Cref, metric=metric)

    Xexp = exp_map(X, Cref, metric=metric, Cm12=True)
    D2 = distance(Xexp, Cref, metric=metric, squared=True)
    assert_array_almost_equal(D2.flatten(), G)


@pytest.mark.parametrize("kindX, kindC", [("sym", "spd"), ("herm", "hpd")])
@pytest.mark.parametrize("metric", metrics)
def test_innerproduct_property_conjsymmetry(kindX, kindC, metric, get_mats):
    n_matrices, n_channels = 5, 2
    X = get_mats(n_matrices, n_channels, kindX)
    Y = get_mats(n_matrices, n_channels, kindX)
    Cref = get_mats(1, n_channels, kindC)[0]
    G1 = innerproduct(X, Y, Cref, metric=metric)
    G2 = innerproduct(Y, X, Cref, metric=metric)
    assert_array_almost_equal(G1.conj(), G2)


@pytest.mark.parametrize("kindX, kindC", [("sym", "spd"), ("herm", "hpd")])
@pytest.mark.parametrize("metric", metrics)
def test_innerproduct_property_linearity(kindX, kindC, metric,
                                         get_mats, rndstate):
    n_matrices, n_channels = 4, 3
    X = get_mats(n_matrices, n_channels, kindX)
    Y = get_mats(n_matrices, n_channels, kindX)
    Z = get_mats(n_matrices, n_channels, kindX)
    Cref = get_mats(1, n_channels, kindC)[0]
    a, b = rndstate.uniform(0.01, 0.99, size=2)

    Gxy = innerproduct(X, Y, Cref, metric=metric)
    Gxz = innerproduct(X, Z, Cref, metric=metric)
    Gyz = innerproduct(Y, Z, Cref, metric=metric)

    Gaxpbz = innerproduct(a * X + b * Y, Z, Cref, metric=metric)
    aGxzpbGyz = a * Gxz + b * Gyz
    assert_array_almost_equal(Gaxpbz, aGxzpbGyz.real)

    Gxaypbz = innerproduct(X, a * Y + b * Z, Cref, metric=metric)
    aGxypbGxz = a * Gxy + b * Gxz
    assert_array_almost_equal(Gxaypbz, aGxypbGxz.real)


@pytest.mark.parametrize("kindX, kindC", [("sym", "spd"), ("herm", "hpd")])
@pytest.mark.parametrize("metric", metrics)
def test_innerproduct_property_pos_def(kindX, kindC, metric, get_mats):
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, kindX)
    Cref = get_mats(1, n_channels, kindC)[0]
    G = innerproduct(X, None, Cref, metric=metric)
    xp = get_namespace(G)
    assert bool(xp.all(G > 0))


@pytest.mark.parametrize("kind", ["real", "comp"])
@pytest.mark.parametrize("n_dim1, n_dim2", [(4, 5), (5, 4)])
def test_innerproduct_euclid(kind, n_dim1, n_dim2, get_mats):
    """Euclidean inner-product for non-square matrices"""
    n_matrices = 3
    X = get_mats(n_matrices, [n_dim1, n_dim2], kind)
    Y = get_mats(n_matrices, [n_dim1, n_dim2], kind)
    G = innerproduct_euclid(X, Y)

    xp = get_namespace(X)
    G1 = np.empty((n_matrices,))
    G2 = np.empty((n_matrices,))
    for i in range(n_matrices):
        # tr(X^H @ Y)
        G1[i] = xp.real(xp.linalg.trace(xp.conj(X[i]).mT @ Y[i]))
        # vec(X)^H . vec(Y)
        G2[i] = xp.real(
            xp.vecdot(xp.reshape(X[i], (-1,)), xp.reshape(Y[i], (-1,)))
        )
    assert_array_almost_equal(G, G1)
    assert_array_almost_equal(G, G2)


@pytest.mark.parametrize("kindX, kindC", [("sym", "spd"), ("herm", "hpd")])
def test_innerproduct_riemann(kindX, kindC, get_mats):
    n_channels = 5
    X, Y = get_mats(2, n_channels, kindX)
    Cref = get_mats(1, n_channels, kindC)[0]
    G = innerproduct_riemann(X, Y, Cref)

    # Eq(2.6) in [Moakher2005]
    G1 = np.trace(np.linalg.solve(Cref, X) @ np.linalg.solve(Cref, Y))
    assert_array_almost_equal(G, G1)


@pytest.mark.parametrize("kindX, kindC", [("sym", "spd"), ("herm", "hpd")])
def test_innerproduct_wasserstein(kindX, kindC, get_mats):
    n_channels = 4
    X, Y = get_mats(2, n_channels, kindX)
    Cref = get_mats(1, n_channels, kindC)[0]
    G = innerproduct_wasserstein(X, Y, Cref)
    xp = get_namespace(X)

    # Eq(6) in [Malago2018]
    LX = xp.asarray(solve_continuous_lyapunov(Cref, X))
    LY = xp.asarray(solve_continuous_lyapunov(Cref, Y))
    G1 = xp.real(xp.trace(LX @ Cref @ LY))
    assert_array_almost_equal(G, G1)


@pytest.mark.parametrize("kindX, kindC", [("sym", "spd"), ("herm", "hpd")])
@pytest.mark.parametrize("metric", metrics)
def test_norm_properties(kindX, kindC, metric, backend, get_mats):
    n_channels = 4
    X, Y = get_mats(2, n_channels, kindX)
    Cref = get_mats(1, n_channels, kindC)[0]

    nx = norm(X, Cref, metric=metric)

    if backend == "numpy":
        assert isinstance(nx, float)

    # positivity
    assert nx >= 0

    # triangle inequality
    ny = norm(Y, Cref, metric=metric)
    assert norm(X + Y, Cref, metric=metric) <= nx + ny


###############################################################################


@pytest.mark.parametrize("ftransport", [
    transport_euclid,
    transport_logchol,
    transport_logeuclid,
    transport_riemann,
    transport_wasserstein,
])
def test_transport_broadcasting(ftransport, get_mats):
    n_dim5, n_dim4, n_matrices, n_channels = 2, 4, 7, 3
    X = get_mats([n_dim5, n_dim4, n_matrices], n_channels, "herm")
    A, B = get_mats(2, n_channels, "hpd")

    # 2D array
    T2 = ftransport(X[0, 0, 0], A, B)
    assert T2.shape == (n_channels, n_channels)

    # 3D array
    T3 = ftransport(X[0, 0], A, B)
    assert T3.shape == (n_matrices, n_channels, n_channels)
    assert T3[0] == approx(T2)

    # 4D array
    T4 = ftransport(X[0], A, B)
    assert T4.shape == (n_dim4, n_matrices, n_channels, n_channels)
    assert T4[0, 0] == approx(T2)

    # 5D array
    T5 = ftransport(X, A, B)
    assert T5.shape == (n_dim5, n_dim4, n_matrices, n_channels, n_channels)
    assert T5[0, 0, 0] == approx(T2)


@pytest.mark.parametrize("kindX, kindAB", [("sym", "spd"), ("herm", "hpd")])
@pytest.mark.parametrize("metric", [
    "logchol",
    "logeuclid",
    "riemann",
    "wasserstein",
])
def test_transport_properties(kindX, kindAB, metric, get_mats, rndstate):
    n_matrices, n_channels = 10, 3
    X = get_mats(n_matrices, n_channels, kindX)
    A, B, C = get_mats(3, n_channels, kindAB)

    Xt = transport(X, A, B, metric=metric)

    # trivial transport
    assert transport(X, A, A, metric=metric) == approx(X)

    # keep symmetry
    assert is_hermitian(Xt)

    # reversibility
    assert transport(Xt, B, A, metric=metric) == approx(X)

    # linearity
    Y = get_mats(n_matrices, n_channels, kindX)
    a, b = rndstate.uniform(0.01, 0.99, size=2)
    Yt = transport(Y, A, B, metric=metric)
    aXtpbYt = transport(a * X + b * Y, A, B, metric=metric)
    assert aXtpbYt == approx(a * Xt + b * Yt)

    # consistency wrt composition of geodesics
    xp = get_namespace(X)
    Xt_ABBC = transport(transport(X, A, B, metric), B, C, metric)
    alphas = np.linspace(0, 1, 5)
    G_AB = [geodesic(A, B, alpha, metric=metric) for alpha in alphas]
    G_BC = [geodesic(B, C, alpha, metric=metric) for alpha in alphas]
    G = xp.stack(G_AB + G_BC)
    Xt_AC = xp.asarray(X, copy=True)
    for i in range(len(G)-1):
        Xt_AC = transport(Xt_AC, G[i], G[i+1], metric)
    assert Xt_AC == approx(Xt_ABBC)

    # isometry, ie keep inner product and norm
    ia = innerproduct(X, Y, A, metric=metric)
    ib = innerproduct(Xt, Yt, B, metric=metric)
    assert ia == approx(ib)
    assert norm(X, A, metric=metric) == approx(norm(Xt, B, metric=metric))


@pytest.mark.numpy_only
def test_transport_riemann_vs_whitening(get_mats):
    """AIR PT from mean to identity is equivalent to a whitening"""
    n_matrices, n_channels = 15, 2
    X = get_mats(n_matrices, n_channels, "spd")

    Xw = Whitening(dim_red=None, metric="riemann").fit_transform(X)

    M = mean_riemann(X)
    T = log_map_riemann(X, M, C12=True)
    Tt = transport(T, M, np.eye(n_channels), metric="riemann")
    Xt = exp_map_riemann(Tt, np.eye(n_channels), Cm12=True)
    assert Xw == approx(Xt)


@pytest.mark.numpy_only
@pytest.mark.parametrize("kindX, kindQ", [("sym", "orth"), ("herm", "unit")])
def test_transport_wasserstein_commuting(kindX, kindQ, get_mats, rndstate):
    """BW transport has a closed form for commuting endpoints.

    Table 7 of [Thanwerdas2023], proved in its Appendix A.
    """
    n_channels = 4
    Q = get_mats(1, n_channels, kind=kindQ)[0]
    Qh = Q.conj().T
    d = rndstate.uniform(0.5, 3, n_channels)
    delta = rndstate.uniform(0.5, 3, n_channels)
    A = Q @ np.diag(d) @ Qh
    B = Q @ np.diag(delta) @ Qh
    X = get_mats(6, n_channels, kind=kindX)

    Xt = transport(X, A, B, metric="wasserstein")

    Xr = Qh @ X @ Q
    fac = np.sqrt(np.add.outer(delta, delta) / np.add.outer(d, d))
    expected = Q @ (fac * Xr) @ Qh
    assert Xt == approx(expected)


@pytest.mark.numpy_only
@pytest.mark.parametrize("n_steps", [0, -1, 1.5])
def test_transport_wasserstein_invalid_n_steps(n_steps, get_mats):
    """transport_wasserstein rejects non-positive-integer n_steps."""
    A, B = get_mats(2, 3, "spd")
    X = get_mats(1, 3, "sym")[0]
    with pytest.raises(ValueError):
        transport_wasserstein(X, A, B, n_steps=n_steps)


@pytest.mark.numpy_only
@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.uint8])
def test_transport_wasserstein_accepts_numpy_int(dtype, get_mats):
    """n_steps accepts NumPy integers of any width, not just np.int64."""
    A, B = get_mats(2, 3, "spd")
    X = get_mats(1, 3, "sym")[0]
    out_np = transport_wasserstein(X, A, B, n_steps=dtype(20))
    out_py = transport_wasserstein(X, A, B, n_steps=20)
    assert out_np == approx(out_py)


@pytest.mark.numpy_only
def test_transport_wasserstein_dispatch_forwards_n_steps(get_mats):
    """transport(metric='wasserstein') forwards n_steps."""
    A, B = get_mats(2, 3, "spd")
    X = get_mats(1, 3, "sym")[0]
    coarse = transport(X, A, B, metric="wasserstein", n_steps=1)
    assert coarse == approx(transport_wasserstein(X, A, B, n_steps=1))
    # n_steps is actually used: a single RK4 step differs from the default
    assert not np.allclose(
        to_numpy(coarse),
        to_numpy(transport(X, A, B, metric="wasserstein")),
        atol=1e-6,
    )
    # integer-like kwargs (e.g. NumPy int) survive the dispatcher too
    got = transport(X, A, B, metric="wasserstein", n_steps=np.int64(20))
    assert got == approx(transport_wasserstein(X, A, B, n_steps=20))
