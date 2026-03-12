import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

from pyriemann.utils._innerproduct import (
    innerproduct,
    innerproduct_euclid,
    innerproduct_logeuclid,
    innerproduct_riemann
)


metrics = ["euclid", "logeuclid", "riemann"]


@pytest.mark.parametrize("metric", metrics)
def test_innerproduct_build(metric, get_mats):
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    Y = get_mats(n_matrices, n_channels, "spd")
    Cref = get_mats(1, n_channels, "spd")[0]
    K = innerproduct(X, Y, Cref, metric=metric)
    assert_array_equal(K, globals()[f"innerproduct_{metric}"](X, Y, Cref))


def test_innerproduct_metric_string_error(get_mats):
    """Test generic innerproduct function error raise"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):
        innerproduct(X, X, X[0], metric="foo")


@pytest.mark.parametrize("metric", metrics)
def test_innerproduct_input_dimension_error(metric, get_mats):
    """Test errors for incorrect dimension"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    Y = get_mats(n_matrices, n_channels + 1, "spd")
    Cref = get_mats(1, n_channels + 2, "spd")[0]
    with pytest.raises(ValueError):
        innerproduct(X, Y, Cref, metric=metric)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", metrics)
def test_innerproduct_x_x(kind, metric, get_mats):
    """Test innerproduct for X = Y"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, kind)
    Cref = get_mats(1, n_channels, kind)[0]
    G = innerproduct(X, X, Cref, metric=metric)
    assert G.shape == (n_matrices,)
    G1 = innerproduct(X, None, Cref, metric=metric)
    assert_array_equal(G, G1)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", metrics)
def test_innerproduct_x_y(kind, metric, get_mats):
    """Test innerproduct for different X and Y"""
    n_matrices, n_channels = 4, 3
    X = get_mats(n_matrices, n_channels, kind)
    Y = get_mats(n_matrices, n_channels, kind)
    Cref = get_mats(1, n_channels, kind)[0]
    G = innerproduct(X, Y, Cref, metric=metric)
    assert G.shape == (n_matrices,)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", metrics)
def test_innerproduct_property_conjsymmetry(kind, metric, get_mats):
    n_matrices, n_channels = 5, 2
    X = get_mats(n_matrices, n_channels, kind)
    Y = get_mats(n_matrices, n_channels, kind)
    Cref = get_mats(1, n_channels, kind)[0]
    G1 = innerproduct(X, Y, Cref, metric=metric)
    G2 = innerproduct(Y, X, Cref, metric=metric)
    assert_array_almost_equal(G1.conj(), G2)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", metrics)
def test_innerproduct_property_linearity(kind, metric, get_mats, rndstate):
    n_matrices, n_channels = 4, 3
    X = get_mats(n_matrices, n_channels, kind)
    Y = get_mats(n_matrices, n_channels, kind)
    Z = get_mats(n_matrices, n_channels, kind)
    Cref = get_mats(1, n_channels, kind)[0]
    a, b = rndstate.uniform(0.01, 0.99, size=2)
    if kind == "hpd":
        a_, b_ = rndstate.uniform(0.01, 0.99, size=2)
        a += 1j * a_
        b += 1j * b_

    Gxy = innerproduct(X, Y, Cref, metric=metric)
    Gxz = innerproduct(X, Z, Cref, metric=metric)
    Gyz = innerproduct(Y, Z, Cref, metric=metric)

    Gaxpbz = innerproduct(a * X + b * Y, Z, Cref, metric=metric)
    assert_array_almost_equal(Gaxpbz, a.conj() * Gxz + b.conj() * Gyz)

    Gxaypbz = innerproduct(X, a * Y + b * Z, Cref, metric=metric)
    assert_array_almost_equal(Gxaypbz, a * Gxy + b * Gxz)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", metrics)
def test_innerproduct_property_positive_definite(kind, metric, get_mats):
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, kind)
    Cref = get_mats(1, n_channels, kind)[0]
    G = innerproduct(X, None, Cref, metric=metric)
    assert np.all(G > 0)


@pytest.mark.parametrize("kind", ["real", "comp"])
@pytest.mark.parametrize("n_dim1, n_dim2", [(4, 5), (5, 4)])
def test_innerproduct_euclid(kind, n_dim1, n_dim2, get_mats):
    """Euclidean inner-product for non-square matrices"""
    n_matrices = 3
    X = get_mats(n_matrices, [n_dim1, n_dim2], kind)
    Y = get_mats(n_matrices, [n_dim1, n_dim2], kind)
    G = innerproduct_euclid(X, Y)

    G1 = np.empty((n_matrices,), dtype=X.dtype)
    G2 = np.empty((n_matrices,), dtype=X.dtype)
    for i in range(n_matrices):
        G1[i] = np.trace(X[i].conj().T @ Y[i])
        G2[i] = np.dot(X[i].conj().flatten(), Y[i].flatten())
    assert_array_almost_equal(G, G1)
    assert_array_almost_equal(G, G2)


def test_innerproduct_logeuclid(get_mats):
    n_channels = 4
    X, Y = get_mats(2, n_channels, "spd")
    Cref = get_mats(1, n_channels, "spd")[0]
    G = innerproduct_logeuclid(X, Y, Cref)
    assert isinstance(G, float)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_innerproduct_riemann(kind, get_mats):
    n_channels = 5
    X, Y = get_mats(2, n_channels, kind)
    Cref = get_mats(1, n_channels, kind)[0]
    G = innerproduct_riemann(X, Y, Cref)

    if kind == "hpd":
        return

    # Eq(2.6) in [Moakher2005]
    Cinv = np.linalg.inv(Cref)
    G1 = np.trace(Cinv @ X @ Cinv @ Y)
    assert_array_almost_equal(G, G1)
    G2 = np.trace(X @ Cinv @ Y @ Cinv)
    assert_array_almost_equal(G, G2)
