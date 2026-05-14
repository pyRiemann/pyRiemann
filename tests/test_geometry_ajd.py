from array_api_compat import array_namespace as get_namespace, device as xpd
import numpy as np
import pytest

from conftest import approx, assert_array_equal
from pyriemann.geometry.ajd import ajd, rjd, ajd_pham, uwedge


@pytest.mark.parametrize("kind", ["sym", "spd"])
@pytest.mark.parametrize(
    "method, algo",
    [
        ("rjd", rjd),
        ("ajd_pham", ajd_pham),
        ("uwedge", uwedge),
        (uwedge, uwedge),
    ]
)
def test_ajd(kind, method, algo, get_mats):
    """Test ajd algos"""
    if kind == "sym" and method == "ajd_pham":
        return
    eps, n_iter_max = 1e-6, 100
    n_matrices, n_channels = 10, 3
    X = get_mats(n_matrices, n_channels, kind)

    V, D = ajd(X, method=method, eps=eps, n_iter_max=n_iter_max)
    assert V.shape == (n_channels, n_channels)
    assert D.shape == (n_matrices, n_channels, n_channels)

    xp = get_namespace(X)
    if method == "rjd":
        assert D == approx(V.mT @ X @ V)
        eye = xp.eye(n_channels, dtype=X.dtype, device=xpd(X))
        assert V.mT @ V == approx(eye)  # check orthogonality
    else:
        assert D == approx(V @ X @ V.mT)

    V_, D_ = algo(X, eps=eps, n_iter_max=n_iter_max)
    assert_array_equal(V, V_)
    assert_array_equal(D, D_)


@pytest.mark.parametrize("method", ["rjd", "ajd_pham", "uwedge"])
@pytest.mark.parametrize("use_init", [True, False])
def test_ajd_init(method, use_init, get_mats_params):
    """Test init for ajd algos"""
    n_matrices, n_channels = 6, 4
    X, _, evecs = get_mats_params(n_matrices, n_channels, "spd")
    if use_init:
        ajd(X, method=method, init=evecs)
    else:
        ajd(X, method=method)


@pytest.mark.parametrize("method", ["abc", 42])
def test_ajd_method_error(method):
    with pytest.raises(ValueError):
        ajd(np.ones((3, 2, 2)), method=method)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_ajdpham(kind, get_mats):
    """Test pham's ajd"""
    n_matrices, n_channels = 7, 4
    X = get_mats(n_matrices, n_channels, kind)
    V, D = ajd_pham(X)
    assert D == approx(V @ X @ V.conj().mT)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_ajdpham_weight_none_equivalent_uniform(kind, get_mats):
    """Test pham's ajd weights: none is equivalent to uniform values"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, kind)
    xp = get_namespace(X)
    ones = xp.ones(n_matrices, dtype=X.real.dtype, device=xpd(X))
    V, D = ajd_pham(X)
    Vw, Dw = ajd_pham(X, sample_weight=ones)
    assert_array_equal(V, Vw)
    assert_array_equal(D, Dw)

    ajd(X, method="ajd_pham", sample_weight=ones)


def test_ajdpham_weight_positive(get_mats, get_weights):
    """Test pham's ajd weights: must be strictly positive"""
    n_matrices, n_channels = 4, 2
    X = get_mats(n_matrices, n_channels, "spd")
    weights = get_weights(n_matrices)
    with pytest.raises(ValueError):  # not strictly positive weight
        weights[0] = 0
        ajd_pham(X, sample_weight=weights)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_ajdpham_weight_zero(kind, get_mats, get_weights):
    """Setting one weight to almost 0 it's almost like not passing the mat"""
    n_matrices, n_channels = 5, 4
    X = get_mats(n_matrices, n_channels, kind)
    weights = get_weights(n_matrices)
    V, D = ajd_pham(X[1:], sample_weight=weights[1:])
    weights[0] = 1e-12
    Vw, Dw = ajd_pham(X, sample_weight=weights)
    assert V == approx(Vw, rel=1e-4, abs=1e-8)
    assert D == approx(Dw[1:], rel=1e-4, abs=1e-8)
