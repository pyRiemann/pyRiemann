import numpy as np
from numpy.testing import assert_array_equal
import pytest
from pytest import approx

from pyriemann.utils.ajd import ajd, rjd, ajd_pham, uwedge


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
    mats = get_mats(n_matrices, n_channels, kind)

    V, D = ajd(mats, method=method, eps=eps, n_iter_max=n_iter_max)
    assert V.shape == (n_channels, n_channels)
    assert D.shape == (n_matrices, n_channels, n_channels)

    if method == "rjd":
        assert D == approx(V.T @ mats @ V)
        assert V.T @ V == approx(np.eye(n_channels))  # check orthogonality
    else:
        assert D == approx(V @ mats @ V.T)

    V_, D_ = algo(mats, eps=eps, n_iter_max=n_iter_max)
    assert_array_equal(V, V_)
    assert_array_equal(D, D_)


@pytest.mark.parametrize("method", ["rjd", "ajd_pham", "uwedge"])
@pytest.mark.parametrize("use_init", [True, False])
def test_ajd_init(method, use_init, get_mats_params):
    """Test init for ajd algos"""
    n_matrices, n_channels = 6, 4
    mats, _, evecs = get_mats_params(n_matrices, n_channels, "spd")
    if use_init:
        ajd(mats, method=method, init=evecs)
    else:
        ajd(mats, method=method)


@pytest.mark.parametrize("method", ["abc", 42])
def test_ajd_method_error(method):
    with pytest.raises(ValueError):
        ajd(np.ones((3, 2, 2)), method=method)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_ajdpham(kind, get_mats):
    """Test pham's ajd"""
    n_matrices, n_channels = 7, 4
    mats = get_mats(n_matrices, n_channels, kind)
    V, D = ajd_pham(mats)
    assert D == approx(V @ mats @ V.conj().T)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_ajdpham_weight_none_equivalent_uniform(kind, get_mats):
    """Test pham's ajd weights: none is equivalent to uniform values"""
    n_matrices, n_channels = 5, 3
    mats = get_mats(n_matrices, n_channels, kind)
    V, D = ajd_pham(mats)
    Vw, Dw = ajd_pham(mats, sample_weight=np.ones(n_matrices))
    assert_array_equal(V, Vw)
    assert_array_equal(D, Dw)

    ajd(mats, method="ajd_pham", sample_weight=np.ones(n_matrices))


def test_ajdpham_weight_positive(get_mats, get_weights):
    """Test pham's ajd weights: must be strictly positive"""
    n_matrices, n_channels = 4, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    weights = get_weights(n_matrices)
    with pytest.raises(ValueError):  # not strictly positive weight
        weights[0] = 0
        ajd_pham(mats, sample_weight=weights)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_ajdpham_weight_zero(kind, get_mats, get_weights):
    """Setting one weight to almost 0 it's almost like not passing the mat"""
    n_matrices, n_channels = 5, 4
    mats = get_mats(n_matrices, n_channels, kind)
    weights = get_weights(n_matrices)
    V, D = ajd_pham(mats[1:], sample_weight=weights[1:])
    weights[0] = 1e-12
    Vw, Dw = ajd_pham(mats, sample_weight=weights)
    assert V == approx(Vw, rel=1e-4, abs=1e-8)
    assert D == approx(Dw[1:], rel=1e-4, abs=1e-8)
