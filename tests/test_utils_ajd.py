from numpy.testing import assert_array_equal
import numpy as np
import pytest
from pytest import approx

from pyriemann.utils.ajd import ajd, rjd, ajd_pham, uwedge


@pytest.mark.parametrize(
    "method, algo",
    [
        ("rjd", rjd),
        ("ajd_pham", ajd_pham),
        ("uwedge", uwedge),
        (uwedge, uwedge),
    ]
)
def test_ajd(method, algo, get_covmats):
    """Test ajd algos"""
    eps, n_iter_max = 1e-6, 100
    n_matrices, n_channels = 10, 3
    mats = get_covmats(n_matrices, n_channels)

    V, D = ajd(mats, method=method, eps=eps, n_iter_max=n_iter_max)
    assert V.shape == (n_channels, n_channels)
    assert D.shape == (n_matrices, n_channels, n_channels)

    if method == "rjd":
        assert D == approx(V.T @ mats @ V)
        assert V.T @ V == approx(np.eye(n_channels))  # check orthogonality
    else:
        assert D == approx(V @ mats @ V.T)

    V_, D_ = algo(mats, eps=eps, n_iter_max=n_iter_max)
    assert np.all(V == V_)
    assert np.all(D == D_)


@pytest.mark.parametrize("method", ["rjd", "ajd_pham", "uwedge"])
@pytest.mark.parametrize("init", [True, False])
def test_ajd_init(method, init, get_covmats_params):
    """Test init for ajd algos"""
    n_matrices, n_channels = 9, 4
    mats, _, evecs = get_covmats_params(n_matrices, n_channels)
    if init:
        ajd(mats, method=method, init=evecs)
    else:
        ajd(mats, method=method)


@pytest.mark.parametrize("method", ["rjd", "ajd_pham", "uwedge"])
def test_ajd_init_error(method, get_covmats):
    """Test init errors for ajd algos"""
    n_matrices, n_channels = 4, 3
    mats = get_covmats(n_matrices, n_channels)
    with pytest.raises(ValueError):  # not 2D array
        ajd(mats, method=method, init=np.ones((3, 2, 2)))
    with pytest.raises(ValueError):  # not square array
        ajd(mats, method=method, init=np.ones((3, 2)))
    with pytest.raises(ValueError):  # shape not equal to n_channels
        ajd(mats, method=method, init=np.ones((2, 2)))


@pytest.mark.parametrize("method", ["abc", 42])
def test_ajd_method_error(method):
    with pytest.raises(ValueError):
        ajd(np.ones((3, 2, 2)), method=method)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_ajd_pham(kind, get_mats):
    """Test pham's ajd"""
    n_matrices, n_channels = 7, 4
    mats = get_mats(n_matrices, n_channels, kind)
    V, D = ajd_pham(mats)
    assert D == approx(V @ mats @ V.conj().T)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_ajd_pham_weight_none_equivalent_uniform(kind, get_mats):
    """Test pham's ajd weights: none is equivalent to uniform values"""
    n_matrices, n_channels = 5, 3
    mats = get_mats(n_matrices, n_channels, kind)
    V, D = ajd_pham(mats)
    Vw, Dw = ajd_pham(mats, sample_weight=np.ones(n_matrices))
    assert_array_equal(V, Vw)  # same result as ajd_pham without weight
    assert_array_equal(D, Dw)

    ajd(mats, method="ajd_pham", sample_weight=np.ones(n_matrices))


def test_ajd_pham_weight_positive(get_covmats):
    """Test pham's ajd weights: must be strictly positive"""
    n_matrices, n_channels = 4, 2
    mats = get_covmats(n_matrices, n_channels)
    w = 1.23 * np.ones(n_matrices)
    with pytest.raises(ValueError):  # not strictly positive weight
        w[0] = 0
        ajd_pham(mats, sample_weight=w)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_ajd_pham_weight_zero(kind, get_mats):
    """Setting one weight to almost 0 it's almost like not passing the mat"""
    n_matrices, n_channels = 5, 4
    mats = get_mats(n_matrices, n_channels, kind)
    w = 4.32 * np.ones(n_matrices)
    V, D = ajd_pham(mats[1:], sample_weight=w[1:])
    w[0] = 1e-12
    Vw, Dw = ajd_pham(mats, sample_weight=w)
    assert V == approx(Vw, rel=1e-4, abs=1e-8)
    assert D == approx(Dw[1:], rel=1e-4, abs=1e-8)
