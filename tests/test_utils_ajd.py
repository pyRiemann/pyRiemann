from numpy.testing import assert_array_equal
import numpy as np
import pytest
from pytest import approx

from pyriemann.utils.ajd import rjd, ajd_pham, uwedge


@pytest.mark.parametrize("ajd", [rjd, ajd_pham, uwedge])
@pytest.mark.parametrize("init", [True, False])
def test_ajd(ajd, init, get_covmats_params):
    """Test ajd algos"""
    n_matrices, n_channels = 5, 3
    covmats, _, evecs = get_covmats_params(n_matrices, n_channels)
    if init:
        V, D = ajd(covmats)
    else:
        V, D = ajd(covmats, init=evecs)
    assert V.shape == (n_channels, n_channels)
    assert D.shape == (n_matrices, n_channels, n_channels)

    if ajd is rjd:
        assert V.T @ V == approx(np.eye(n_channels))  # check orthogonality


@pytest.mark.parametrize("ajd", [rjd, ajd_pham, uwedge])
def test_ajd_init_error(ajd, get_covmats):
    """Test init for ajd algos"""
    n_matrices, n_channels = 5, 3
    covmats = get_covmats(n_matrices, n_channels)
    with pytest.raises(ValueError):  # not 2D array
        ajd(covmats, init=np.ones((3, 2, 2)))
    with pytest.raises(ValueError):  # not square array
        ajd(covmats, init=np.ones((3, 2)))
    with pytest.raises(ValueError):  # shape not equal to n_channels
        ajd(covmats, init=np.ones((2, 2)))


def test_pham_weight_none_equivalent_uniform(get_covmats):
    """Test pham's ajd weights: none is equivalent to uniform values"""
    n_matrices, n_channels, w_val = 5, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    V, D = ajd_pham(covmats)
    assert V.shape == (n_channels, n_channels)
    assert D.shape == (n_matrices, n_channels, n_channels)

    Vw, Dw = ajd_pham(covmats, sample_weight=w_val * np.ones(n_matrices))
    assert_array_equal(V, Vw)  # same result as ajd_pham without weight
    assert_array_equal(D, Dw)


def test_pham_weight_positive(get_covmats):
    """Test pham's ajd weights: must be strictly positive"""
    n_matrices, n_channels, w_val = 5, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    w = w_val * np.ones(n_matrices)
    with pytest.raises(ValueError):  # not strictly positive weight
        w[0] = 0
        ajd_pham(covmats, sample_weight=w)


def test_pham_weight_zero(get_covmats):
    """Setting one weight to almost 0 it's almost like not passing the mat"""
    n_matrices, n_channels, w_val = 5, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    w = w_val * np.ones(n_matrices)
    V, D = ajd_pham(covmats[1:], sample_weight=w[1:])
    w[0] = 1e-12
    Vw, Dw = ajd_pham(covmats, sample_weight=w)
    assert V == approx(Vw, rel=1e-4, abs=1e-8)
    assert D == approx(Dw[1:], rel=1e-4, abs=1e-8)
