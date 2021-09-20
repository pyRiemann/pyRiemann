from numpy.testing import assert_array_equal
import numpy as np
import pytest
from pytest import approx

from pyriemann.utils.ajd import rjd, ajd_pham, uwedge, _get_normalized_weight


def test_get_normalized_weight(get_covmats):
    """Test get_normalized_weight"""
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)
    w = _get_normalized_weight(None, covmats)
    assert np.sum(w) == approx(1.0, abs=1e-10)


def test_get_normalized_weight_length(get_covmats):
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)
    w = _get_normalized_weight(None, covmats)
    with pytest.raises(ValueError):  # not same length
        _get_normalized_weight(w[: n_trials // 2], covmats)


def test_get_normalized_weight_pos(get_covmats):
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)
    w = _get_normalized_weight(None, covmats)
    with pytest.raises(ValueError):  # not strictly positive weight
        w[0] = 0
        _get_normalized_weight(w, covmats)


@pytest.mark.parametrize("ajd", [rjd, ajd_pham])
def test_ajd_shape(ajd, get_covmats):
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)
    V, D = rjd(covmats)
    assert V.shape == (n_channels, n_channels)
    assert D.shape == (n_trials, n_channels, n_channels)


def test_pham(get_covmats):
    """Test pham's ajd"""
    n_trials, n_channels, w_val = 5, 3, 2
    covmats = get_covmats(n_trials, n_channels)
    V, D = ajd_pham(covmats)
    assert V.shape == (n_channels, n_channels)
    assert D.shape == (n_trials, n_channels, n_channels)

    Vw, Dw = ajd_pham(covmats, sample_weight=w_val * np.ones(n_trials))
    assert_array_equal(V, Vw)  # same result as ajd_pham without weight
    assert_array_equal(D, Dw)


def test_pham_pos_weight(get_covmats):
    # Test that weight must be strictly positive
    n_trials, n_channels, w_val = 5, 3, 2
    covmats = get_covmats(n_trials, n_channels)
    w = w_val * np.ones(n_trials)
    with pytest.raises(ValueError):  # not strictly positive weight
        w[0] = 0
        ajd_pham(covmats, sample_weight=w)


def test_pham_zero_weight(get_covmats):
    # now test that setting one weight to almost zero it's almost
    # like not passing the matrix
    n_trials, n_channels, w_val = 5, 3, 2
    covmats = get_covmats(n_trials, n_channels)
    w = w_val * np.ones(n_trials)
    V, D = ajd_pham(covmats[1:], sample_weight=w[1:])
    w[0] = 1e-12

    Vw, Dw = ajd_pham(covmats, sample_weight=w)
    assert V == approx(Vw, rel=1e-4, abs=1e-8)
    assert D == approx(Dw[1:], rel=1e-4, abs=1e-8)


@pytest.mark.parametrize("init", [True, False])
def test_uwedge(init, get_covmats_params):
    """Test uwedge."""
    n_trials, n_channels = 5, 3
    covmats, _, A = get_covmats_params(n_trials, n_channels)
    if init:
        V, D = uwedge(covmats)
    else:
        V, D = uwedge(covmats, init=A)
    assert V.shape == (n_channels, n_channels)
    assert D.shape == (n_trials, n_channels, n_channels)
