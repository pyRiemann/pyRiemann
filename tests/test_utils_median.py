import numpy as np
import pytest
from pytest import approx

from pyriemann.utils.median import (
    median_euclid,
    median_riemann,
)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("median", [median_euclid, median_riemann])
def test_median_shape(kind, median, get_mats):
    """Test the shape of median"""
    n_matrices, n_channels = 3, 4
    mats = get_mats(n_matrices, n_channels, kind)
    C = median(mats)
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("median", [median_euclid, median_riemann])
def test_median_shape_with_init(kind, median, get_mats):
    """Test the shape of median with init"""
    n_matrices, n_channels = 5, 3
    mats = get_mats(n_matrices, n_channels, kind)
    C = median(mats, init=mats[0])
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("median", [median_euclid, median_riemann])
def test_median_weight_zero(kind, median, get_mats):
    """Setting one weight to almost 0 it's almost like not passing the mat"""
    n_matrices, n_channels, w_val = 5, 3, 2
    mats = get_mats(n_matrices, n_channels, kind)
    w = w_val * np.ones(n_matrices)
    C = median(mats[1:], weights=w[1:])
    w[0] = 1e-12
    Cw = median(mats, weights=w)
    assert C == approx(Cw, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("median", [median_euclid, median_riemann])
def test_median_warning_convergence(median, get_mats):
    """Test warning for convergence not reached"""
    n_matrices, n_channels = 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    with pytest.warns(UserWarning):
        median(mats, maxiter=0)


@pytest.mark.parametrize("n_values", [3, 5, 7])
def test_median_euclid_1d(n_values, rndstate):
    """Compare geometric Euclidean median to marginal median in 1D"""
    values = 100 * rndstate.randn(n_values)
    np_med = np.median(values)
    py_med = median_euclid(values[..., np.newaxis, np.newaxis])[0, 0]
    assert np_med == approx(py_med)


@pytest.mark.parametrize("complex_valued", [True, False])
def test_median_euclid(rndstate, complex_valued):
    """Test the Euclidean median for generic matrices"""
    n_matrices, n_dim0, n_dim1 = 10, 3, 4
    mats = rndstate.randn(n_matrices, n_dim0, n_dim1)
    if complex_valued:
        mats = mats + 1j * rndstate.randn(n_matrices, n_dim0, n_dim1)
    assert median_euclid(mats).shape == (n_dim0, n_dim1)


@pytest.mark.parametrize("step_size", [0, 2.5])
def test_median_riemann_stepsize_error(step_size, get_mats):
    n_matrices, n_channels = 1, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):
        median_riemann(mats, step_size=step_size)
