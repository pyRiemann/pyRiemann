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
    X = get_mats(n_matrices, n_channels, kind)
    M = median(X)
    assert M.shape == (n_channels, n_channels)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("median", [median_euclid, median_riemann])
def test_median_shape_with_init(kind, median, get_mats):
    """Test the shape of median with init"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, kind)
    M = median(X, init=X[0])
    assert M.shape == (n_channels, n_channels)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("median", [median_euclid, median_riemann])
def test_median_weight_zero(kind, median, get_mats, get_weights):
    """Setting one weight to almost 0 it's almost like not passing the mat"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, kind)
    weights = get_weights(n_matrices)

    M = median(X[1:], weights=weights[1:])
    weights[0] = 1e-12
    Mw = median(X, weights=weights)
    assert M == approx(Mw, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("median", [median_euclid, median_riemann])
def test_median_warning_convergence(median, get_mats):
    """Test warning for convergence not reached"""
    n_matrices, n_channels = 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    with pytest.warns(UserWarning):
        median(X, maxiter=0)


@pytest.mark.parametrize("n_values", [3, 5, 7])
def test_median_euclid_scalars(n_values, rndstate):
    """Compare geometric Euclidean median to marginal median for scalars"""
    values = 100 * rndstate.randn(n_values)
    np_med = np.median(values)
    py_med = median_euclid(values[..., np.newaxis, np.newaxis])[0, 0]
    assert np_med == approx(py_med)


@pytest.mark.parametrize("n_dim1, n_dim2", [(4, 5), (5, 4)])
@pytest.mark.parametrize("kind", ["real", "comp"])
def test_median_euclid(n_dim1, n_dim2, kind, get_mats):
    """Euclidean median for non-square matrices"""
    n_matrices = 10
    X = get_mats(n_matrices, [n_dim1, n_dim2], kind)
    assert median_euclid(X).shape == (n_dim1, n_dim2)


@pytest.mark.parametrize("step_size", [0, 2.5])
def test_median_riemann_stepsize_error(step_size, get_mats):
    n_matrices, n_channels = 1, 2
    X = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):
        median_riemann(X, step_size=step_size)
