import numpy as np
from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal,
    assert_raises,
)
import pytest

from pyriemann.utils.base import logm
from pyriemann.utils.kernel import (
    kernel,
    kernel_euclid,
    kernel_logeuclid,
    kernel_riemann
)
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.test import is_sym_pos_semi_def as is_spsd

metrics = ["euclid", "logeuclid", "riemann"]


@pytest.mark.parametrize("metric", metrics)
def test_kernel_build(metric, get_mats):
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = kernel(X, X, metric=metric)
    assert_array_almost_equal(K, globals()[f"kernel_{metric}"](X))


@pytest.mark.parametrize("metric", metrics)
def test_kernel_metric_string(metric, get_mats):
    """Test generic kernel function"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = globals()[f"kernel_{metric}"](X)
    K1 = kernel(X, metric=metric)
    assert_array_equal(K, K1)


def test_kernel_metric_string_error(get_mats):
    """Test generic kernel function error raise"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):
        kernel(X, metric="foo")


@pytest.mark.parametrize("metric", metrics)
def test_kernel_input_dimension_error(metric, get_mats):
    """Test errors for incorrect dimension"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    Y = get_mats(n_matrices, n_channels + 1, "spd")
    Cref = get_mats(1, n_channels + 2, "spd")[0]
    with pytest.raises(ValueError):
        kernel(X, Y, metric=metric)
    with pytest.raises(ValueError):
        kernel(X, Cref=Cref, metric=metric)


@pytest.mark.parametrize("metric", metrics)
def test_kernel_x_x(metric, get_mats):
    """Test kernel for X = Y"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = kernel(X, X, metric=metric)
    assert K.shape == (n_matrices, n_matrices)


@pytest.mark.parametrize("metric", metrics)
def test_kernel_x_y(metric, get_mats):
    """Test kernel for different X and Y"""
    n_matrices_X, n_matrices_Y, n_channels = 4, 5, 3
    X = get_mats(n_matrices_X, n_channels, "spd")
    Y = get_mats(n_matrices_Y, n_channels, "spd")
    K = kernel(X, Y, metric=metric)
    assert K.shape == (n_matrices_X, n_matrices_Y)


@pytest.mark.parametrize("n_channels", [4, 5, 6])
@pytest.mark.parametrize("metric", metrics)
def test_kernel_cref(n_channels, metric, get_mats):
    n_matrices_X, n_matrices_Y = 7, 8
    X = get_mats(n_matrices_X, n_channels, "spd")
    Y = get_mats(n_matrices_Y, n_channels, "spd")
    K = kernel(X, Y, metric=metric)

    if metric == "euclid":
        Cref = np.zeros((n_channels, n_channels))
    elif metric == "logeuclid":
        Cref = np.eye(n_channels)
    elif metric == "riemann":
        Cref = mean_covariance(X)
    K1 = kernel(X, Y, Cref=Cref, metric=metric)
    assert_array_equal(K, K1)

    K2 = kernel(X, Y, Cref=X[0], metric=metric)
    assert_raises(AssertionError, assert_array_equal, K, K2)


@pytest.mark.parametrize("metric", metrics)
def test_kernel_property_symmetry(metric, get_mats):
    n_matrices_X, n_matrices_Y, n_channels = 5, 4, 2
    X = get_mats(n_matrices_X, n_channels, "spd")
    Y = get_mats(n_matrices_Y, n_channels, "spd")
    Cref = get_mats(1, n_channels, "spd")[0]
    K1 = kernel(X, Y, Cref=Cref, metric=metric)
    K2 = kernel(Y, X, Cref=Cref, metric=metric)
    assert_array_almost_equal(K1, K2.T)


@pytest.mark.parametrize("metric", metrics)
def test_kernel_property_positive_semi_definite(metric, get_mats):
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = kernel(X, X, metric=metric, reg=1e-6)
    assert is_spsd(K)


@pytest.mark.parametrize("n_dim1, n_dim2", [(4, 5), (5, 4)])
def test_kernel_euclid(n_dim1, n_dim2, get_mats):
    """Euclidean kernel for non-square matrices"""
    n_matrices_X, n_matrices_Y = 2, 3
    X = get_mats(n_matrices_X, [n_dim1, n_dim2], "real")
    Y = get_mats(n_matrices_Y, [n_dim1, n_dim2], "real")
    K = kernel_euclid(X, Y)

    K1 = np.empty((n_matrices_X, n_matrices_Y))
    K2 = np.empty((n_matrices_X, n_matrices_Y))
    for i in range(n_matrices_X):
        for j in range(n_matrices_Y):
            K1[i, j] = np.trace(X[i].T @ Y[j])
            K2[i, j] = np.dot(X[i].flatten(), Y[j].flatten())
    assert_array_almost_equal(K, K1)
    assert_array_almost_equal(K, K2)

    kernel_euclid(X, Y, Cref=X[0])


def test_kernel_logeuclid(get_mats):
    n_matrices_X, n_matrices_Y, n_channels = 5, 4, 3
    X = get_mats(n_matrices_X, n_channels, "spd")
    Y = get_mats(n_matrices_Y, n_channels, "spd")
    Cref = np.eye(n_channels)

    # equivalence with Riemannian kernel when Cref is identity
    # Eq(10) of [Barachant2013]
    Kle = kernel_logeuclid(X, Y, Cref=Cref)
    Kr = kernel_riemann(X, Y, Cref=Cref)
    assert_array_almost_equal(Kle, Kr)


def test_kernel_riemann(get_mats):
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = kernel_riemann(X, Cref=np.eye(n_channels), reg=0)

    # test correctness
    log_X = logm(X)
    tensor = np.tensordot(log_X, log_X.T, axes=1)
    K1 = np.trace(tensor, axis1=1, axis2=2)
    assert_array_almost_equal(K, K1)
