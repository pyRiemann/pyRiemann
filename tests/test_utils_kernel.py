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

rker_str = ["euclid", "logeuclid", "riemann"]


@pytest.mark.parametrize("ker", rker_str)
def test_kernel_x_x(ker, get_mats):
    """Test kernel build"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = kernel(X, X, metric=ker)
    assert K.shape == (n_matrices, n_matrices)
    assert is_spsd(K)
    assert_array_almost_equal(K, globals()[f"kernel_{ker}"](X))


@pytest.mark.parametrize("ker", rker_str)
def test_kernel_x_y(ker, get_mats):
    """Test kernel for different X and Y"""
    n_matrices_X, n_matrices_Y, n_channels = 4, 5, 3
    X = get_mats(n_matrices_X, n_channels, "spd")
    Y = get_mats(n_matrices_Y, n_channels, "spd")
    K = kernel(X, Y, metric=ker)
    assert K.shape == (n_matrices_X, n_matrices_Y)


@pytest.mark.parametrize("ker", rker_str)
def test_metric_string(ker, get_mats):
    """Test generic kernel function"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = globals()[f"kernel_{ker}"](X)
    K1 = kernel(X, metric=ker)
    assert_array_equal(K, K1)


def test_metric_string_error(get_mats):
    """Test generic kernel function error raise"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):
        kernel(X, metric="foo")


@pytest.mark.parametrize("ker", rker_str)
def test_input_dimension_error(ker, get_mats):
    """Test errors for incorrect dimension"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    Y = get_mats(n_matrices, n_channels + 1, "spd")
    cref = get_mats(1, n_channels + 1, "spd")[0]
    if ker in ["logeuclid", "riemann"]:
        with pytest.raises(AssertionError):
            kernel(X, Cref=cref, metric=ker)
    with pytest.raises(AssertionError):
        kernel(X, Y, metric=ker)


@pytest.mark.parametrize("n_channels", [4, 5, 6])
@pytest.mark.parametrize("ker", rker_str)
def test_cref(n_channels, ker, get_mats):
    n_matrices_X, n_matrices_Y = 7, 4
    X = get_mats(n_matrices_X, n_channels, "spd")
    Y = get_mats(n_matrices_Y, n_channels, "spd")
    K = kernel(X, Y, metric=ker)

    if ker == "euclid":
        Cref = np.zeros((n_channels, n_channels))
    elif ker == "logeuclid":
        Cref = np.eye(n_channels)
    elif ker == "riemann":
        Cref = mean_covariance(X)

    K1 = kernel(X, Y, Cref=Cref, metric=ker)
    assert_array_equal(K, K1)

    K2 = kernel(X, Y, Cref=X[0], metric=ker)
    assert_raises(AssertionError, assert_array_equal, K, K2)


@pytest.mark.parametrize("n_dim0, n_dim1", [(4, 4), (4, 5), (5, 4)])
def test_euclid(n_dim0, n_dim1, rndstate):
    """Test Euclidean kernel for generic matrices"""
    n_matrices_X, n_matrices_Y = 2, 3
    X = rndstate.randn(n_matrices_X, n_dim0, n_dim1)
    Y = rndstate.randn(n_matrices_Y, n_dim0, n_dim1)
    K = kernel_euclid(X, Y)

    K1 = np.empty((n_matrices_X, n_matrices_Y))
    K2 = np.empty((n_matrices_X, n_matrices_Y))
    for i in range(n_matrices_X):
        for j in range(n_matrices_Y):
            K1[i, j] = np.trace(X[i].T @ Y[j])
            K2[i, j] = np.dot(X[i].flatten(), Y[j].flatten())
    assert_array_almost_equal(K, K1)
    assert_array_almost_equal(K, K2)

    kernel_euclid(X, Y, Cref=rndstate.randn(n_dim0, n_dim1))


def test_logeuclid(get_mats):
    n_matrices, n_channels = 5, 4
    X = get_mats(n_matrices, n_channels, "spd")
    Y = get_mats(n_matrices, n_channels, "spd")
    Cref = np.eye(n_channels)

    # test equivalence with Riemannian kernel when Cref is identity
    Kle = kernel_logeuclid(X, Y, Cref=Cref)
    Kr = kernel_riemann(X, Y, Cref=Cref)
    assert_array_almost_equal(Kle, Kr)


def test_riemann(get_mats):
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = kernel_riemann(X, Cref=np.eye(n_channels), reg=0)

    # test correctness
    log_X = logm(X)
    tensor = np.tensordot(log_X, log_X.T, axes=1)
    K1 = np.trace(tensor, axis1=1, axis2=2)
    assert_array_almost_equal(K, K1)
