import numpy as np
from numpy.core import tensordot, trace
from numpy.testing import assert_array_equal, assert_array_almost_equal
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

rker_str = ['euclid', 'logeuclid', 'riemann']
rker_fct = [kernel_euclid, kernel_logeuclid, kernel_riemann]


@pytest.mark.parametrize("ker", rker_fct)
def test_kernel_x_x(ker, rndstate, get_covmats):
    """Test Kernel build"""
    n_matrices, n_channels = 12, 3
    X = get_covmats(n_matrices, n_channels)
    K = ker(X, X)
    assert is_spsd(K)
    assert K.shape == (n_matrices, n_matrices)


@pytest.mark.parametrize("ker", rker_str)
def test_kernel_cref(ker, rndstate, get_covmats):
    """Test Kernel reference"""
    n_matrices, n_channels = 5, 3
    X = get_covmats(n_matrices, n_channels)
    cref = mean_covariance(X, metric=ker)
    K = kernel(X, X, metric=ker)
    K1 = kernel(X, X, Cref=cref, metric=ker)
    assert_array_equal(K, K1)


@pytest.mark.parametrize("ker", rker_str)
def test_kernel_x_y(ker, rndstate, get_covmats):
    """Test Kernel for different X and Y."""
    n_matrices, n_channels = 5, 3
    X = get_covmats(n_matrices, n_channels)
    Y = get_covmats(n_matrices + 1, n_channels)
    K = kernel(X, Y, metric=ker)
    assert K.shape == (n_matrices, n_matrices + 1)


@pytest.mark.parametrize("ker", rker_str)
def test_metric_string(ker, rndstate, get_covmats):
    """Test generic Kernel function."""
    n_matrices, n_channels = 5, 3
    X = get_covmats(n_matrices, n_channels)
    K = globals()[f'kernel_{ker}'](X)
    K1 = kernel(X, metric=ker)
    assert_array_equal(K, K1)


def test_metric_string_error(rndstate, get_covmats):
    """Test generic Kernel function error raise."""
    n_matrices, n_channels = 5, 3
    X = get_covmats(n_matrices, n_channels)
    with pytest.raises(ValueError):
        kernel(X, metric='foo')


@pytest.mark.parametrize("ker", rker_str)
def test_input_dimension_error(ker, rndstate, get_covmats):
    """Test errors for incorrect dimension."""
    n_matrices, n_channels = 5, 3
    X = get_covmats(n_matrices, n_channels)
    Y = get_covmats(n_matrices, n_channels + 1)
    cref = get_covmats(1, n_channels + 1)[0]
    if ker == 'riemann':
        with pytest.raises(AssertionError):
            kernel(X, Cref=cref, metric=ker)
    with pytest.raises(AssertionError):
        kernel(X, Y, metric=ker)


@pytest.mark.parametrize("n_dim0, n_dim1", [(4, 4), (4, 5), (5, 4)])
def test_euclid(n_dim0, n_dim1, rndstate):
    """Test the Euclidean kernel for generic matrices"""
    n_matrices_X, n_matrices_Y = 2, 3
    X = rndstate.randn(n_matrices_X, n_dim0, n_dim1)
    Y = rndstate.randn(n_matrices_Y, n_dim0, n_dim1)
    K = kernel_euclid(X, Y)
    assert K.shape == (n_matrices_X, n_matrices_Y)
    assert_array_almost_equal(K[0, 0], np.trace(X[0].T @ Y[0]))


def test_riemann_correctness(rndstate, get_covmats):
    """Test example correctness of Riemann kernel."""
    n_matrices, n_channels = 5, 3
    X = get_covmats(n_matrices, n_channels)
    K = kernel_riemann(X, Cref=np.eye(n_channels), reg=0)

    log_X = logm(X)
    tensor = tensordot(log_X, log_X.T, axes=1)
    K1 = trace(tensor, axis1=1, axis2=2)
    assert_array_almost_equal(K, K1)
