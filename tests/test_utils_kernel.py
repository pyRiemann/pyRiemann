from pyriemann.utils.kernel import kernel_riemann, kernel
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import logm

from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np

from pyriemann.utils.test import is_pos_semi_def as is_spsd


def test_riemann_kernel(rndstate, get_covmats):
    """Test Riemannian Kernel build"""
    n_matrices, n_channels = 5, 3
    cov = get_covmats(n_matrices, n_channels)
    K = kernel_riemann(cov, cov, np.eye(n_channels))

    log_cov = np.array([logm(c) for c in cov])
    tensor = np.tensordot(log_cov, log_cov.T, axes=1)
    K1 = np.trace(tensor, axis1=1, axis2=2)
    assert is_spsd(K1)
    assert is_spsd(K)
    assert_array_almost_equal(K, K1)


def test_riemann_kernel_cref(rndstate, get_covmats):
    """Test Riemannian Kernel reference"""
    n_matrices, n_channels = 5, 3
    cov = get_covmats(n_matrices, n_channels)
    cref = mean_riemann(cov)
    K = kernel_riemann(cov, cov)
    K1 = kernel_riemann(cov, cov, cref)
    assert_array_equal(K, K1)


def test_riemann_kernel_x_y(rndstate, get_covmats):
    """Test Riemannian Kernel reference"""
    n_matrices, n_channels = 5, 3
    cov = get_covmats(n_matrices, n_channels)
    cov2 = get_covmats(n_matrices + 1, n_channels)
    K = kernel_riemann(cov, cov2)

    assert K.shape == (n_matrices, n_matrices + 1)


def test_generic_kernel(rndstate, get_covmats):
    """Test Riemannian Kernel reference"""
    n_matrices, n_channels = 5, 3
    cov = get_covmats(n_matrices, n_channels)
    K = kernel(cov, cov, np.eye(n_channels), metric='riemann')
    K1 = kernel_riemann(cov, cov, np.eye(n_channels))

    assert_array_equal(K, K1)
