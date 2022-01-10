from pyriemann.utils.kernel import (kernel,
                                    kernel_euclid,
                                    kernel_logeuclid,
                                    kernel_riemann)
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.test import is_pos_semi_def as is_spsd

from numpy.testing import assert_array_equal
import pytest

rker_str = ['riemann', 'euclid', 'logeuclid']
rker_fct = [kernel_riemann, kernel_euclid, kernel_logeuclid]


@pytest.mark.parametrize("ker", rker_fct)
def test_kernel(ker, rndstate, get_covmats):
    """Test Kernel build"""
    n_matrices, n_channels = 12, 3
    cov = get_covmats(n_matrices, n_channels)
    K = ker(cov, cov)
    assert is_spsd(K)
    assert K.shape == (n_matrices, n_matrices)


@pytest.mark.parametrize("ker", rker_str)
def test_kernel_cref(ker, rndstate, get_covmats):
    """Test Kernel reference"""
    n_matrices, n_channels = 5, 3
    cov = get_covmats(n_matrices, n_channels)
    cref = mean_covariance(cov, metric=ker)
    K = kernel(cov, cov, metric=ker)
    K1 = kernel(cov, cov, cref, metric=ker)
    assert_array_equal(K, K1)


@pytest.mark.parametrize("ker", rker_str)
def test_riemann_kernel_x_y(ker, rndstate, get_covmats):
    """Test Riemannian Kernel for different X and Y."""
    n_matrices, n_channels = 5, 3
    cov = get_covmats(n_matrices, n_channels)
    cov2 = get_covmats(n_matrices + 1, n_channels)
    K = kernel(cov, cov2, metric=ker)

    assert K.shape == (n_matrices, n_matrices + 1)


@pytest.mark.parametrize("ker", rker_str)
def test_metric_string(ker, rndstate, get_covmats):
    """Test generic Kernel function."""
    n_matrices, n_channels = 5, 3
    cov = get_covmats(n_matrices, n_channels)
    K = globals()[f'kernel_{ker}'](cov)
    K1 = kernel(cov, metric=ker)
    assert_array_equal(K, K1)


def test_metric_string_error(rndstate, get_covmats):
    """Test generic Kernel function error raise."""
    n_matrices, n_channels = 5, 3
    cov = get_covmats(n_matrices, n_channels)
    with pytest.raises(ValueError):
        kernel(cov, metric='foo')


@pytest.mark.parametrize("ker", rker_str)
def test_input_dimension_error(ker, rndstate, get_covmats):
    """Test errors for incorrect dimension."""
    n_matrices, n_channels = 5, 3
    cov = get_covmats(n_matrices, n_channels)
    cov2 = get_covmats(n_matrices, n_channels+1)
    cref = get_covmats(1, n_channels+1)[0]

    with pytest.raises(AssertionError):
        kernel(cov, Cref=cref, metric=ker)
    with pytest.raises(AssertionError):
        kernel(cov, cov2, metric=ker)
