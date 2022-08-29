import pytest

from pyriemann.utils.median import (
    median_euclid,
    median_riemann,
)


@pytest.mark.parametrize(
    "median",
    [
        median_euclid,
        median_riemann,
    ],
)
def test_median_shape(median, get_covmats):
    """Test the shape of median"""
    n_matrices, n_channels = 5, 3
    covmats = get_covmats(n_matrices, n_channels)
    C = median(covmats)
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize(
    "median", [
        median_euclid,
        median_riemann,
    ]
)
def test_median_shape_with_init(median, get_covmats):
    """Test the shape of median with init"""
    n_matrices, n_channels = 5, 3
    covmats = get_covmats(n_matrices, n_channels)
    C = median(covmats, init=covmats[0])
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize(
    "median", [
        median_euclid,
        median_riemann,
    ]
)
def test_median_warning_convergence(median, get_covmats):
    """Test warning for convergence not reached"""
    n_matrices, n_channels = 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    with pytest.warns(UserWarning):
        median(covmats, maxiter=0)
