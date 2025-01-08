import numpy as np
import pytest

from pyriemann.datasets.sampling import sample_gaussian_spd
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.test import is_sym_pos_def as is_spd


@pytest.mark.parametrize("n_jobs", [1, -1])
@pytest.mark.parametrize("sampling_method", ["auto", "slice", "rejection"])
def test_sample_gaussian_spd_dim2(n_jobs, sampling_method):
    """Test Riemannian Gaussian sampling for dim=2."""
    n_matrices, n_dim, sigma = 5, 2, 1.
    mean = np.eye(n_dim)
    X = sample_gaussian_spd(n_matrices, mean, sigma, random_state=42,
                            n_jobs=n_jobs, sampling_method=sampling_method)
    assert X.shape == (n_matrices, n_dim, n_dim)  # X shape mismatch
    assert is_spd(X)  # X is an array of SPD matrices


@pytest.mark.parametrize("n_dim", [3, 4])
@pytest.mark.parametrize("n_jobs", [1, -1])
@pytest.mark.parametrize("sampling_method", ["auto", "slice"])
def test_sample_gaussian_spd_dimsup(n_dim, n_jobs, sampling_method):
    """Test Riemannian Gaussian sampling for dim>2."""
    n_matrices, sigma = 5, 1.
    mean = np.eye(n_dim)
    X = sample_gaussian_spd(n_matrices, mean, sigma, random_state=42,
                            n_jobs=n_jobs, sampling_method=sampling_method)
    assert X.shape == (n_matrices, n_dim, n_dim)  # X shape mismatch
    assert is_spd(X)  # X is an array of SPD matrices


def test_sample_gaussian_spd_error():
    with pytest.raises(ValueError):  # unknown sampling method
        sample_gaussian_spd(5, np.eye(2), 1., sampling_method="blabla")
    with pytest.raises(ValueError):  # dim=3 not yet supported with rejection
        n_dim = 3
        sample_gaussian_spd(5, np.eye(n_dim), 1., sampling_method="rejection")


@pytest.mark.parametrize("n_jobs", [1, -1])
def test_sample_gaussian_spd_sigma(n_jobs):
    """Test sigma parameter from Riemannian Gaussian sampling."""
    n_matrices, n_dim, sig_1, sig_2 = 5, 4, 1., 2.
    mean = np.eye(n_dim)
    X1 = sample_gaussian_spd(
        n_matrices, mean, sig_1, random_state=42, n_jobs=n_jobs
        )
    X2 = sample_gaussian_spd(
        n_matrices, mean, sig_2, random_state=66, n_jobs=n_jobs
        )
    avg_d1 = np.mean([distance_riemann(X1_i, mean) for X1_i in X1])
    avg_d2 = np.mean([distance_riemann(X2_i, mean) for X2_i in X2])
    assert avg_d1 < avg_d2


def test_sample_gaussian_spd_sigma_errors():
    n_matrices, n_dim = 3, 4
    mean, sigma = np.eye(n_dim), 2.
    with pytest.raises(ValueError):  # mean is not a matrix
        sample_gaussian_spd(n_matrices, np.ones(n_dim), sigma)
    with pytest.raises(ValueError):  # sigma is not a scalar
        sample_gaussian_spd(n_matrices, mean, np.ones(n_dim))
    with pytest.raises(ValueError):  # n_matrices is negative
        sample_gaussian_spd(-n_matrices, mean, sigma)
    with pytest.raises(ValueError):  # n_matrices is not an integer
        sample_gaussian_spd(4.2, mean, sigma)
