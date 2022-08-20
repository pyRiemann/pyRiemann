import pytest
import numpy as np

from pyriemann.datasets.sampling import (sample_gaussian_spd,
                                         generate_random_spd_matrix)
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.test import is_sym_pos_def as is_spd


@pytest.mark.parametrize("n_jobs", [1, -1])
def test_sample_gaussian_spd(n_jobs):
    """Test Riemannian Gaussian sampling."""
    n_matrices, n_dim, sigma = 10, 8, 1.
    mean = np.eye(n_dim)
    X = sample_gaussian_spd(
        n_matrices, mean, sigma, random_state=42, n_jobs=n_jobs, sampling_method=None
        )
    assert X.shape == (n_matrices, n_dim, n_dim)  # X shape mismatch
    assert is_spd(X)  # X is an array of SPD matrices


def test_generate_random_spd_matrix():
    """Test generating random SPD matrix"""
    n_dim = 16
    X = generate_random_spd_matrix(n_dim, random_state=None)
    assert X.shape == (n_dim, n_dim)  # X shape mismatch
    assert is_spd(X)  # X is a SPD matrix


@pytest.mark.parametrize("n_jobs", [1, -1])
def test_sigma_gaussian_spd(n_jobs):
    """Test sigma parameter from Riemannian Gaussian sampling."""
    n_matrices, n_dim, sig_1, sig_2 = 10, 8, 1., 2.
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


def test_functions_error():
    n_matrices, n_dim = 10, 16
    mean, sigma = np.eye(n_dim), 2.
    with pytest.raises(ValueError):  # mean is not a matrix
        sample_gaussian_spd(n_matrices, np.ones(n_dim), sigma)
    with pytest.raises(ValueError):  # sigma is not a scalar
        sample_gaussian_spd(n_matrices, mean, np.ones(n_dim))
    with pytest.raises(ValueError):  # n_matrices is negative
        sample_gaussian_spd(-n_matrices, mean, sigma)
    with pytest.raises(ValueError):  # n_matrices is not an integer
        sample_gaussian_spd(4.2, mean, sigma)
    with pytest.raises(ValueError):  # n_dim is not an integer
        generate_random_spd_matrix(1.7)
    with pytest.raises(ValueError):  # n_dim is negative
        generate_random_spd_matrix(-n_dim)
