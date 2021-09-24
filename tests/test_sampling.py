from pyriemann.datasets.sampling import (sample_gaussian_spd,
                                         generate_random_spd_matrix)
from conftest import is_positive_definite
import numpy as np


def test_sample_gaussian_spd():
    """Test Riemannian Gaussian sampling."""
    n_dim = 16
    n_matrices = 50
    mean = np.eye(n_dim)
    sigma = 2.00
    X = sample_gaussian_spd(n_matrices, mean, sigma, random_state=None)
    assert X.shape == (n_matrices, n_dim, n_dim)  # X shape mismatch
    assert is_positive_definite(X)  # X is an array of SPD matrices


def test_generate_random_spd_matrix():
    """Test generating random SPD matrix"""
    n_dim = 16
    X = generate_random_spd_matrix(n_dim, random_state=None)
    assert X.shape == (n_dim, n_dim)  # X shape mismatch
    assert is_positive_definite(X)  # X is a SPD matrix
