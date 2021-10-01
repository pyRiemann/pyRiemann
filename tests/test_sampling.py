from pyriemann.datasets.sampling import (sample_gaussian_spd,
                                         generate_random_spd_matrix)
from pyriemann.utils.distance import distance_riemann
from conftest import is_positive_definite
import numpy as np
import pytest


def test_sample_gaussian_spd():
    """Test Riemannian Gaussian sampling."""
    n_matrices, n_dim, sigma = 50, 16, 2.
    mean = np.eye(n_dim)
    X = sample_gaussian_spd(n_matrices, mean, sigma, random_state=None)
    assert X.shape == (n_matrices, n_dim, n_dim)  # X shape mismatch
    assert is_positive_definite(X)  # X is an array of SPD matrices


def test_generate_random_spd_matrix():
    """Test generating random SPD matrix"""
    n_dim = 16
    X = generate_random_spd_matrix(n_dim, random_state=None)
    assert X.shape == (n_dim, n_dim)  # X shape mismatch
    assert is_positive_definite(X)  # X is a SPD matrix


def test_sigma_gaussian_spd():
    """Test sigma parameter from Riemannian Gaussian sampling."""
    n_matrices, n_dim, sig_1, sig_2 = 50, 8, 1., 2.
    mean = np.eye(n_dim)
    X1 = sample_gaussian_spd(n_matrices, mean, sig_1, random_state=None)
    X2 = sample_gaussian_spd(n_matrices, mean, sig_2, random_state=None)
    avg_d1 = np.mean([distance_riemann(X1_i, mean) for X1_i in X1])
    avg_d2 = np.mean([distance_riemann(X2_i, mean) for X2_i in X2])
    assert avg_d1 < avg_d2
