import pytest
import numpy as np

from pyriemann.datasets.sampling import generate_random_spd_matrix
from pyriemann.datasets.simulated import (make_gaussian_blobs,
                                          make_outliers)
from pyriemann.utils.test import is_sym_pos_def as is_spd


def test_gaussian_blobs():
    """Test function for sampling Gaussian blobs."""
    n_matrices, n_dim = 50, 8
    X, y = make_gaussian_blobs(n_matrices=n_matrices,
                               n_dim=n_dim,
                               class_sep=2.0,
                               class_disp=1.0,
                               return_centers=False,
                               random_state=None)
    assert X.shape == (2*n_matrices, n_dim, n_dim)  # X shape mismatch
    assert is_spd(X)  # X is an array of SPD matrices
    assert y.shape == (2*n_matrices,)  # y shape mismatch
    assert np.unique(y).shape == (2,)  # Unexpected number of classes
    assert sum(y == 0) == n_matrices  # Unexpected number of samples in class 0
    assert sum(y == 1) == n_matrices  # Unexpected number of samples in class 1
    _, _, centers = make_gaussian_blobs(n_matrices=1,
                                        n_dim=n_dim,
                                        class_sep=2.0,
                                        class_disp=1.0,
                                        return_centers=True,
                                        random_state=None)
    assert centers.shape == (2, n_dim, n_dim)  # centers shape mismatch


def test_generate_random_spd_matrix():
    """Test function for sampling outliers"""
    n_matrices, n_dim, sigma = 100, 8, 1.
    mean = generate_random_spd_matrix(n_dim)
    X = make_outliers(n_matrices=n_matrices,
                      mean=mean,
                      sigma=sigma,
                      outlier_coeff=10,
                      random_state=None)
    assert X.shape == (n_matrices, n_dim, n_dim)  # X shape mismatch
    assert is_spd(X)  # X is an array of SPD matrices


def test_functions_error():
    n_matrices, n_dim, class_sep, class_disp = 10, 16, 2., 1.
    with pytest.raises(ValueError):  # n_matrices is not an integer
        make_gaussian_blobs(n_matrices=float(n_matrices),
                            n_dim=n_dim,
                            class_sep=class_sep,
                            class_disp=class_disp)
    with pytest.raises(ValueError):  # n_matrices is negative
        make_gaussian_blobs(n_matrices=-n_matrices,
                            n_dim=n_dim,
                            class_sep=class_sep,
                            class_disp=class_disp)
    with pytest.raises(ValueError):  # n_dim is not an integer
        make_gaussian_blobs(n_matrices=n_matrices,
                            n_dim=float(n_dim),
                            class_sep=class_sep,
                            class_disp=class_disp)
    with pytest.raises(ValueError):  # n_dim is negative
        make_gaussian_blobs(n_matrices=n_matrices,
                            n_dim=-n_dim,
                            class_sep=class_sep,
                            class_disp=class_disp)
    with pytest.raises(ValueError):  # class_sep is not a scalar
        make_gaussian_blobs(n_matrices=n_matrices,
                            n_dim=n_dim,
                            class_sep=class_sep * np.ones(n_dim),
                            class_disp=class_disp)
    with pytest.raises(ValueError):  # class_disp is not a scalar
        make_gaussian_blobs(n_matrices=n_matrices,
                            n_dim=n_dim,
                            class_sep=class_sep,
                            class_disp=class_disp * np.ones(n_dim))
