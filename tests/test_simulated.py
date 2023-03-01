import pytest
import numpy as np

from pyriemann.datasets.sampling import generate_random_spd_matrix
from pyriemann.datasets.simulated import (
    make_covariances,
    make_matrices,
    make_masks,
    make_gaussian_blobs,
    make_outliers,
)
from pyriemann.utils.test import (
    is_real,
    is_sym_pos_def as is_spd,
    is_sym_pos_semi_def as is_spsd,
    is_herm_pos_def as is_hpd,
    is_herm_pos_semi_def as is_hpsd,
)


def test_make_covariances(rndstate):
    """Test function for make covariances."""
    n_matrices, n_channels = 5, 4
    X, evals, evecs = make_covariances(n_matrices=n_matrices,
                                       n_channels=n_channels,
                                       return_params=True,
                                       rs=rndstate)
    assert X.shape == (n_matrices, n_channels, n_channels)  # X shape mismatch
    assert evals.shape == (n_matrices, n_channels)  # evals shape mismatch
    assert evecs.shape == (n_channels, n_channels)  # evecs shape mismatch


@pytest.mark.parametrize(
    "kind", ["real", "comp", "spd", "spsd", "hpd", "hpsd"]
)
def test_make_matrices(rndstate, kind):
    """Test function for make matrices."""
    n_matrices, n_dim = 5, 4
    X = make_matrices(
        n_matrices=n_matrices,
        n_dim=n_dim,
        kind=kind,
        return_params=False,
        eigvecs_same=False,
        rs=rndstate,
    )
    assert X.shape == (n_matrices, n_dim, n_dim)

    if kind == "real":
        assert is_real(X)
    elif kind == "comp":
        assert not is_real(X)
    elif kind == "spd":
        assert is_spd(X)
        assert is_spsd(X)
    elif kind == "spsd":
        assert is_spsd(X)
        assert not is_spd(X)
    elif kind == "hpd":
        assert is_hpd(X)
        assert is_hpsd(X)
    else:  # hpsd
        assert is_hpsd(X)
        assert not is_hpd(X)


@pytest.mark.parametrize("kind", ["spd", "spsd", "hpd", "hpsd"])
@pytest.mark.parametrize("eigvecs_same", [False, True])
def test_make_matrices_return(rndstate, kind, eigvecs_same):
    """Test function for make matrices."""
    n_matrices, n_dim = 5, 4
    X, evals, evecs = make_matrices(
        n_matrices=n_matrices,
        n_dim=n_dim,
        kind=kind,
        return_params=True,
        eigvecs_same=eigvecs_same,
        rs=rndstate,
    )
    assert X.shape == (n_matrices, n_dim, n_dim)
    assert evals.shape == (n_matrices, n_dim)
    if eigvecs_same:
        assert evecs.shape == (n_dim, n_dim)
    else:
        assert evecs.shape == (n_matrices, n_dim, n_dim)


def test_make_masks(rndstate):
    """Test function for make masks."""
    n_masks, n_dim0, n_dim1_min, = 5, 10, 3
    M = make_masks(n_masks, n_dim0, n_dim1_min, rndstate)

    for m in M:
        dim0, dim1 = m.shape
        assert dim0 == n_dim0  # 1st dim mismatch
        assert n_dim1_min <= dim1 <= n_dim0  # 2nd dim mismatch


def test_gaussian_blobs():
    """Test function for sampling Gaussian blobs."""
    n_matrices, n_dim = 5, 4
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
    n_matrices, n_dim, sigma = 5, 4, 1.
    mean = generate_random_spd_matrix(n_dim)
    X = make_outliers(n_matrices=n_matrices,
                      mean=mean,
                      sigma=sigma,
                      outlier_coeff=10,
                      random_state=None)
    assert X.shape == (n_matrices, n_dim, n_dim)  # X shape mismatch
    assert is_spd(X)  # X is an array of SPD matrices


def test_functions_error():
    n_matrices, n_dim, class_sep, class_disp = 5, 4, 2., 1.
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
    with pytest.raises(TypeError):  # n_dim is not an integer
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
