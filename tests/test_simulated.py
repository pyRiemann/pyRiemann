import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from pyriemann.datasets.simulated import (
    mat_kinds,
    make_matrices,
    make_masks,
    make_gaussian_blobs,
    make_outliers,
)
from pyriemann.utils.base import ctranspose
from pyriemann.utils.test import (
    is_real, is_sym, is_hermitian,
    is_sym_pos_def as is_spd,
    is_sym_pos_semi_def as is_spsd,
    is_herm_pos_def as is_hpd,
    is_herm_pos_semi_def as is_hpsd,
)


@pytest.mark.parametrize("kind", mat_kinds)
def test_make_matrices(rndstate, kind):
    """Test function for make matrices."""
    n_matrices, n_dim = 5, 3
    X = make_matrices(
        n_matrices=n_matrices,
        n_dim=n_dim,
        kind=kind,
        rs=rndstate,
        return_params=False,
        evals_low=0.7,
        evals_high=3.0,
        eigvecs_same=False,
        eigvecs_mean=1.0,
        eigvecs_std=2.0,
    )
    assert X.shape == (n_matrices, n_dim, n_dim)

    if kind == "real":
        assert is_real(X)
        return
    if kind == "comp":
        assert not is_real(X)
        return

    # all other types are symmetric or Hermitian
    assert_array_almost_equal(X, ctranspose(X))

    if kind == "sym":
        assert is_sym(X)
    elif kind == "herm":
        assert is_hermitian(X)
    elif kind == "spd":
        assert is_spd(X)
        assert is_spsd(X)
    elif kind == "spsd":
        assert is_spsd(X)
        assert not is_spd(X, tol=1e-9)
    elif kind == "hpd":
        assert is_hpd(X)
        assert is_hpsd(X)
    elif kind == "hpsd":
        assert is_hpsd(X)
        assert not is_hpd(X, tol=1e-9)


@pytest.mark.parametrize("kind", ["spd", "spsd", "hpd", "hpsd"])
@pytest.mark.parametrize("n_matrices", [3, 4, 5])
@pytest.mark.parametrize("n_dim", [2, 3, 4])
@pytest.mark.parametrize("eigvecs_same", [False, True])
def test_make_matrices_return(rndstate, kind, n_matrices, n_dim, eigvecs_same):
    """Test function for make matrices."""
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


def test_gaussian_blobs_errors():
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


@pytest.mark.parametrize("n_matrices", [3, 4, 5])
@pytest.mark.parametrize("n_dim", [2, 3, 4])
def test_make_outliers(rndstate, get_mats, n_matrices, n_dim):
    mean, sigma = get_mats(1, n_dim, "spd")[0], 0.5
    X = make_outliers(n_matrices, mean, sigma, random_state=None)
    assert X.shape == (n_matrices, n_dim, n_dim)
