import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from pytest import approx

from pyriemann.datasets.simulated import (
    _make_eyes,
    mat_kinds,
    make_matrices,
    make_masks,
    make_gaussian_blobs,
    make_outliers,
    make_classification_transfer,
)
from pyriemann.geometry.base import ctranspose
from pyriemann.geometry.distance import distance_riemann
from pyriemann.geometry.mean import mean_riemann
from pyriemann.geometry.test import (
    is_real, is_sym, is_hermitian,
    is_sym_pos_def as is_spd,
    is_sym_pos_semi_def as is_spsd,
    is_herm_pos_def as is_hpd,
    is_herm_pos_semi_def as is_hpsd,
)
from pyriemann.transfer import decode_domains

pytestmark = pytest.mark.numpy_only


@pytest.mark.parametrize("kind", mat_kinds)
def test_make_matrices_square(rndstate, kind):
    """Test make_matrices for square matrices."""
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

    if kind in ["inv", "cinv"]:
        assert np.all(np.linalg.det(X) != 0)
        return

    if kind in ["orth", "unit"]:
        eyes = _make_eyes(n_matrices, n_dim)
        assert_array_almost_equal(X @ ctranspose(X), eyes)
        assert_array_almost_equal(ctranspose(X) @ X, eyes)
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


@pytest.mark.parametrize("kind", ["real", "comp"])
@pytest.mark.parametrize("n_dim", [[3], [3, 4], [3, 4, 5]])
def test_make_matrices_nonsquare(rndstate, kind, n_dim):
    """Test make_matrices for non-square matrices."""
    n_matrices = 6
    X = make_matrices(
        n_matrices=n_matrices,
        n_dim=n_dim,
        kind=kind,
        rs=rndstate
    )
    assert X.shape == (n_matrices, *n_dim)


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


@pytest.mark.parametrize("n_classes", [2, 3, 4])
@pytest.mark.parametrize("n_domains", [2, 3])
@pytest.mark.parametrize("n_dim", [2, 3])
@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_make_classification_transfer(n_classes, n_domains, n_dim, kind):
    """Test classification transfer for several classes and domains."""
    n_matrices = 5
    class_names = [f"class_{i}" for i in range(n_classes)]
    domain_names = [f"domain_{i}" for i in range(n_domains)]
    X, y_enc = make_classification_transfer(
        n_matrices=n_matrices,
        random_state=17,
        class_names=class_names,
        domain_names=domain_names,
        n_dim=n_dim,
        kind=kind,
    )

    n_matrices_tot = n_matrices * n_classes * n_domains
    assert X.shape == (n_matrices_tot, n_dim, n_dim)
    assert y_enc.shape == (n_matrices_tot,)
    if kind == "spd":
        assert is_spd(X)
    else:
        assert is_hpd(X)
        assert not is_real(X)

    _, y, domains = decode_domains(X, y_enc)
    assert set(y) == set(class_names)
    assert set(domains) == set(domain_names)
    for domain in domain_names:
        for class_ in class_names:
            assert np.sum((domains == domain) & (y == class_)) == n_matrices


@pytest.mark.parametrize("n_dim", [2, 3, 4])
def test_make_classification_transfer_domain_sep(n_dim):
    """Test that domains are separated by the requested distance."""
    domain_seps = [2.0, 5.0]
    X, y_enc = make_classification_transfer(
        n_matrices=10,
        domain_sep=domain_seps,
        theta=[0.0, np.pi / 4],
        random_state=42,
        domain_names=["src", "tgt_0", "tgt_1"],
        n_dim=n_dim,
    )
    _, _, domains = decode_domains(X, y_enc)

    mean_src = mean_riemann(X[domains == "src"])
    assert distance_riemann(mean_src, np.eye(n_dim)) == approx(0, abs=1e-6)
    for i, domain in enumerate(["tgt_0", "tgt_1"]):
        mean_tgt = mean_riemann(X[domains == domain])
        assert distance_riemann(mean_src, mean_tgt) == approx(domain_seps[i])


def test_make_classification_transfer_default_is_unchanged():
    """Test that default parameters give 2x2 SPD matrices, 2 classes."""
    n_matrices = 6
    X, y_enc = make_classification_transfer(
        n_matrices=n_matrices, random_state=3
    )
    assert X.shape == (4 * n_matrices, 2, 2)
    assert is_spd(X)
    _, y, domains = decode_domains(X, y_enc)
    assert_array_equal(np.unique(y), ["1", "2"])
    assert_array_equal(
        np.unique(domains), ["source_domain", "target_domain"]
    )


def test_make_classification_transfer_errors():
    with pytest.raises(ValueError):  # only one class
        make_classification_transfer(n_matrices=2, class_names=[1])
    with pytest.raises(ValueError):  # only one domain
        make_classification_transfer(n_matrices=2, domain_names=["src"])
    with pytest.raises(ValueError):  # n_dim is too low
        make_classification_transfer(n_matrices=2, n_dim=1)
    with pytest.raises(ValueError):  # n_dim is not an integer
        make_classification_transfer(n_matrices=2, n_dim=2.0)
    with pytest.raises(ValueError):  # unsupported kind
        make_classification_transfer(n_matrices=2, kind="spsd")
    with pytest.raises(ValueError):  # not one theta per target domain
        make_classification_transfer(n_matrices=2, theta=[0.0, 1.0])
