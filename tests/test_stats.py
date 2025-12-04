import numpy as np
import pytest

from pyriemann.stats import PermutationDistance, PermutationModel
from pyriemann.spatialfilters import CSP


def test_permutation_badmode():
    """Test one way permutation test"""
    with pytest.raises(ValueError):
        PermutationDistance(mode="badmode")


@pytest.mark.parametrize("mode", ["ttest", "ftest"])
def test_permutation_mode(mode, get_mats, get_labels):
    """Test one way permutation test"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)
    p = PermutationDistance(100, mode=mode)
    p.test(X, y)


def test_permutation_pairwise(get_mats, get_labels):
    """Test one way permutation pairwise test"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)
    groups = np.array([0] * 3 + [1] * 3)
    # pairwise
    p = PermutationDistance(100, mode="pairwise")
    p.test(X, y)
    # with group
    p.test(X, y, groups=groups)


def test_permutation_pairwise_estimator(get_mats, get_labels):
    """Test one way permutation with estimator"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)
    # with custom estimator
    p = PermutationDistance(10, mode="pairwise", estimator=CSP(2, log=False))
    p.test(X, y)


def test_permutation_pairwise_unique(get_mats, get_labels):
    """Test one way permutation with estimator"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)
    # unique perms
    p = PermutationDistance(1000)
    p.test(X, y)


def test_permutation_pairwise_plot(get_mats, get_labels):
    """Test one way permutation with estimator"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)
    p = PermutationDistance(100, mode="pairwise")
    p.test(X, y)
    p.plot(nbins=2)


def test_permutation_model(get_mats, get_labels):
    """Test one way permutation test"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)
    # pairwise
    p = PermutationModel(10)
    p.test(X, y)
