from conftest import requires_matplotlib
import numpy as np
from pyriemann.stats import PermutationDistance, PermutationModel
from pyriemann.spatialfilters import CSP
import pytest


def test_permutation_badmode():
    """Test one way permutation test"""
    with pytest.raises(ValueError):
        PermutationDistance(mode="badmode")


@pytest.mark.parametrize("mode", ["ttest", "ftest"])
def test_permutation_mode(mode, get_covmats, get_labels):
    """Test one way permutation test"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    p = PermutationDistance(100, mode=mode)
    p.test(covmats, labels)


def test_permutation_pairwise(get_covmats, get_labels):
    """Test one way permutation pairwise test"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    groups = np.array([0] * 3 + [1] * 3)
    # pairwise
    p = PermutationDistance(100, mode="pairwise")
    p.test(covmats, labels)
    # with group
    p.test(covmats, labels, groups=groups)


def test_permutation_pairwise_estimator(get_covmats, get_labels):
    """Test one way permutation with estimator"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    # with custom estimator
    p = PermutationDistance(10, mode="pairwise", estimator=CSP(2, log=False))
    p.test(covmats, labels)


def test_permutation_pairwise_unique(get_covmats, get_labels):
    """Test one way permutation with estimator"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    # unique perms
    p = PermutationDistance(1000)
    p.test(covmats, labels)


@requires_matplotlib
def test_permutation_pairwise_plot(get_covmats, get_labels):
    """Test one way permutation with estimator"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    p = PermutationDistance(100, mode="pairwise")
    p.test(covmats, labels)
    p.plot(nbins=2)


def test_permutation_model(get_covmats, get_labels):
    """Test one way permutation test"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    # pairwise
    p = PermutationModel(10)
    p.test(covmats, labels)
