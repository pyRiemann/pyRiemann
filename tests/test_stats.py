from conftest import covmats, requires_matplotlib  # noqa: F401
import numpy as np
from pyriemann.stats import PermutationDistance, PermutationModel
from pyriemann.spatialfilters import CSP
import pytest


def test_permutation_badmode(covmats):  # noqa: F811
    """Test one way permutation test"""
    with pytest.raises(ValueError):
        PermutationDistance(mode="badmode")


@pytest.mark.parametrize("mode", ["ttest", "ftest"])
def test_permutation_mode(mode, covmats):  # noqa: F811
    """Test one way permutation test"""
    labels = np.array([0, 1]).repeat(3)
    p = PermutationDistance(100, mode=mode)
    p.test(covmats, labels)


def test_permutation_pairwise(covmats):  # noqa: F811
    """Test one way permutation pairwise test"""
    labels = np.array([0, 1]).repeat(3)
    groups = np.array([0] * 3 + [1] * 3)
    # pairwise
    p = PermutationDistance(100, mode="pairwise")
    p.test(covmats, labels)
    # with group
    p.test(covmats, labels, groups=groups)


def test_permutation_pairwise_estimator(covmats):  # noqa: F811
    """Test one way permutation with estimator"""
    labels = np.array([0, 1]).repeat(3)
    # with custom estimator
    p = PermutationDistance(10, mode="pairwise", estimator=CSP(2, log=False))
    p.test(covmats, labels)


def test_permutation_pairwise_unique(covmats):  # noqa: F811
    """Test one way permutation with estimator"""
    labels = np.array([0, 1]).repeat(3)
    # unique perms
    p = PermutationDistance(1000)
    p.test(covmats, labels)


@requires_matplotlib
def test_permutation_pairwise_plot(covmats):  # noqa: F811
    """Test one way permutation with estimator"""
    labels = np.array([0, 1]).repeat(3)
    p = PermutationDistance(100, mode="pairwise")
    p.test(covmats, labels)
    p.plot(nbins=2)


def test_permutation_model(covmats):  # noqa: F811
    """Test one way permutation test"""
    labels = np.array([0, 1]).repeat(3)
    # pairwise
    p = PermutationModel(10)
    p.test(covmats, labels)
