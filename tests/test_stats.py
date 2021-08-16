from conftest import covmats
import numpy as np
from pyriemann.stats import PermutationDistance, PermutationModel
from pyriemann.spatialfilters import CSP
import pytest


def test_permutation_distance(covmats):
    """Test one way permutation test"""
    labels = np.array([0, 1]).repeat(3)
    groups = np.array([0] * 3 + [1] * 3)

    with pytest.raises(ValueError):
        PermutationDistance(mode="badmode")

    # pairwise
    p = PermutationDistance(100, mode="pairwise")
    p.test(covmats, labels)
    # with group
    p.test(covmats, labels, groups=groups)
    # t-test
    p = PermutationDistance(100, mode="ttest")
    p.test(covmats, labels)
    # f-test
    p = PermutationDistance(100, mode="ftest")
    p.test(covmats, labels)
    # with custom estimator
    p = PermutationDistance(10, mode="pairwise", estimator=CSP(2, log=False))
    p.test(covmats, labels)
    # unique perms
    p = PermutationDistance(1000)
    p.test(covmats, labels)
    try:
        import matplotlib.pyplot as plt

        p.plot(nbins=2)
    except ImportError:
        pass


def test_permutation_model(covmats):
    """Test one way permutation test"""
    labels = np.array([0, 1]).repeat(3)
    # pairwise
    p = PermutationModel(10)
    p.test(covmats, labels)
