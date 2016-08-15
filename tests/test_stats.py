import numpy as np
from pyriemann.stats import PermutationDistance, PermutationModel


def generate_cov(Nt, Ne):
    """Generate a set of cavariances matrices for test purpose."""
    rs = np.random.RandomState(1234)
    diags = 2.0 + 0.1 * rs.randn(Nt, Ne)
    A = 2*rs.rand(Ne, Ne) - 1
    A /= np.atleast_2d(np.sqrt(np.sum(A**2, 1))).T
    covmats = np.empty((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.dot(np.dot(A, np.diag(diags[i])), A.T)
    return covmats


def test_permutation_distance():
    """Test one way permutation test"""
    covset = generate_cov(10, 30)
    labels = np.array([0, 1]).repeat(5)
    # pairwise
    p = PermutationDistance(100, mode='pairwise')
    p.test(covset, labels)
    # t-test
    p = PermutationDistance(100, mode='ttest')
    p.test(covset, labels)
    # f-test
    p = PermutationDistance(100, mode='ftest')
    p.test(covset, labels)
    # unique perms
    p = PermutationDistance(1000)
    p.test(covset, labels)
    p.plot(nbins=2)


def test_permutation_model():
    """Test one way permutation test"""
    covset = generate_cov(10, 30)
    labels = np.array([0, 1]).repeat(5)
    # pairwise
    p = PermutationModel(10)
    p.test(covset, labels)
