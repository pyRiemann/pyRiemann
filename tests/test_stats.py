import numpy as np
from pyriemann.stats import (PermutationTest, PermutationTestTwoWay,
                             RiemannDistanceMetric)


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


def test_metric():
    """Test one way permutation test"""
    X = generate_cov(10, 3)
    rm = RiemannDistanceMetric()
    rm.pairwise(X)
    rm.pairwise(X, X)
    rm.get_metric()


def test_permutation_test():
    """Test one way permutation test"""
    covset = generate_cov(10, 30)
    labels = np.array([0, 1]).repeat(5)
    # base
    p = PermutationTest(10)
    p.test(covset, labels)
    # fit perm
    p = PermutationTest(10, fit_perms=True)
    p.test(covset, labels)
    # unique perms
    p = PermutationTest(1000)
    p.test(covset, labels)
    p.summary()
    p.plot(nbins=2)


def test_permutation2way_test():
    """Test two way permutation test"""
    covset = generate_cov(40, 2)
    labels = np.array([0, 1]).repeat(20)
    labels2 = np.array([4, 5, 2, 3]).repeat(10)
    p = PermutationTestTwoWay(200)
    p.test(covset, labels2, labels)
    p.summary()
    p.test(covset, labels2, labels, names=['a', 'b'])
    p.summary()

    p.plot(nbins=2)
