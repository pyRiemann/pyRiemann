import numpy as np
from nose.tools import assert_raises
from pyriemann.clustering import Kmeans, KmeansPerClassTransform, Potato


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


def test_Kmeans_init():
    """Test Kmeans"""
    covset = generate_cov(20, 3)
    labels = np.array([0, 1]).repeat(10)

    # init
    km = Kmeans(2)

    # fit
    km.fit(covset)

    # fit with init
    km = Kmeans(2, init=covset[0:2])
    km.fit(covset)

    # fit with labels
    km.fit(covset, y=labels)

    # predict
    km.predict(covset)

    # transform
    km.transform(covset)

    # n_jobs
    km = Kmeans(2, n_jobs=2)
    km.fit(covset)


def test_KmeansPCT_init():
    """Test Kmeans PCT"""
    covset = generate_cov(20, 3)
    labels = np.array([0, 1]).repeat(10)

    # init
    km = KmeansPerClassTransform(2)

    # fit
    km.fit(covset, labels)

    # transform
    km.transform(covset)


def test_Potato_init():
    """Test Potato"""
    covset = generate_cov(20, 3)
    labels = np.array([0, 1]).repeat(10)

    # init
    pt = Potato()

    # fit no labels
    pt.fit(covset)

    # fit with labels
    assert_raises(ValueError, pt.fit, covset, y=[1])
    assert_raises(ValueError, pt.fit, covset, y=[0] * 20)
    assert_raises(ValueError, pt.fit, covset, y=[0, 2, 3] + [1] * 17)
    pt.fit(covset, labels)

    # transform
    pt.transform(covset)

    # predict
    pt.predict(covset)

    # predict_proba
    pt.predict_proba(covset)

    # lower threshold
    pt = Potato(threshold=1)
    pt.fit(covset)
