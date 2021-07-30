import numpy as np
from numpy.testing import assert_array_equal
import pytest
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
    n_trials, n_channels = 20, 3
    covset = generate_cov(n_trials, n_channels)
    cov = covset[0][np.newaxis, ...]  # to test potato with a single trial
    labels = np.array([0, 1]).repeat(n_trials // 2)

    # init
    with pytest.raises(ValueError):  # positive and neg labels equal
        Potato(pos_label=0)
    pt = Potato()

    # fit no labels
    pt.fit(covset)

    # fit with labels
    with pytest.raises(ValueError):
        pt.fit(covset, y=[1])
    with pytest.raises(ValueError):
        pt.fit(covset, y=[0] * 20)
    with pytest.raises(ValueError):
        pt.fit(covset, y=[0, 2, 3] + [1] * 17)
    pt.fit(covset, labels)

    # partial_fit
    with pytest.raises(ValueError):  # potato not fitted
        Potato().partial_fit(covset)
    with pytest.raises(ValueError):  # unequal # of chans
        pt.partial_fit(generate_cov(2, n_channels + 1))
    with pytest.raises(ValueError):  # alpha < 0
        pt.partial_fit(covset, labels, alpha=-0.1)
    with pytest.raises(ValueError):  # alpha > 1
        pt.partial_fit(covset, labels, alpha=1.1)
    with pytest.raises(ValueError):  # no positive labels
        pt.partial_fit(covset, [0] * n_trials)
    pt.partial_fit(covset, labels, alpha=0.6)
    pt.partial_fit(cov, alpha=0.1)

    # transform
    pt.transform(covset)
    pt.transform(cov)

    # predict
    pt.predict(covset)
    pt.predict(cov)

    # predict_proba
    pt.predict_proba(covset)
    pt.predict_proba(cov)

    # potato with a single channel
    covset_1chan = generate_cov(n_trials, 1)
    pt.fit_transform(covset_1chan)
    pt.predict(covset_1chan)
    pt.predict_proba(covset_1chan)

    # lower threshold
    pt = Potato(threshold=1)
    pt.fit(covset)

    # test positive labels
    pt = Potato(threshold=1, pos_label=2, neg_label=7)
    pt.fit(covset)
    assert_array_equal(np.unique(pt.predict(covset)), [2, 7])
    # fit with custom positive label
    pt.fit(covset, y=[2]*n_trials)
