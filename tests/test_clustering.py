import numpy as np
from numpy.testing import assert_array_equal
import pytest

from conftest import get_metrics
from pyriemann.clustering import (
    Kmeans,
    KmeansPerClassTransform,
    Potato,
    PotatoField,
)


@pytest.mark.parametrize(
    "clust", [Kmeans, KmeansPerClassTransform, Potato, PotatoField]
)
class ClusteringTestCase:
    def test_two_clusters(self, clust, get_mats, get_labels):
        n_clusters = 2
        n_matrices, n_channels = 6, 4
        mats = get_mats(n_matrices, n_channels, "spd")
        if clust is Kmeans:
            self.clf_predict(clust, mats, n_clusters)
            self.clf_transform(clust, mats, n_clusters)
            self.clf_jobs(clust, mats, n_clusters)
            self.clf_centroids(clust, mats, n_clusters)
            self.clf_fit_independence(clust, mats)
        if clust is KmeansPerClassTransform:
            n_classes = 2
            labels = get_labels(n_matrices, n_classes)
            self.clf_transform_per_class(clust, mats, n_clusters, labels)
            self.clf_jobs(clust, mats, n_clusters, labels)
            self.clf_fit_labels_independence(clust, mats, labels)
        if clust is Potato:
            self.clf_transform(clust, mats)
            self.clf_predict(clust, mats)
            self.clf_predict_proba(clust, mats)
            self.clf_partial_fit(clust, mats)
            self.clf_fit_independence(clust, mats)
        if clust is PotatoField:
            n_potatoes = 3
            mats = [get_mats(n_matrices, n_channels, "spd"),
                       get_mats(n_matrices, n_channels + 2, "spd"),
                       get_mats(n_matrices, n_channels + 1, "spd")]
            self.clf_transform(clust, mats, n_potatoes)
            self.clf_predict(clust, mats, n_potatoes)
            self.clf_predict_proba(clust, mats, n_potatoes)
            self.clf_partial_fit(clust, mats, n_potatoes)
            self.clf_fit_independence(clust, mats, n_potatoes)

    def test_three_clusters(self, clust, get_mats, get_labels):
        n_clusters = 3
        n_matrices, n_channels = 6, 3
        mats = get_mats(n_matrices, n_channels, "spd")
        if clust is Kmeans:
            self.clf_predict(clust, mats, n_clusters)
            self.clf_transform(clust, mats, n_clusters)
            self.clf_jobs(clust, mats, n_clusters)
            self.clf_centroids(clust, mats, n_clusters)
            self.clf_fit_independence(clust, mats)
        if clust is KmeansPerClassTransform:
            n_classes = 2
            labels = get_labels(n_matrices, n_classes)
            self.clf_transform_per_class(clust, mats, n_clusters, labels)
            self.clf_jobs(clust, mats, n_clusters, labels)
            self.clf_fit_labels_independence(clust, mats, labels)


class TestRiemannianClustering(ClusteringTestCase):
    def clf_transform(self, clust, mats, n_clusters=None):
        n_matrices = len(mats)
        if n_clusters is None:
            clf = clust()
        elif clust is PotatoField:
            n_matrices = len(mats[0])
            clf = clust(n_potatoes=n_clusters)
        else:
            clf = clust(n_clusters=n_clusters)
        clf.fit(mats)
        transformed = clf.transform(mats)
        if n_clusters is None:
            assert transformed.shape == (n_matrices,)
        else:
            assert transformed.shape == (n_matrices, n_clusters)

    def clf_jobs(self, clust, mats, n_clusters, labels=None):
        n_matrices = mats.shape[0]
        clf = clust(n_clusters=n_clusters, n_jobs=2)
        if labels is None:
            clf.fit(mats)
        else:
            clf.fit(mats, labels)
        transformed = clf.transform(mats)
        assert len(transformed) == (n_matrices)

    def clf_centroids(self, clust, mats, n_clusters):
        _, n_channels, n_channels = mats.shape
        clf = clust(n_clusters=n_clusters).fit(mats)
        centroids = clf.centroids()
        shape = (n_clusters, n_channels, n_channels)
        assert np.array(centroids).shape == shape

    def clf_transform_per_class(self, clust, mats, n_clusters, labels):
        n_classes = len(np.unique(labels))
        n_matrices = mats.shape[0]
        clf = clust(n_clusters=n_clusters)
        clf.fit(mats, labels)
        transformed = clf.transform(mats)
        assert transformed.shape == (n_matrices, n_classes * n_clusters)

    def clf_predict(self, clust, mats, n_clusters=None):
        n_matrices = len(mats)
        if n_clusters is None:
            clf = clust()
        elif clust is PotatoField:
            n_matrices = len(mats[0])
            clf = clust(n_potatoes=n_clusters)
        else:
            clf = clust(n_clusters=n_clusters)
        clf.fit(mats)
        predicted = clf.predict(mats)
        assert predicted.shape == (n_matrices,)

    def clf_predict_proba(self, clust, mats, n=None):
        if n is None:
            n_matrices = len(mats)
            clf = clust()
        else:  # PotatoField
            n_matrices = len(mats[0])
            clf = clust(n_potatoes=n)
        clf.fit(mats)
        probabilities = clf.predict(mats)
        assert probabilities.shape == (n_matrices,)

    def clf_partial_fit(self, clust, mats, n=None):
        if n is None:
            clf = clust()
        else:  # PotatoField
            clf = clust(n_potatoes=n)
        clf.fit(mats)
        clf.partial_fit(mats)
        if n is None:
            clf.partial_fit(mats[np.newaxis, 0])  # fit one covmat at a time
        else:
            clf.partial_fit([c[np.newaxis, 0] for c in mats])

    def clf_fit_independence(self, clust, mats, n=None):
        if n is None:
            clf = clust()
        else:  # PotatoField
            clf = clust(n_potatoes=n)
        clf.fit(mats).transform(mats)
        # retraining with different size should erase previous fit
        if n is None:
            new_mats = mats[:, :-1, :-1]
        else:
            new_mats = [c[:, :-1, :-1] for c in mats]
        clf.fit(new_mats).transform(new_mats)

    def clf_fit_labels_independence(self, clust, mats, labels):
        clf = clust()
        clf.fit(mats, labels).transform(mats)
        # retraining with different size should erase previous fit
        new_mats = mats[:, :-1, :-1]
        clf.fit(new_mats, labels).transform(new_mats)


@pytest.mark.parametrize("clust", [Kmeans, KmeansPerClassTransform])
@pytest.mark.parametrize("init", ["random", "ndarray"])
@pytest.mark.parametrize("n_init", [1, 5])
@pytest.mark.parametrize("metric", get_metrics())
def test_kmeans_init(clust, init, n_init, metric, get_mats, get_labels):
    n_clusters, n_classes, n_matrices, n_channels = 2, 3, 9, 5
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)
    print(labels)

    if init == "ndarray":
        clf = clust(
            n_clusters=n_clusters,
            metric=metric,
            init=mats[:n_clusters],
            n_init=n_init,
        )
    else:
        clf = clust(
            n_clusters=n_clusters,
            metric=metric,
            init=init,
            n_init=n_init
        )
    clf.fit(mats, labels)

    if clust is Kmeans:
        assert clf.labels_.shape == (n_matrices,)
        centroids = clf.centroids()
        assert centroids.shape == (n_clusters, n_channels, n_channels)
    elif clust is KmeansPerClassTransform:
        assert clf.classes_.shape == (n_classes,)
        assert len(clf.covmeans_) <= n_clusters * n_classes

    dists = clf.transform(mats)
    if clust is Kmeans:
        assert dists.shape == (n_matrices, n_clusters)
    elif clust is KmeansPerClassTransform:
        assert dists.shape == (n_matrices, clf.covmeans_.shape[0])


def test_potato_fit_equal_labels(get_mats):
    n_matrices, n_channels = 6, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):
        Potato(pos_label=0).fit(mats)


@pytest.mark.parametrize("y_fail", [[1], [0] * 6, [0] * 7, [0, 1, 2] * 2])
def test_potato_fit_error(y_fail, get_mats):
    n_matrices, n_channels = 6, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):
        Potato().fit(mats, y=y_fail)


def test_potato_partial_fit_not_fitted(get_mats):
    n_matrices, n_channels = 6, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):  # potato not fitted
        Potato().partial_fit(mats)


def test_potato_partial_fit_diff_channels(get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)
    pt = Potato().fit(mats, labels)
    with pytest.raises(ValueError):  # unequal # of chans
        pt.partial_fit(get_mats(2, n_channels + 1, "spd"))


def test_potato_partial_fit_no_poslabel(get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)
    pt = Potato().fit(mats, labels)
    with pytest.raises(ValueError):  # no positive labels
        pt.partial_fit(mats, [0] * n_matrices)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_potato_partial_fit_alpha(alpha, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)
    pt = Potato().fit(mats, labels)
    with pytest.raises(ValueError):
        pt.partial_fit(mats, labels, alpha=alpha)


def test_potato_1channel(get_mats):
    n_matrices, n_channels = 6, 1
    mats_1chan = get_mats(n_matrices, n_channels, "spd")
    pt = Potato()
    pt.fit_transform(mats_1chan)
    pt.predict(mats_1chan)
    pt.predict_proba(mats_1chan)


def test_potato_threshold(get_mats):
    n_matrices, n_channels = 6, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    pt = Potato(threshold=2.5)
    pt.fit(mats)


def test_potato_specific_labels(get_mats):
    n_matrices, n_channels = 10, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    mats[-1] = 10 * np.eye(n_channels)
    pt = Potato(threshold=2.0, pos_label=2, neg_label=7)
    pt.fit(mats)
    assert_array_equal(np.unique(pt.predict(mats)), [2, 7])
    # fit with custom positive label
    pt.fit(mats, y=[2] * n_matrices)


def test_potato_field_fit(get_mats):
    n_potatoes, n_matrices, n_channels = 2, 6, 3
    mats1 = get_mats(n_matrices, n_channels, "spd")
    mats2 = get_mats(n_matrices, n_channels + 1, "spd")
    mats = [mats1, mats2]
    with pytest.raises(ValueError):  # n_potatoes too low
        PotatoField(n_potatoes=0).fit(mats)
    with pytest.raises(ValueError):   # p_threshold out of bounds
        PotatoField(p_threshold=0).fit(mats)
    with pytest.raises(ValueError):  # p_threshold out of bounds
        PotatoField(p_threshold=1).fit(mats)
    pf = PotatoField(n_potatoes=n_potatoes)
    with pytest.raises(ValueError):  # n_potatoes not equal to input length
        pf.fit([mats1, mats1, mats2])
    with pytest.raises(ValueError):  # n_matrices not equal
        pf.fit([mats1, mats2[:1]])


@pytest.mark.parametrize("method",
                         ["partial_fit", "transform", "predict_proba"])
def test_potato_field_method(get_mats, method):
    n_potatoes, n_matrices, n_channels = 2, 6, 3
    mats1 = get_mats(n_matrices, n_channels, "spd")
    mats2 = get_mats(n_matrices, n_channels + 1, "spd")
    mats = [mats1, mats2]
    pf = PotatoField(n_potatoes=n_potatoes).fit(mats)
    with pytest.raises(ValueError):  # n_potatoes not equal to input length
        getattr(pf, method)([mats1, mats1, mats2])
    with pytest.raises(ValueError):  # n_matrices not equal
        getattr(pf, method)([mats1, mats2[:1]])
