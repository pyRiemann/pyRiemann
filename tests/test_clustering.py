from conftest import get_metrics
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from pyriemann.clustering import (Kmeans, KmeansPerClassTransform, Potato,
                                  PotatoField)


@pytest.mark.parametrize(
    "clust", [Kmeans, KmeansPerClassTransform, Potato, PotatoField]
)
class ClusteringTestCase:
    def test_two_clusters(self, clust, get_covmats, get_labels):
        n_clusters = 2
        n_trials, n_channels = 6, 4
        covmats = get_covmats(n_trials, n_channels)
        if clust is Kmeans:
            self.clf_predict(clust, covmats, n_clusters)
            self.clf_transform(clust, covmats, n_clusters)
            self.clf_jobs(clust, covmats, n_clusters)
            self.clf_centroids(clust, covmats, n_clusters)
            self.clf_fit_independence(clust, covmats)
        if clust is KmeansPerClassTransform:
            n_classes = 2
            labels = get_labels(n_trials, n_classes)
            self.clf_transform_per_class(clust, covmats, n_clusters, labels)
            self.clf_jobs(clust, covmats, n_clusters, labels)
            self.clf_fit_labels_independence(clust, covmats, labels)
        if clust is Potato:
            self.clf_transform(clust, covmats)
            self.clf_predict(clust, covmats)
            self.clf_predict_proba(clust, covmats)
            self.clf_partial_fit(clust, covmats)
            self.clf_fit_independence(clust, covmats)
        if clust is PotatoField:
            n_potatoes = 3
            covmats = [get_covmats(n_trials, n_channels),
                       get_covmats(n_trials, n_channels + 2),
                       get_covmats(n_trials, n_channels + 1)]
            self.clf_transform(clust, covmats, n_potatoes)
            self.clf_predict(clust, covmats, n_potatoes)
            self.clf_predict_proba(clust, covmats, n_potatoes)
            self.clf_partial_fit(clust, covmats, n_potatoes)
            self.clf_fit_independence(clust, covmats, n_potatoes)

    def test_three_clusters(self, clust, get_covmats, get_labels):
        n_clusters = 3
        n_trials, n_channels = 6, 3
        covmats = get_covmats(n_trials, n_channels)
        if clust is Kmeans:
            self.clf_predict(clust, covmats, n_clusters)
            self.clf_transform(clust, covmats, n_clusters)
            self.clf_jobs(clust, covmats, n_clusters)
            self.clf_centroids(clust, covmats, n_clusters)
            self.clf_fit_independence(clust, covmats)
        if clust is KmeansPerClassTransform:
            n_classes = 2
            labels = get_labels(n_trials, n_classes)
            self.clf_transform_per_class(clust, covmats, n_clusters, labels)
            self.clf_jobs(clust, covmats, n_clusters, labels)
            self.clf_fit_labels_independence(clust, covmats, labels)


class TestRiemannianClustering(ClusteringTestCase):
    def clf_transform(self, clust, covmats, n_clusters=None):
        n_trials = len(covmats)
        if n_clusters is None:
            clf = clust()
        elif clust is PotatoField:
            n_trials = len(covmats[0])
            clf = clust(n_potatoes=n_clusters)
        else:
            clf = clust(n_clusters=n_clusters)
        clf.fit(covmats)
        transformed = clf.transform(covmats)
        if n_clusters is None:
            assert transformed.shape == (n_trials,)
        elif clust is PotatoField:
            assert transformed.shape == (n_clusters, n_trials)
        else:
            assert transformed.shape == (n_trials, n_clusters)

    def clf_jobs(self, clust, covmats, n_clusters, labels=None):
        n_trials = covmats.shape[0]
        clf = clust(n_clusters=n_clusters, n_jobs=2)
        if labels is None:
            clf.fit(covmats)
        else:
            clf.fit(covmats, labels)
        transformed = clf.transform(covmats)
        assert len(transformed) == (n_trials)

    def clf_centroids(self, clust, covmats, n_clusters):
        _, n_channels, n_channels = covmats.shape
        clf = clust(n_clusters=n_clusters).fit(covmats)
        centroids = clf.centroids()
        shape = (n_clusters, n_channels, n_channels)
        assert np.array(centroids).shape == shape

    def clf_transform_per_class(self, clust, covmats, n_clusters, labels):
        n_classes = len(np.unique(labels))
        n_trials = covmats.shape[0]
        clf = clust(n_clusters=n_clusters)
        clf.fit(covmats, labels)
        transformed = clf.transform(covmats)
        assert transformed.shape == (n_trials, n_classes * n_clusters)

    def clf_predict(self, clust, covmats, n_clusters=None):
        n_trials = len(covmats)
        if n_clusters is None:
            clf = clust()
        elif clust is PotatoField:
            n_trials = len(covmats[0])
            clf = clust(n_potatoes=n_clusters)
        else:
            clf = clust(n_clusters=n_clusters)
        clf.fit(covmats)
        predicted = clf.predict(covmats)
        assert predicted.shape == (n_trials,)

    def clf_predict_proba(self, clust, covmats, n=None):
        if n is None:
            n_trials = len(covmats)
            clf = clust()
        else:  # PotatoField
            n_trials = len(covmats[0])
            clf = clust(n_potatoes=n)
        clf.fit(covmats)
        probabilities = clf.predict(covmats)
        assert probabilities.shape == (n_trials,)

    def clf_partial_fit(self, clust, covmats, n=None):
        if n is None:
            clf = clust()
        else:  # PotatoField
            clf = clust(n_potatoes=n)
        clf.fit(covmats)
        clf.partial_fit(covmats)
        if n is None:
            clf.partial_fit(covmats[np.newaxis, 0])  # fit one covmat at a time
        else:
            clf.partial_fit([c[np.newaxis, 0] for c in covmats])

    def clf_fit_independence(self, clust, covmats, n=None):
        if n is None:
            clf = clust()
        else:  # PotatoField
            clf = clust(n_potatoes=n)
        clf.fit(covmats).transform(covmats)
        # retraining with different size should erase previous fit
        if n is None:
            new_covmats = covmats[:, :-1, :-1]
        else:
            new_covmats = [c[:, :-1, :-1] for c in covmats]
        clf.fit(new_covmats).transform(new_covmats)

    def clf_fit_labels_independence(self, clust, covmats, labels):
        clf = clust()
        clf.fit(covmats, labels).transform(covmats)
        # retraining with different size should erase previous fit
        new_covmats = covmats[:, :-1, :-1]
        clf.fit(new_covmats, labels).transform(new_covmats)


@pytest.mark.parametrize("clust", [Kmeans, KmeansPerClassTransform])
@pytest.mark.parametrize("init", ["random", "ndarray"])
@pytest.mark.parametrize("n_init", [1, 5])
@pytest.mark.parametrize("metric", get_metrics())
def test_km_init_metric(clust, init, n_init, metric, get_covmats, get_labels):
    n_clusters, n_trials, n_channels = 2, 6, 3
    covmats = get_covmats(n_trials, n_channels)
    labels = get_labels(n_trials, n_clusters)
    if init == "ndarray":
        clf = clust(
            n_clusters=n_clusters,
            metric=metric,
            init=covmats[:n_clusters],
            n_init=n_init,
        )
    else:
        clf = clust(
            n_clusters=n_clusters, metric=metric, init=init, n_init=n_init
        )
    clf.fit(covmats, labels)
    transformed = clf.transform(covmats)
    assert len(transformed) == n_trials


def test_Potato_fit_equal_labels(get_covmats):
    n_trials, n_channels = 6, 3
    covmats = get_covmats(n_trials, n_channels)
    with pytest.raises(ValueError):
        Potato(pos_label=0).fit(covmats)


@pytest.mark.parametrize("y_fail", [[1], [0] * 6, [0] * 7, [0, 1, 2] * 2])
def test_Potato_fit_error(y_fail, get_covmats):
    n_trials, n_channels = 6, 3
    covmats = get_covmats(n_trials, n_channels)
    with pytest.raises(ValueError):
        Potato().fit(covmats, y=y_fail)


def test_Potato_partial_fit_not_fitted(get_covmats):
    n_trials, n_channels = 6, 3
    covmats = get_covmats(n_trials, n_channels)
    with pytest.raises(ValueError):  # potato not fitted
        Potato().partial_fit(covmats)


def test_Potato_partial_fit_diff_channels(get_covmats, get_labels):
    n_trials, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_trials, n_channels)
    labels = get_labels(n_trials, n_classes)
    pt = Potato().fit(covmats, labels)
    with pytest.raises(ValueError):  # unequal # of chans
        pt.partial_fit(get_covmats(2, n_channels + 1))


def test_Potato_partial_fit_no_poslabel(get_covmats, get_labels):
    n_trials, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_trials, n_channels)
    labels = get_labels(n_trials, n_classes)
    pt = Potato().fit(covmats, labels)
    with pytest.raises(ValueError):  # no positive labels
        pt.partial_fit(covmats, [0] * n_trials)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_Potato_partial_fit_alpha(alpha, get_covmats, get_labels):
    n_trials, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_trials, n_channels)
    labels = get_labels(n_trials, n_classes)
    pt = Potato().fit(covmats, labels)
    with pytest.raises(ValueError):
        pt.partial_fit(covmats, labels, alpha=alpha)


def test_Potato_1channel(get_covmats):
    n_trials, n_channels = 6, 1
    covmats_1chan = get_covmats(n_trials, n_channels)
    pt = Potato()
    pt.fit_transform(covmats_1chan)
    pt.predict(covmats_1chan)
    pt.predict_proba(covmats_1chan)


def test_Potato_threshold(get_covmats):
    n_trials, n_channels = 6, 3
    covmats = get_covmats(n_trials, n_channels)
    pt = Potato(threshold=1)
    pt.fit(covmats)


def test_Potato_specific_labels(get_covmats):
    n_trials, n_channels = 6, 3
    covmats = get_covmats(n_trials, n_channels)
    pt = Potato(threshold=1, pos_label=2, neg_label=7)
    pt.fit(covmats)
    assert_array_equal(np.unique(pt.predict(covmats)), [2, 7])
    # fit with custom positive label
    pt.fit(covmats, y=[2] * n_trials)


def test_PotatoField_fit(get_covmats):
    n_potatoes, n_trials, n_channels = 2, 6, 3
    covmats1 = get_covmats(n_trials, n_channels)
    covmats2 = get_covmats(n_trials, n_channels + 1)
    covmats = [covmats1, covmats2]
    with pytest.raises(ValueError):  # n_potatoes too low
        PotatoField(n_potatoes=0).fit(covmats)
    with pytest.raises(ValueError):   # p_threshold out of bounds
        PotatoField(p_threshold=0).fit(covmats)
    with pytest.raises(ValueError):  # p_threshold out of bounds
        PotatoField(p_threshold=1).fit(covmats)
    pf = PotatoField(n_potatoes=n_potatoes)
    with pytest.raises(ValueError):  # n_potatoes not equal to input length
        pf.fit([covmats1, covmats1, covmats2])
    with pytest.raises(ValueError):  # n_trials not equal
        pf.fit([covmats1, covmats2[:1]])


def test_PotatoField_partialfit(get_covmats):
    n_potatoes, n_trials, n_channels = 2, 6, 3
    covmats1 = get_covmats(n_trials, n_channels)
    covmats2 = get_covmats(n_trials, n_channels + 1)
    covmats = [covmats1, covmats2]
    pf = PotatoField(n_potatoes=n_potatoes).fit(covmats)
    with pytest.raises(ValueError):  # n_potatoes not equal to input length
        pf.partial_fit([covmats1, covmats1, covmats2])
    with pytest.raises(ValueError):  # n_trials not equal
        pf.partial_fit([covmats1, covmats2[:1]])


def test_PotatoField_transform(get_covmats):
    n_potatoes, n_trials, n_channels = 2, 6, 3
    covmats1 = get_covmats(n_trials, n_channels)
    covmats2 = get_covmats(n_trials, n_channels + 1)
    covmats = [covmats1, covmats2]
    pf = PotatoField(n_potatoes=n_potatoes).fit(covmats)
    with pytest.raises(ValueError):  # n_potatoes not equal to input length
        pf.transform([covmats1, covmats1, covmats2])
    with pytest.raises(ValueError):  # n_trials not equal
        pf.transform([covmats1, covmats2[:1]])


def test_PotatoField_predictproba(get_covmats):
    n_potatoes, n_trials, n_channels = 2, 6, 3
    covmats1 = get_covmats(n_trials, n_channels)
    covmats2 = get_covmats(n_trials, n_channels + 1)
    covmats = [covmats1, covmats2]
    pf = PotatoField(n_potatoes=n_potatoes).fit(covmats)
    with pytest.raises(ValueError):  # n_potatoes not equal to input length
        pf.predict_proba([covmats1, covmats1, covmats2])
    with pytest.raises(ValueError):  # n_trials not equal
        pf.predict_proba([covmats1, covmats2[:1]])
