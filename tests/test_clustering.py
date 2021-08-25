from conftest import get_metrics
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from pyriemann.clustering import Kmeans, KmeansPerClassTransform, Potato


@pytest.mark.parametrize("clust", [Kmeans, KmeansPerClassTransform, Potato])
class ClusteringTestCase:
    def test_two_clusters(self, clust, get_covmats):
        n_clusters = 2
        n_trials, n_channels = 6, 3
        covmats = get_covmats(n_trials, n_channels)
        if clust is Kmeans:
            self.clf_predict(clust, covmats, n_clusters)
            self.clf_transform(clust, covmats, n_clusters)
            self.clf_jobs(clust, covmats, n_clusters)
            self.clf_centroids(clust, covmats, n_clusters)
        if clust is KmeansPerClassTransform:
            labels = np.array([0, 1]).repeat(n_trials // 2)
            self.clf_transform_per_class(clust, covmats, n_clusters, labels)
            self.clf_jobs(clust, covmats, n_clusters, labels)
        if clust is Potato:
            self.clf_transform(clust, covmats)
            self.clf_predict(clust, covmats)
            self.clf_predict_proba(clust, covmats)
            self.clf_partial_fit(clust, covmats)

    def test_three_clusters(self, clust, get_covmats):
        n_clusters = 3
        n_trials, n_channels = 6, 3
        covmats = get_covmats(n_trials, n_channels)
        if clust is Kmeans:
            self.clf_predict(clust, covmats, n_clusters)
            self.clf_transform(clust, covmats, n_clusters)
            self.clf_jobs(clust, covmats, n_clusters)
            self.clf_centroids(clust, covmats, n_clusters)
        if clust is KmeansPerClassTransform:
            labels = np.array([0, 1]).repeat(n_trials // 2)
            self.clf_transform_per_class(clust, covmats, n_clusters, labels)
            self.clf_jobs(clust, covmats, n_clusters, labels)


class TestRiemannianClustering(ClusteringTestCase):
    def clf_transform(self, clust, covmats, n_clusters=None):
        n_trials = covmats.shape[0]
        if n_clusters is None:
            clf = clust()
        else:
            clf = clust(n_clusters=n_clusters)
        clf.fit(covmats)
        transformed = clf.transform(covmats)
        if n_clusters is None:
            assert transformed.shape == (n_trials,)
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
        n_trials = covmats.shape[0]
        if n_clusters is None:
            clf = clust()
        else:
            clf = clust(n_clusters=n_clusters)
        clf.fit(covmats)
        predicted = clf.predict(covmats)
        assert predicted.shape == (n_trials,)

    def clf_predict_proba(self, clust, covmats):
        n_trials = covmats.shape[0]
        clf = clust()
        clf.fit(covmats)
        probabilities = clf.predict(covmats)
        assert probabilities.shape == (n_trials,)

    def clf_partial_fit(self, clust, covmats):
        clf = clust()
        clf.fit(covmats)
        clf.partial_fit(covmats)
        clf.partial_fit(covmats[np.newaxis, 0])  # fit one sample at a time


@pytest.mark.parametrize("clust", [Kmeans, KmeansPerClassTransform])
@pytest.mark.parametrize("init", ["random", "ndarray"])
@pytest.mark.parametrize("n_init", [1, 5])
@pytest.mark.parametrize("metric", get_metrics())
def test_km_init_metric(clust, init, n_init, metric, get_covmats):
    n_clusters, n_trials, n_channels = 2, 6, 3
    covmats = get_covmats(n_trials, n_channels)
    labels = np.array([0, 1]).repeat(n_trials // n_clusters)
    if init == "ndarray":
        clf = clust(
            n_clusters=n_clusters,
            metric=metric,
            init=covmats[:n_clusters],
            n_init=n_init,
        )
    else:
        clf = clust(
            n_clusters=n_clusters,
            metric=metric,
            init=init,
            n_init=n_init
        )
    clf.fit(covmats, labels)
    transformed = clf.transform(covmats)
    assert len(transformed) == n_trials


def test_Potato_equal_labels():
    with pytest.raises(ValueError):
        Potato(pos_label=0)


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


def test_Potato_partial_fit_diff_channels(get_covmats):
    n_trials, n_channels = 6, 3
    covmats = get_covmats(n_trials, n_channels)
    labels = np.array([0, 1]).repeat(n_trials // 2)
    pt = Potato().fit(covmats, labels)
    with pytest.raises(ValueError):  # unequal # of chans
        pt.partial_fit(get_covmats(2, n_channels + 1))


def test_Potato_partial_fit_no_poslabel(get_covmats):
    n_trials, n_channels = 6, 3
    covmats = get_covmats(n_trials, n_channels)
    labels = np.array([0, 1]).repeat(n_trials // 2)
    pt = Potato().fit(covmats, labels)
    with pytest.raises(ValueError):  # no positive labels
        pt.partial_fit(covmats, [0] * n_trials)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_Potato_partial_fit_alpha(alpha, get_covmats):
    n_trials, n_channels = 6, 3
    covmats = get_covmats(n_trials, n_channels)
    labels = np.array([0, 1]).repeat(n_trials // 2)
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
