import numpy as np
from numpy.testing import assert_array_equal
import pytest

from conftest import get_metrics
from pyriemann.clustering import (
    Kmeans,
    KmeansPerClassTransform,
    MeanShift,
    Potato,
    PotatoField,
)

clusts = [Kmeans, KmeansPerClassTransform, MeanShift, Potato, PotatoField]


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("clust", clusts)
def test_clustering_two_clusters(kind, clust,
                                 get_mats, get_labels, get_weights):
    n_clusters, n_matrices, n_channels = 2, 6, 4
    mats = get_mats(n_matrices, n_channels, kind)
    weights = get_weights(n_matrices)

    if clust is Kmeans:
        clt_fit(clust, mats, n_clusters, None)
        clt_predict(clust, mats, n_clusters)
        clt_transform(clust, mats, n_clusters)
        clt_jobs(clust, mats, n_clusters)
        clt_centroids(clust, mats, n_clusters)
        clt_fit_independence(clust, mats)

    if clust is KmeansPerClassTransform:
        n_classes = n_clusters
        labels = get_labels(n_matrices, n_classes)
        clt_fit(clust, mats, n_clusters, labels)
        clt_transform_per_class(clust, mats, n_clusters, labels)
        clt_jobs(clust, mats, n_clusters, labels)
        clt_fit_labels_independence(clust, mats, labels)

    if clust is MeanShift:
        clt_fit(clust, mats, n_clusters, None)
        clt_predict(clust, mats)

    if clust is Potato:
        clt_fit(clust, mats, n_clusters, None)
        clt_fit_weights(clust, mats, weights)
        clt_transform(clust, mats)
        clt_predict(clust, mats)
        clt_predict_proba(clust, mats)
        clt_partial_fit(clust, mats)
        clt_fit_independence(clust, mats)
        clt_fittransform(clust, mats)

    if clust is PotatoField:
        n_potatoes = 3
        mats = [get_mats(n_matrices, n_channels, kind),
                get_mats(n_matrices, n_channels + 2, kind),
                get_mats(n_matrices, n_channels + 1, kind)]
        clt_fit_weights(clust, mats, weights)
        clt_transform(clust, mats, n_potatoes)
        clt_predict(clust, mats, n_potatoes)
        clt_predict_proba(clust, mats, n_potatoes)
        clt_partial_fit(clust, mats, n_potatoes)
        clt_fit_independence(clust, mats, n_potatoes)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("clust", clusts)
def test_clustering_three_clusters(kind, clust, get_mats, get_labels):
    n_clusters, n_matrices, n_channels = 3, 6, 2
    mats = get_mats(n_matrices, n_channels, kind)

    if clust is Kmeans:
        clt_fit(clust, mats, n_clusters, None)
        clt_predict(clust, mats, n_clusters)
        clt_transform(clust, mats, n_clusters)
        clt_jobs(clust, mats, n_clusters)
        clt_centroids(clust, mats, n_clusters)
        clt_fit_independence(clust, mats)

    if clust is KmeansPerClassTransform:
        n_classes = 2
        labels = get_labels(n_matrices, n_classes)
        clt_transform_per_class(clust, mats, n_clusters, labels)
        clt_jobs(clust, mats, n_clusters, labels)
        clt_fit_labels_independence(clust, mats, labels)

    if clust is MeanShift:
        clt_fit(clust, mats, n_clusters, None)
        clt_predict(clust, mats)


def clt_fit(clust, mats, n_clusters, labels):
    n_matrices, n_channels, _ = mats.shape
    n_classes = len(np.unique(labels))

    clt = clust()
    clt.fit(mats, labels)

    if clust is Kmeans:
        assert clt.labels_.shape == (n_matrices,)
        assert len(clt.mdm_.covmeans_) <= n_clusters
        assert clt.mdm_.covmeans_.shape[1:] == (n_channels, n_channels)
        assert_array_equal(clt.mdm_.covmeans_, clt.centroids())
    if clust is KmeansPerClassTransform:
        assert clt.classes_.shape == (n_classes,)
        assert len(clt.covmeans_) <= n_clusters * n_classes
        assert clt.covmeans_.shape[1:] == (n_channels, n_channels)
    if clust is MeanShift:
        assert clt.labels_.shape == (n_matrices,)
        assert len(clt.modes_) >= 1
        assert clt.modes_.shape[1:] == (n_channels, n_channels)
    if clust is Potato:
        assert clt.covmean_.shape == (n_channels, n_channels)


def clt_fit_weights(clust, mats, weights):
    if clust is PotatoField:
        clt = clust(n_potatoes=len(mats))
    else:
        clt = clust()
    clt.fit(mats, sample_weight=weights)


def clt_transform(clust, mats, n_clusters=None):
    n_matrices = len(mats)
    if n_clusters is None:
        clt = clust()
    elif clust is PotatoField:
        n_matrices = len(mats[0])
        clt = clust(n_potatoes=n_clusters)
    else:
        clt = clust(n_clusters=n_clusters)
    clt.fit(mats)
    transf = clt.transform(mats)
    if n_clusters is None:
        assert transf.shape == (n_matrices,)
    else:
        assert transf.shape == (n_matrices, n_clusters)


def clt_jobs(clust, mats, n_clusters, labels=None):
    n_matrices = mats.shape[0]
    clt = clust(n_clusters=n_clusters, n_jobs=2)
    if labels is None:
        clt.fit(mats)
    else:
        clt.fit(mats, labels)

    if clust in [Kmeans, KmeansPerClassTransform]:
        transf = clt.transform(mats)
        assert len(transf) == (n_matrices)


def clt_centroids(clust, mats, n_clusters):
    n_channels = mats.shape[-1]
    clt = clust(n_clusters=n_clusters).fit(mats)
    centroids = clt.centroids()
    assert centroids.shape == (n_clusters, n_channels, n_channels)


def clt_transform_per_class(clust, mats, n_clusters, labels):
    n_classes = len(np.unique(labels))
    n_matrices = mats.shape[0]
    clt = clust(n_clusters=n_clusters)
    clt.fit(mats, labels)
    transf = clt.transform(mats)
    assert transf.shape == (n_matrices, n_classes * n_clusters)


def clt_predict(clust, mats, n_clusters=None):
    n_matrices = len(mats)
    if n_clusters is None:
        clt = clust()
    elif clust is PotatoField:
        n_matrices = len(mats[0])
        clt = clust(n_potatoes=n_clusters)
    else:
        clt = clust(n_clusters=n_clusters)
    clt.fit(mats)
    pred = clt.predict(mats)
    assert pred.shape == (n_matrices,)


def clt_predict_proba(clust, mats, n=None):
    if n is None:
        n_matrices = len(mats)
        clt = clust()
    else:  # PotatoField
        n_matrices = len(mats[0])
        clt = clust(n_potatoes=n)
    clt.fit(mats)
    proba = clt.predict_proba(mats)
    assert proba.shape == (n_matrices,)


def clt_partial_fit(clust, mats, n=None):
    if n is None:
        clt = clust()
    else:  # PotatoField
        clt = clust(n_potatoes=n)
    clt.fit(mats)
    clt.partial_fit(mats)
    if n is None:
        clt.partial_fit(mats[np.newaxis, 0])  # fit one covmat at a time
    else:
        clt.partial_fit([m[np.newaxis, 0] for m in mats])


def clt_fit_independence(clust, mats, n=None):
    if n is None:
        clt = clust()
    else:  # PotatoField
        clt = clust(n_potatoes=n)
    clt.fit(mats).transform(mats)
    # retraining with different size should erase previous fit
    if n is None:
        new_mats = mats[:, :-1, :-1]
    else:
        new_mats = [m[:, :-1, :-1] for m in mats]
    clt.fit(new_mats).transform(new_mats)


def clt_fit_labels_independence(clust, mats, labels):
    clt = clust()
    clt.fit(mats, labels).transform(mats)
    # retraining with different size should erase previous fit
    new_mats = mats[:, :-1, :-1]
    clt.fit(new_mats, labels).transform(new_mats)


def clt_fittransform(clust, mats):
    clt = clust()
    transf = clt.fit_transform(mats)
    transf2 = clt.fit(mats).transform(mats)
    assert_array_equal(transf, transf2)


@pytest.mark.parametrize("clust", [Kmeans, KmeansPerClassTransform])
@pytest.mark.parametrize("init", ["random", "ndarray"])
@pytest.mark.parametrize("n_init", [1, 5])
@pytest.mark.parametrize("metric", get_metrics())
def test_kmeans(clust, init, n_init, metric, get_mats, get_labels):
    n_clusters, n_classes, n_matrices, n_channels = 2, 3, 9, 5
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)

    if init == "ndarray":
        clt = clust(
            n_clusters=n_clusters,
            metric=metric,
            init=mats[:n_clusters],
            n_init=n_init,
        )
    else:
        clt = clust(
            n_clusters=n_clusters,
            metric=metric,
            init=init,
            n_init=n_init
        )

    clt.fit(mats, labels)
    dists = clt.transform(mats)
    if clust is Kmeans:
        assert dists.shape == (n_matrices, n_clusters)
    elif clust is KmeansPerClassTransform:
        assert dists.shape == (n_matrices, clt.covmeans_.shape[0])


@np.vectorize
def callable_kernel(x):
    return np.exp(- np.abs(x))


@pytest.mark.parametrize("kernel", [
    "normal", "uniform", callable_kernel,
])
@pytest.mark.parametrize("metric", [
    "euclid", "logchol", "logeuclid", "riemann", "wasserstein"
])
def test_meanshift(kernel, metric, get_mats, get_labels):
    n_matrices, n_channels = 10, 3
    mats = get_mats(n_matrices, n_channels, "spd")

    clt = MeanShift(
        kernel=kernel,
        metric=metric,
    )
    clt.fit(mats)


@pytest.mark.parametrize("use_weight", [True, False])
def test_potato_fit(use_weight, get_mats, get_weights):
    n_matrices, n_channels = 6, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    y = np.ones(n_matrices)
    y[0] = 0
    if use_weight:
        weights = get_weights(n_matrices)
    else:
        weights = None
    Potato().fit(mats, y, sample_weight=weights)


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


def test_potato_partialfit_not_fitted(get_mats):
    n_matrices, n_channels = 6, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):  # potato not fitted
        Potato().partial_fit(mats)


def test_potato_partialfit_diff_channels(get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)
    pt = Potato().fit(mats, labels)
    with pytest.raises(ValueError):  # unequal # of chans
        pt.partial_fit(get_mats(2, n_channels + 1, "spd"))


def test_potato_partialfit_no_poslabel(get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)
    pt = Potato().fit(mats, labels)
    with pytest.raises(ValueError):  # no positive labels
        pt.partial_fit(mats, [0] * n_matrices)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_potato_partialfit_alpha(alpha, get_mats, get_labels):
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


def test_potatofield_fit(get_mats):
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


@pytest.mark.parametrize(
    "method", ["partial_fit", "transform", "predict_proba"]
)
def test_potatofield_method(get_mats, method):
    n_potatoes, n_matrices, n_channels = 2, 6, 3
    mats1 = get_mats(n_matrices, n_channels, "spd")
    mats2 = get_mats(n_matrices, n_channels + 1, "spd")
    mats = [mats1, mats2]
    pf = PotatoField(n_potatoes=n_potatoes).fit(mats)
    with pytest.raises(ValueError):  # n_potatoes not equal to input length
        getattr(pf, method)([mats1, mats1, mats2])
    with pytest.raises(ValueError):  # n_matrices not equal
        getattr(pf, method)([mats1, mats2[:1]])
