import numpy as np
from numpy.testing import assert_array_equal
import pytest
from pytest import approx

from pyriemann.clustering import (
    Kmeans,
    KmeansPerClassTransform,
    MeanShift,
    GaussianMixture,
    Potato,
    PotatoField,
)

clusts = [
    Kmeans,
    KmeansPerClassTransform,
    MeanShift,
    GaussianMixture,
    Potato,
    PotatoField,
]


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("clust", clusts)
def test_clustering_two_clusters(kind, clust,
                                 get_mats, get_labels, get_weights):
    if kind == "hpd" and clust in [GaussianMixture]:
        pytest.skip()

    n_clusters, n_matrices, n_channels = 2, 40, 3
    X = get_mats(n_matrices, n_channels, kind)
    weights = get_weights(n_matrices)

    if clust is Kmeans:
        clt_fit(clust, X, n_clusters, None)
        clt_predict(clust, X, n_clusters)
        clt_fitpredict(clust, X, n_clusters)
        clt_transform(clust, X, n_clusters)
        clt_fittransform(clust, X)
        clt_jobs(clust, X, n_clusters)
        clt_centroids(clust, X, n_clusters)
        clt_fit_independence(clust, X)

    if clust is KmeansPerClassTransform:
        n_classes = n_clusters
        y = get_labels(n_matrices, n_classes)
        clt_fit(clust, X, n_clusters, y)
        clt_transform_per_class(clust, X, n_clusters, y)
        clt_fittransform_per_class(clust, X, n_clusters, y)
        clt_jobs(clust, X, n_clusters, y)
        clt_fit_labels_independence(clust, X, y)

    if clust is MeanShift:
        clt_fit(clust, X, n_clusters, None)
        clt_predict(clust, X)
        clt_fitpredict(clust, X)

    if clust is GaussianMixture:
        clt_fit(clust, X, n_clusters, None)
        clt_predict(clust, X)
        clt_fitpredict(clust, X)
        clt_predict_proba(clust, X)

    if clust is Potato:
        clt_fit(clust, X, n_clusters, None)
        clt_fit_weights(clust, X, weights)
        clt_transform(clust, X)
        clt_predict(clust, X)
        clt_predict_proba(clust, X)
        clt_partial_fit(clust, X)
        clt_fit_independence(clust, X)
        clt_fittransform(clust, X)

    if clust is PotatoField:
        n_potatoes = 3
        X = [
            get_mats(n_matrices, n_channels, kind),
            get_mats(n_matrices, n_channels + 2, kind),
            get_mats(n_matrices, n_channels + 1, kind),
        ]
        clt_fit_weights(clust, X, weights)
        clt_transform(clust, X, n_potatoes)
        clt_predict(clust, X, n_potatoes)
        clt_predict_proba(clust, X, n_potatoes)
        clt_partial_fit(clust, X, n_potatoes)
        clt_fit_independence(clust, X, n_potatoes)
        clt_fittransform(clust, X, n_potatoes)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("clust", clusts)
def test_clustering_three_clusters(kind, clust, get_mats, get_labels):
    if kind == "hpd" and clust in [GaussianMixture]:
        pytest.skip()

    n_clusters, n_matrices, n_channels = 3, 30, 2
    X = get_mats(n_matrices, n_channels, kind)

    if clust is Kmeans:
        clt_fit(clust, X, n_clusters, None)
        clt_predict(clust, X, n_clusters)
        clt_transform(clust, X, n_clusters)
        clt_jobs(clust, X, n_clusters)
        clt_centroids(clust, X, n_clusters)
        clt_fit_independence(clust, X)

    if clust is KmeansPerClassTransform:
        n_classes = 2
        y = get_labels(n_matrices, n_classes)
        clt_transform_per_class(clust, X, n_clusters, y)
        clt_jobs(clust, X, n_clusters, y)
        clt_fit_labels_independence(clust, X, y)

    if clust is MeanShift:
        clt_fit(clust, X, n_clusters, None)
        clt_predict(clust, X)

    if clust is GaussianMixture:
        clt_fit(clust, X, n_clusters, None)
        clt_predict(clust, X)
        clt_fitpredict(clust, X)
        clt_predict_proba(clust, X)


def clt_fit(clust, X, n_clusters, labels):
    n_matrices, n_channels, _ = X.shape
    n_classes = len(np.unique(labels))

    clt = clust()
    clt.fit(X, labels)

    if clust is Kmeans:
        assert clt.labels_.shape == (n_matrices,)
        assert len(clt.mdm_.covmeans_) <= n_clusters
        assert clt.mdm_.covmeans_.shape[1:] == (n_channels, n_channels)
        assert_array_equal(clt.mdm_.covmeans_, clt.centroids())
        return
    if clust is KmeansPerClassTransform:
        assert clt.classes_.shape == (n_classes,)
        assert len(clt.covmeans_) <= n_clusters * n_classes
        assert clt.covmeans_.shape[1:] == (n_channels, n_channels)
        return
    if clust is MeanShift:
        assert clt.labels_.shape == (n_matrices,)
        assert len(clt.modes_) >= 1
        assert clt.modes_.shape[1:] == (n_channels, n_channels)
        return
    if clust is GaussianMixture:
        assert len(clt.components_) == clt.n_components
        assert clt.weights_.shape == (clt.n_components,)
        assert clt.weights_.sum() == approx(1)
        return
    if clust is Potato:
        assert clt.covmean_.shape == (n_channels, n_channels)


def clt_fit_weights(clust, X, weights):
    if clust is PotatoField:
        clt = clust(n_potatoes=len(X))
    else:
        clt = clust()
    clt.fit(X, sample_weight=weights)


def clt_transform(clust, X, n_clusters=None):
    n_matrices = len(X)
    if n_clusters is None:
        clt = clust()
    elif clust is PotatoField:
        n_matrices = len(X[0])
        clt = clust(n_potatoes=n_clusters)
    else:
        clt = clust(n_clusters=n_clusters)
    transf = clt.fit(X).transform(X)

    if n_clusters is None:
        assert transf.shape == (n_matrices,)
    else:
        assert transf.shape == (n_matrices, n_clusters)


def clt_jobs(clust, X, n_clusters, labels=None):
    n_matrices = X.shape[0]
    clt = clust(n_clusters=n_clusters, n_jobs=2)
    if labels is None:
        clt.fit(X)
    else:
        clt.fit(X, labels)

    if clust in [Kmeans, KmeansPerClassTransform]:
        transf = clt.transform(X)
        assert len(transf) == (n_matrices)


def clt_centroids(clust, X, n_clusters):
    n_channels = X.shape[-1]
    clt = clust(n_clusters=n_clusters).fit(X)
    centroids = clt.centroids()
    assert centroids.shape == (n_clusters, n_channels, n_channels)


def clt_transform_per_class(clust, X, n_clusters, y):
    n_classes, n_matrices = len(np.unique(y)), X.shape[0]
    clt = clust(n_clusters=n_clusters)
    transf = clt.fit(X, y).transform(X)
    assert transf.shape == (n_matrices, n_classes * n_clusters)


def clt_predict(clust, X, n_clusters=None):
    n_matrices = len(X)
    if n_clusters is None:
        clt = clust()
    elif clust is PotatoField:
        n_matrices = len(X[0])
        clt = clust(n_potatoes=n_clusters)
    else:
        clt = clust(n_clusters=n_clusters)
    pred = clt.fit(X).predict(X)
    assert pred.shape == (n_matrices,)


def clt_fitpredict(clust, X, n_clusters=None):
    if n_clusters is None:
        clt = clust()
    else:
        clt = clust(n_clusters=n_clusters)
    if hasattr(clt, "random_state"):
        clt.set_params(**{"random_state": 42})
    pred = clt.fit(X).predict(X)
    if hasattr(clt, "random_state"):
        clt.set_params(**{"random_state": 42})
    pred2 = clt.fit_predict(X)
    assert_array_equal(pred, pred2)


def clt_predict_proba(clust, X, n=None):
    if n is None:
        n_matrices = len(X)
        clt = clust()
    else:  # PotatoField
        n_matrices = len(X[0])
        clt = clust(n_potatoes=n)
    clt.fit(X)
    prob = clt.predict_proba(X)
    assert prob.shape[0] == n_matrices
    if prob.ndim > 1:
        assert prob.sum(axis=-1) == approx(np.ones(n_matrices))


def clt_partial_fit(clust, X, n=None):
    if n is None:
        clt = clust()
    else:  # PotatoField
        clt = clust(n_potatoes=n)
    clt.fit(X)
    clt.partial_fit(X)
    if n is None:
        clt.partial_fit(X[np.newaxis, 0])  # fit one covmat at a time
    else:
        clt.partial_fit([x[np.newaxis, 0] for x in X])


def clt_fit_independence(clust, X, n=None):
    if n is None:
        clt = clust()
    else:  # PotatoField
        clt = clust(n_potatoes=n)
    clt.fit(X).transform(X)
    # retraining with different size should erase previous fit
    if n is None:
        Xnew = X[:, :-1, :-1]
    else:
        Xnew = [x[:, :-1, :-1] for x in X]
    clt.fit(Xnew).transform(Xnew)


def clt_fit_labels_independence(clust, X, labels):
    clt = clust()
    clt.fit(X, labels).transform(X)
    # retraining with different size should erase previous fit
    Xnew = X[:, :-1, :-1]
    clt.fit(Xnew, labels).transform(Xnew)


def clt_fittransform(clust, X, n_clusters=None):
    if n_clusters is None:
        clt = clust()
    elif clust is PotatoField:
        clt = clust(n_potatoes=n_clusters)
    if hasattr(clt, "random_state"):
        clt.set_params(**{"random_state": 42})
    Xt = clt.fit(X).transform(X)
    Xt2 = clt.fit_transform(X)
    assert_array_equal(Xt, Xt2)


def clt_fittransform_per_class(clust, X, n_clusters, y):
    clt = clust(n_clusters=n_clusters, random_state=42)
    Xt = clt.fit(X, y).transform(X)
    Xt2 = clt.fit_transform(X, y)
    assert_array_equal(Xt, Xt2)


@pytest.mark.parametrize("clust", [Kmeans, KmeansPerClassTransform])
@pytest.mark.parametrize("init", ["random", "ndarray"])
@pytest.mark.parametrize("n_init", [1, 5])
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_kmeans(clust, init, n_init, metric, get_mats, get_labels):
    n_clusters, n_classes, n_matrices, n_channels = 2, 3, 9, 5
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)

    if init == "ndarray":
        clt = clust(
            n_clusters=n_clusters,
            metric=metric,
            init=X[:n_clusters],
            n_init=n_init,
        )
    else:
        clt = clust(
            n_clusters=n_clusters,
            metric=metric,
            init=init,
            n_init=n_init
        )

    clt.fit(X, y)
    dists = clt.transform(X)
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
    X = get_mats(n_matrices, n_channels, "spd")

    clt = MeanShift(
        kernel=kernel,
        metric=metric,
    )
    clt.fit(X)


@pytest.mark.parametrize("n_components", [2, 4])
def test_gmm_init(n_components, get_mats, get_weights):
    n_matrices, n_channels = 50, 2
    X = get_mats(n_matrices, n_channels, "spd")
    means_init = get_mats(n_components, n_channels, "spd")
    weights_init = get_weights(n_components)

    gmm = GaussianMixture(
        n_components=n_components,
        means_init=means_init,
        weights_init=weights_init,
    )
    gmm.fit(X)


@pytest.mark.parametrize("use_weight", [True, False])
def test_potato_fit(use_weight, get_mats, get_weights):
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = np.ones(n_matrices)
    y[0] = 0
    if use_weight:
        weights = get_weights(n_matrices)
    else:
        weights = None
    Potato().fit(X, y, sample_weight=weights)


def test_potato_fit_equal_labels(get_mats):
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):
        Potato(pos_label=0).fit(X)


@pytest.mark.parametrize("y_fail", [[1], [0] * 6, [0] * 7, [0, 1, 2] * 2])
def test_potato_fit_error(y_fail, get_mats):
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):
        Potato().fit(X, y=y_fail)


def test_potato_partialfit_not_fitted(get_mats):
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):  # potato not fitted
        Potato().partial_fit(X)


def test_potato_partialfit_diff_channels(get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)
    pt = Potato().fit(X, y)
    with pytest.raises(ValueError):  # unequal # of chans
        pt.partial_fit(get_mats(2, n_channels + 1, "spd"))


def test_potato_partialfit_no_poslabel(get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)
    pt = Potato().fit(X, y)
    with pytest.raises(ValueError):  # no positive labels
        pt.partial_fit(X, [0] * n_matrices)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_potato_partialfit_alpha(alpha, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)
    pt = Potato().fit(X, y)
    with pytest.raises(ValueError):
        pt.partial_fit(X, y, alpha=alpha)


def test_potato_1channel(get_mats):
    n_matrices, n_channels = 6, 1
    X_1chan = get_mats(n_matrices, n_channels, "spd")
    pt = Potato()
    pt.fit_transform(X_1chan)
    pt.predict(X_1chan)
    pt.predict_proba(X_1chan)


def test_potato_threshold(get_mats):
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    pt = Potato(threshold=2.5)
    pt.fit(X)


def test_potato_specific_labels(get_mats):
    n_matrices, n_channels = 10, 3
    X = get_mats(n_matrices, n_channels, "spd")
    X[-1] = 10 * np.eye(n_channels)
    pt = Potato(threshold=2.0, pos_label=2, neg_label=7)
    pt.fit(X)
    assert_array_equal(np.unique(pt.predict(X)), [2, 7])
    # fit with custom positive label
    pt.fit(X, y=[2] * n_matrices)


def test_potatofield_fit(get_mats):
    n_potatoes, n_matrices, n_channels = 2, 6, 3
    X1 = get_mats(n_matrices, n_channels, "spd")
    X2 = get_mats(n_matrices, n_channels + 1, "spd")
    X = [X1, X2]
    with pytest.raises(ValueError):  # n_potatoes too low
        PotatoField(n_potatoes=0).fit(X)
    with pytest.raises(ValueError):   # p_threshold out of bounds
        PotatoField(p_threshold=0).fit(X)
    with pytest.raises(ValueError):  # p_threshold out of bounds
        PotatoField(p_threshold=1).fit(X)
    pf = PotatoField(n_potatoes=n_potatoes)
    with pytest.raises(ValueError):  # n_potatoes not equal to input length
        pf.fit([X1, X1, X2])
    with pytest.raises(ValueError):  # n_matrices not equal
        pf.fit([X1, X2[:1]])


@pytest.mark.parametrize(
    "method", ["partial_fit", "transform", "predict_proba"]
)
def test_potatofield_method(get_mats, method):
    n_potatoes, n_matrices, n_channels = 2, 6, 3
    X1 = get_mats(n_matrices, n_channels, "spd")
    X2 = get_mats(n_matrices, n_channels + 1, "spd")
    X = [X1, X2]
    pf = PotatoField(n_potatoes=n_potatoes).fit(X)
    with pytest.raises(ValueError):  # n_potatoes not equal to input length
        getattr(pf, method)([X1, X1, X2])
    with pytest.raises(ValueError):  # n_matrices not equal
        getattr(pf, method)([X1, X2[:1]])
