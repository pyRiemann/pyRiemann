import numpy as np
from numpy.testing import assert_array_equal
import pytest
from pytest import approx

from pyriemann.clustering import (
    Kmeans,
    KmeansPerClassTransform,
    MeanShift,
    Gaussian,
    GaussianMixture
)
from pyriemann.geometry.tangentspace import tangent_space


pytestmark = pytest.mark.numpy_only


clusts = [
    Kmeans,
    KmeansPerClassTransform,
    MeanShift,
    GaussianMixture
]


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("clust", clusts)
def test_clustering_two_clusters(kind, clust, get_mats, get_labels):
    if kind == "hpd" and clust in [GaussianMixture]:
        pytest.skip()

    n_clusters, n_matrices, n_channels = 2, 40, 3
    X = get_mats(n_matrices, n_channels, kind)

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
        clt_score(clust, X)


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
        clt_score(clust, X)


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
        assert clt.weights_.shape == (clt.n_components,)
        assert clt.weights_.sum() == approx(1)
        assert clt.means_.shape == (clt.n_components, n_channels, n_channels)
        n_ts = n_channels * (n_channels + 1) // 2
        assert clt.covariances_.shape == (clt.n_components, n_ts, n_ts)
        return


def clt_fit_weights(clust, X, weights):
    clt = clust()
    clt.fit(X, sample_weight=weights)


def clt_transform(clust, X, n_clusters=None):
    n_matrices = len(X)
    if n_clusters is None:
        clt = clust()
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


def clt_score(clust, X, y=None):
    clt = clust()
    score = clt.fit(X, y).score(X, y)
    assert isinstance(score, float)


###############################################################################


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


def test_gaussian(get_mats, get_weights):
    n_matrices, n = 13, 3
    X = get_mats(n_matrices, n, "spd")
    weights = get_weights(n_matrices)

    gm = Gaussian(n=n, mu=X[0], metric="riemann")

    gm.update_mean(X, weights)
    assert gm.mu.shape == (n, n)

    gm.update_covariance(X, weights)
    n_ts = n * (n + 1) // 2
    assert gm.sigma.shape == (n_ts, n_ts)

    pdf = gm.pdf(X)
    assert pdf.shape == (n_matrices,)

    tv = tangent_space(X, gm.mu, metric="riemann")[0]
    dist = tv.T @ np.linalg.solve(gm.sigma, tv)
    num = np.exp(-0.5 * dist)
    denom = np.sqrt(((2 * np.pi) ** n) * np.linalg.det(gm.sigma))
    pdf_ = num / (denom + 1e-16)
    assert pdf[0] == approx(pdf_)


@pytest.mark.parametrize("n_components", [2, 4])
def test_gmm(n_components, get_mats, get_weights):
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

    n_sampled_matrices = 20
    X, y = gmm.sample(n_sampled_matrices)
    assert X.shape == (n_sampled_matrices, n_channels, n_channels)
    assert y.shape == (n_sampled_matrices,)
