"""Clustering functions."""
import numpy
from sklearn.base import (BaseEstimator, ClassifierMixin, TransformerMixin,
                          ClusterMixin)
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed

from .classification import MDM

#######################################################################


def _fit_single(
        X,
        y=None,
        n_clusters=2,
        init='random',
        random_state=None,
        metric='riemann',
        max_iter=100,
        tol=1e-4):
    # init random state if provided
    mdm = MDM(metric=metric)
    mdm.covmeans = _init_centroids(
        X, n_clusters, init, random_state=random_state)
    if y is not None:
        mdm.classes = numpy.unique(y)
    else:
        mdm.classes = numpy.arange(n_clusters)

    labels = mdm.predict(X)
    k = 0
    while True:
        old_labels = labels.copy()
        mdm.fit(X, old_labels)
        dist = mdm._predict_distances(X)
        labels = mdm.classes[dist.argmin(axis=1)]
        k += 1
        if (k > max_iter) | (numpy.mean(labels == old_labels) > (1 - tol)):
            break
    inertia = sum([sum(dist[labels == mdm.classes[i], i])
                   for i in range(len(mdm.classes))])
    return labels, inertia, mdm


class Kmeans(BaseEstimator, ClassifierMixin, ClusterMixin, TransformerMixin):

    """Kmeans unsupervised clustering."""

    def __init__(self, n_clusters=2, max_iter=100, metric='riemann',
                 random_state=None, init='random', n_init=10, n_jobs=1,
                 tol=1e-4):
        """Init."""
        self.metric = metric
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.mdm = None
        self.seed = random_state
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit."""
        if (self.init is not 'random') | (self.n_init == 1):
            # no need to iterate if init is not random
            labels, inertia, mdm = _fit_single(X, y,
                                               n_clusters=self.n_clusters,
                                               init=self.init,
                                               random_state=self.seed,
                                               metric=self.metric,
                                               max_iter=self.max_iter,
                                               tol=self.tol)
        else:
            numpy.random.seed(self.seed)
            seeds = numpy.random.randint(
                numpy.iinfo(numpy.int32).max, size=self.n_init)
            if self.n_jobs == 1:
                res = []
                for i in range(self.n_init):
                    res = _fit_single(X, y,
                                      n_clusters=self.n_clusters,
                                      init=self.init,
                                      random_state=seeds[i],
                                      metric=self.metric,
                                      max_iter=self.max_iter,
                                      tol=self.tol)
                labels, inertia, mdm = zip(res)
            else:

                res = Parallel(n_jobs=self.n_jobs, verbose=0)(
                    delayed(_fit_single)(X, y,
                                         n_clusters=self.n_clusters,
                                         init=self.init,
                                         random_state=seed,
                                         metric=self.metric,
                                         max_iter=self.max_iter,
                                         tol=self.tol)
                    for seed in seeds)
                labels, inertia, mdm = zip(*res)

            best = numpy.argmin(inertia)
            mdm = mdm[best]
            labels = labels[best]
            inertial = inertia[best]

        self.mdm = mdm
        self.inertia = inertial
        self.labels_ = labels

        return self

    def predict(self, X):
        """predict."""
        return self.mdm.predict(X)

    def transform(self, X):
        """transform."""
        return self.mdm.transform(X)

    def covmeans(self):
        """return centroid."""
        return self.mdm.covmeans


class KmeansPerClassTransform(BaseEstimator, TransformerMixin):

    """Run kmeans for each class."""

    def __init__(self, n_clusters=2, **params):
        """Init."""
        params['n_clusters'] = n_clusters
        self.km = Kmeans(**params)
        self.metric = self.km.metric
        self.covmeans = []

    def fit(self, X, y):
        """fit."""
        self.classes = numpy.unique(y)
        for c in self.classes:
            self.km.fit(X[y == c])
            self.covmeans.extend(self.km.covmeans())
        return self

    def transform(self, X):
        """transform."""
        mdm = MDM(metric=self.metric)
        mdm.covmeans = self.covmeans
        return mdm._predict_distances(X)


class Potato(BaseEstimator, TransformerMixin, ClassifierMixin):

    """Artefact detection with Riemannian potato."""

    def __init__(self, metric='riemann', threshold=3, n_iter_max=100):
        """Init."""
        self.metric = metric
        self.threshold = threshold
        self.n_iter_max = n_iter_max

    def fit(self, X, y=None):
        """fit."""
        self._mdm = MDM(metric=self.metric)

        if y is None:
            y_old = numpy.ones(len(X))
        # start loop
        for n_iter in range(self.n_iter_max):
            ix = (y_old == 1)
            self._mdm.fit(X[ix], y_old[ix])
            y = numpy.zeros(len(X))
            d = numpy.squeeze(numpy.log(self._mdm.transform(X[ix])))
            self._mean = numpy.mean(d)
            self._std = numpy.std(d)
            y[ix] = self._get_z_score(d) < self.threshold

            if numpy.array_equal(y, y_old):
                break
            else:
                y_old = y
        return self

    def transform(self, X):
        """transform."""
        d = numpy.squeeze(numpy.log(self._mdm.transform(X)))
        z = self._get_z_score(d)
        return z

    def predict(self, X):
        """predict artefacts."""
        z = self.transform(X)
        pred = z < self.threshold
        return pred

    def _get_z_score(self, d):
        """get z score from distance."""
        z = (d - self._mean) / self._std
        return z
