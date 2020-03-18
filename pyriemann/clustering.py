"""Clustering functions."""
import numpy
from sklearn.base import (BaseEstimator, ClassifierMixin, TransformerMixin,
                          ClusterMixin)
from sklearn.cluster._kmeans import _init_centroids
from joblib import Parallel, delayed

from .classification import MDM

#######################################################################


def _fit_single(X, y=None, n_clusters=2, init='random', random_state=None,
                metric='riemann', max_iter=100, tol=1e-4, n_jobs=1):
    """helper to fit a single run of centroid."""
    # init random state if provided
    mdm = MDM(metric=metric, n_jobs=n_jobs)
    squared_nomrs = [numpy.linalg.norm(x, ord='fro')**2 for x in X]
    mdm.covmeans_ = _init_centroids(X, n_clusters, init,
                                    random_state=random_state,
                                    x_squared_norms=squared_nomrs)
    if y is not None:
        mdm.classes_ = numpy.unique(y)
    else:
        mdm.classes_ = numpy.arange(n_clusters)

    labels = mdm.predict(X)
    k = 0
    while True:
        old_labels = labels.copy()
        mdm.fit(X, old_labels)
        dist = mdm._predict_distances(X)
        labels = mdm.classes_[dist.argmin(axis=1)]
        k += 1
        if (k > max_iter) | (numpy.mean(labels == old_labels) > (1 - tol)):
            break
    inertia = sum([sum(dist[labels == mdm.classes_[i], i])
                   for i in range(len(mdm.classes_))])
    return labels, inertia, mdm


class Kmeans(BaseEstimator, ClassifierMixin, ClusterMixin, TransformerMixin):

    """Kmean clustering using Riemannian geometry.

    Find clusters that minimize the sum of squared distance to their centroid.
    This is a direct implementation of the kmean algorithm with a riemanian
    metric.

    Parameters
    ----------
    n_cluster: int (default: 2)
        number of clusters.
    max_iter : int (default: 100)
        The maximum number of iteration to reach convergence.
    metric : string (default: 'riemann')
        The type of metric used for centroid and distance estimation.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    init : 'k-means++', 'random' or an ndarray (default 'random')
        Method for initialization of centers.
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
    n_init : int, (default: 10)
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    tol: float, (default: 1e-4)
        the stopping criterion to stop convergence, representing the minimum
        amount of change in labels between two iterations.

    Attributes
    ----------
    mdm_ : MDM instance.
        MDM instance containing the centroids.
    labels_ :
        Labels of each point
    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    Notes
    -----
    .. versionadded:: 0.2.2

    See Also
    --------
    Kmeans
    MDM
    """

    def __init__(self, n_clusters=2, max_iter=100, metric='riemann',
                 random_state=None, init='random', n_init=10, n_jobs=1,
                 tol=1e-4):
        """Init."""
        self.metric = metric
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = random_state
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit (estimates) the clusters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : Kmeans instance
            The Kmean instance.
        """
        if (self.init is not 'random') | (self.n_init == 1):
            # no need to iterate if init is not random
            labels, inertia, mdm = _fit_single(X, y,
                                               n_clusters=self.n_clusters,
                                               init=self.init,
                                               random_state=self.seed,
                                               metric=self.metric,
                                               max_iter=self.max_iter,
                                               tol=self.tol,
                                               n_jobs=self.n_jobs)
        else:
            numpy.random.seed(self.seed)
            seeds = numpy.random.randint(
                numpy.iinfo(numpy.int32).max, size=self.n_init)
            if self.n_jobs == 1:
                res = []
                for i in range(self.n_init):
                    res.append(_fit_single(X, y,
                                      n_clusters=self.n_clusters,
                                      init=self.init,
                                      random_state=seeds[i],
                                      metric=self.metric,
                                      max_iter=self.max_iter,
                                      tol=self.tol))
                labels, inertia, mdm = zip(*res)
            else:

                res = Parallel(n_jobs=self.n_jobs, verbose=0)(
                    delayed(_fit_single)(X, y,
                                         n_clusters=self.n_clusters,
                                         init=self.init,
                                         random_state=seed,
                                         metric=self.metric,
                                         max_iter=self.max_iter,
                                         tol=self.tol,
                                         n_jobs=1)
                    for seed in seeds)
                labels, inertia, mdm = zip(*res)

            best = numpy.argmin(inertia)
            mdm = mdm[best]
            labels = labels[best]
            inertia = inertia[best]

        self.mdm_ = mdm
        self.inertia_ = inertia
        self.labels_ = labels

        return self

    def predict(self, X):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        return self.mdm_.predict(X)

    def transform(self, X):
        """get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_trials, n_cluster)
            the distance to each centroid according to the metric.
        """
        return self.mdm_.transform(X)

    def centroids(self):
        """helper for fast access to the centroid.

        Returns
        -------
        centroids : list of SPD matrices, len (n_cluster)
            Return a list containing the centroid of each cluster.
        """
        return self.mdm_.covmeans_


class KmeansPerClassTransform(BaseEstimator, TransformerMixin):

    """Run kmeans for each class."""

    def __init__(self, n_clusters=2, **params):
        """Init."""
        params['n_clusters'] = n_clusters
        self.km = Kmeans(**params)
        self.metric = self.km.metric

    def fit(self, X, y):
        """fit."""
        self.covmeans_ = []
        self.classes_ = numpy.unique(y)
        for c in self.classes_:
            self.km.fit(X[y == c])
            self.covmeans_.extend(self.km.centroids())
        return self

    def transform(self, X):
        """transform."""
        mdm = MDM(metric=self.metric, n_jobs=self.km.n_jobs)
        mdm.covmeans_ = self.covmeans_
        return mdm._predict_distances(X)


class Potato(BaseEstimator, TransformerMixin, ClassifierMixin):

    """Artefact detection with the Riemannian Potato.

    The Riemannian Potato [1] is a clustering method used to detect artifact in
    EEG signals. The algorithm iteratively estimate the centroid of clean
    signal by rejecting every trial that have a distance greater than several
    standard deviation from it.

    Parameters
    ----------
    metric : string (default 'riemann')
        The type of metric used for centroid and distance estimation.
    threshold : int (default 3)
        The number of standard deviation to reject artifacts.
    n_iter_max : int (default 100)
        The maximum number of iteration to reach convergence.
    pos_label: int (default 1)
        The positive label corresponding to clean data
    neg_label: int (default 0)
        The negative label corresponding to artifact data

    Notes
    -----
    .. versionadded:: 0.2.3

    See Also
    --------
    Kmeans
    MDM

    References
    ----------
    [1] A. Barachant, A. Andreev and M. Congedo, "The Riemannian Potato: an
    automatic and adaptive artifact detection method for online experiments
    using Riemannian geometry", in Proceedings of TOBI Workshop IV, p. 19-20,
    2013.
    """

    def __init__(self, metric='riemann', threshold=3, n_iter_max=100,
                 pos_label=1, neg_label=0):
        """Init."""
        self.metric = metric
        self.threshold = threshold
        self.n_iter_max = n_iter_max
        if pos_label == neg_label:
            raise(ValueError("Positive and Negative labels must be different"))
        self.pos_label = pos_label
        self.neg_label = neg_label

    def fit(self, X, y=None):
        """Fit the potato from covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : Potato instance
            The Potato instance.
        """
        self._mdm = MDM(metric=self.metric)

        if y is not None:
            if len(y) != len(X):
                raise ValueError('y must be the same lenght of X')

            classes = numpy.int32(numpy.unique(y))

            if len(classes) > 2:
                raise ValueError('number of classes must be maximum 2')

            if self.pos_label not in classes:
                raise ValueError('y must contain a positive class')

            y_old = numpy.int32(numpy.array(y) == self.pos_label)
        else:
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
        """return the normalized log-distance to the centroid (z-score).

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        z : ndarray, shape (n_epochs, 1)
            the normalized log-distance to the centroid.
        """
        d = numpy.squeeze(numpy.log(self._mdm.transform(X)))
        z = self._get_z_score(d)
        return z

    def predict(self, X):
        """predict artefact from data.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of bool, shape (n_epochs, 1)
            the artefact detection. True if the trial is clean, and False if
            the trial contain an artefact.
        """
        z = self.transform(X)
        pred = z < self.threshold
        out = numpy.zeros_like(z) + self.neg_label
        out[pred] = self.pos_label
        return out

    def _get_z_score(self, d):
        """get z score from distance."""
        z = (d - self._mean) / self._std
        return z
