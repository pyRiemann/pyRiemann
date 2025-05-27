"""Clustering functions."""
from math import floor
import warnings

from joblib import Parallel, delayed
import numpy as np
from scipy.stats import norm, chi2
import sklearn
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClusterMixin,
    clone,
)
from sklearn.cluster import KMeans as _KMeans

from .classification import MDM, SpdClassifMixin
from .utils.distance import distance, pairwise_distance
from .utils.mean import mean_covariance
from .utils.geodesic import geodesic
from .utils.tangentspace import exp_map, log_map
from .utils.utils import check_metric, check_function


class SpdClustMixin(ClusterMixin):
    """ClusterMixin for SPD matrices"""

    def fit_predict(self, X, y=None):
        """Fit and predict in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Prediction for each matrix according to the closest cluster.
        """
        return self.fit(X, y).predict(X)


def _init_centroids(X, n_clusters, init, random_state, x_squared_norms):
    if random_state is not None:
        random_state = np.random.RandomState(random_state)
    if sklearn.__version__ < "1.3.0":
        return _KMeans(n_clusters=n_clusters, init=init)._init_centroids(
            X,
            x_squared_norms,
            init,
            random_state,
        )
    else:
        n_matrices = X.shape[0]
        return _KMeans(n_clusters=n_clusters, init=init)._init_centroids(
            X,
            x_squared_norms,
            init,
            random_state,
            sample_weight=np.ones(n_matrices) / n_matrices,
        )


def _fit_single(X, y=None, n_clusters=2, init="random", random_state=None,
                metric="riemann", max_iter=100, tol=1e-4, n_jobs=1):
    """helper to fit a single run of centroid."""
    # init random state if provided
    mdm = MDM(metric=metric, n_jobs=n_jobs)
    mdm.metric_mean, mdm.metric_dist = check_metric(metric)
    squared_norms = np.linalg.norm(X, ord="fro", axis=(1, 2))**2
    mdm.covmeans_ = _init_centroids(
        X,
        n_clusters,
        init,
        random_state=random_state,
        x_squared_norms=squared_norms,
    )
    mdm.classes_ = np.arange(n_clusters)

    labels = mdm.predict(X)
    k = 0
    while True:
        old_labels = labels.copy()
        mdm.fit(X, old_labels)
        dist = mdm._predict_distances(X)
        labels = mdm.classes_[dist.argmin(axis=1)]
        k += 1
        if (k > max_iter) | (np.mean(labels == old_labels) > (1 - tol)):
            break
    inertia = sum([
        sum(dist[labels == mdm.classes_[i], i])
        for i in range(len(mdm.classes_))
    ])
    return labels, inertia, mdm


class Kmeans(SpdClassifMixin, SpdClustMixin, TransformerMixin, BaseEstimator):
    """Clustering by k-means with SPD/HPD matrices as inputs.

    The k-means is a clustering method used to find clusters that minimize the
    sum of squared distances between centroids and SPD/HPD matrices [1]_.

    Then, for each new matrix, the class is affected according to the nearest
    centroid.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters.
    max_iter : int, default=100
        Maximum number of iteration to reach convergence.
    metric : string | dict, default="riemann"
        Metric used for mean estimation (for the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`) and
        for distance estimation
        (see :func:`pyriemann.utils.distance.distance`).
        The metric can be a dict with two keys, "mean" and "distance"
        in order to pass different metrics.
    random_state : None | integer | np.RandomState, default=None
        The generator used to initialize the centroids. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    init : "random" | ndarray, shape (n_clusters, n_channels, n_channels), \
            default="random"
        Method for initialization of centroids.
        If "random", it chooses k matrices at random for the initial centroids.
        If an ndarray is passed, it should be of shape
        (n_clusters, n_channels, n_channels) and gives the initial centroids.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    n_jobs : int, default=1
        Number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    tol : float, default=1e-4
        Stopping criterion to stop convergence, representing the minimum
        amount of change in labels between two iterations.

    Attributes
    ----------
    mdm_ : MDM instance
        MDM instance containing the centroids.
    labels_ : ndarray, shape (n_matrices,)
        Labels, ie centroid indices, of each matrix of training set.
    inertia_ : float
        Sum of distances of matrices to their closest cluster centroids.

    Notes
    -----
    .. versionadded:: 0.2.2

    See Also
    --------
    Kmeans
    MDM

    References
    ----------
    .. [1] `Commande robuste d'un effecteur par une interface cerveau machine
        EEG asynchrone
        <https://theses.hal.science/tel-01196752/>`_
        A. Barachant, Thesis, 2012
    """

    def __init__(
        self,
        n_clusters=2,
        max_iter=100,
        metric="riemann",
        random_state=None,
        init="random",
        n_init=10,
        n_jobs=1,
        tol=1e-4,
    ):
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
        """Fit the centroids of clusters.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : Kmeans instance
            The Kmeans instance.
        """
        if isinstance(self.init, str) and self.init == "random":
            np.random.seed(self.seed)
            seeds = np.random.randint(
                np.iinfo(np.int32).max,
                size=self.n_init,
            )

            res = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_single)(
                    X,
                    y,
                    n_clusters=self.n_clusters,
                    init=self.init,
                    random_state=seed,
                    metric=self.metric,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    n_jobs=1,
                ) for seed in seeds
            )
            labels, inertia, mdm = zip(*res)

            best = np.argmin(inertia)
            mdm = mdm[best]
            labels = labels[best]
            inertia = inertia[best]

        else:
            # no need to iterate if init is not random
            labels, inertia, mdm = _fit_single(
                X,
                y,
                n_clusters=self.n_clusters,
                init=self.init,
                random_state=self.seed,
                metric=self.metric,
                max_iter=self.max_iter,
                tol=self.tol,
                n_jobs=self.n_jobs,
            )

        self.mdm_ = mdm
        self.inertia_ = inertia
        self.labels_ = labels

        return self

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Prediction for each matrix according to the closest centroid.
        """
        return self.mdm_.predict(X)

    def transform(self, X):
        """Get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_clusters)
            Distance to each centroid according to the metric.
        """
        return self.mdm_.transform(X)

    def fit_transform(self, X, y=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_clusters)
            Distance to each centroid according to the metric.
        """
        return self.fit(X, y).transform(X)

    def centroids(self):
        """Helper for fast access to the centroids.

        Returns
        -------
        centroids : ndarray, shape (n_clusters, n_channels, n_channels)
            Centroids of each cluster.
        """
        return self.mdm_.covmeans_


class KmeansPerClassTransform(TransformerMixin, BaseEstimator):
    """Clustering by k-means for each class with SPD/HPD matrices as inputs.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters.
    **params : dict
        The keyword arguments passed to :class:`pyriemann.clustering.Kmeans`.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.
    covmeans_ : ndarray, shape (n_centroids, n_channels, n_channels)
        Centroids of each cluster of each class, with n_centroids <=
        n_clusters x n_classes.

    See Also
    --------
    Kmeans
    """

    def __init__(self, n_clusters=2, **params):
        """Init."""
        params["n_clusters"] = n_clusters
        self._km = Kmeans(**params)
        self.metric = self._km.metric

    def fit(self, X, y):
        """Fit the clusters for each class.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels corresponding to each matrix.

        Returns
        -------
        self : KmeansPerClassTransform instance
            The KmeansPerClassTransform instance.
        """
        self.classes_ = np.unique(y)

        covmeans = []
        for c in self.classes_:
            self._km.fit(X[y == c])
            covmeans.extend(self._km.centroids())
        self.covmeans_ = np.stack(covmeans, axis=0)
        return self

    def transform(self, X):
        """Get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_centroids)
            Distance to each centroid according to the metric.
        """
        mdm = MDM(metric=self.metric, n_jobs=self._km.n_jobs)
        mdm.metric_mean, mdm.metric_dist = check_metric(self.metric)
        mdm.covmeans_ = self.covmeans_
        return mdm._predict_distances(X)

    def fit_transform(self, X, y):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels corresponding to each matrix.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_centroids)
            Distance to each centroid according to the metric.
        """
        return self.fit(X, y).transform(X)


@np.vectorize
def kernel_normal(x):
    return np.exp(- x ** 2)


@np.vectorize
def kernel_uniform(x):
    if np.abs(x) <= 1:
        return 1
    return 0


ker_clust_functions = {
    "normal": kernel_normal,
    "uniform": kernel_uniform,
}


class MeanShift(SpdClustMixin, BaseEstimator):
    """Clustering by mean shift with SPD/HPD matrices as inputs.

    The mean shift is a non-parametric clustering method used to find clusters
    on the manifold of SPD/HPD matrices, estimating the gradient of matrices
    density [1]_.

    Parameters
    ----------
    kernel : {"normal", "uniform"} | callable, default="uniform"
        Kernel used for kernel density estimation.
    bandwidth : None | float, default=None
        Bandwidth of the kernel.
    metric : string | dict, default="riemann"
        Metric used for map estimation (for the list of supported metrics,
        see :func:`pyriemann.utils.tangentspace.log_map`) and
        for distance estimation
        (see :func:`pyriemann.utils.distance.distance`).
        The metric can be a dict with two keys, "map" and "distance"
        in order to pass different metrics.
    tol : float, default=1e-4
        Stopping criterion to stop convergence, representing the norm of
        gradient.
    max_iter : int, default=100
        Maximum number of iteration to reach convergence.
    n_jobs : int, default=1
        Number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    modes_ : ndarray, shape (n_modes, n_channels, n_channels)
        Modes of each cluster.
    labels_ : ndarray, shape (n_matrices,)
        Labels, ie mode indices, of each matrix of training set.

    Notes
    -----
    .. versionadded:: 0.9

    See Also
    --------
    Kmeans

    References
    ----------
    .. [1] `Nonlinear Mean Shift over Riemannian Manifolds
        <https://sites.rutgers.edu/peter-meer/wp-content/uploads/sites/69/2019/01/manifoldmsijcv.pdf>`_
        R. Subbarao & P. Meer. International Journal of Computer Vision, 84,
        1-20, 2009
    """  # noqa

    def __init__(
        self,
        kernel="uniform",
        bandwidth=None,
        metric="riemann",
        tol=1e-3,
        max_iter=100,
        n_jobs=1
    ):
        """Init."""
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the modes of clusters.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : MeanShift instance
            The MeanShift instance.
        """
        self._kernel_fun = check_function(self.kernel, ker_clust_functions)
        self.metric_map, self.metric_dist = check_metric(self.metric)
        if self.bandwidth is None:
            self._bandwidth = self._estimate_bandwidth(X, quantile=0.3)
        self._bandwidth2 = self._bandwidth ** 2

        modes = Parallel(n_jobs=self.n_jobs)(
            delayed(self._seek_mode)(X, x) for x in X
        )

        modes = self._fuse_mode(modes)

        self.modes_ = np.array(modes)
        self.labels_ = self.predict(X)

        return self

    def _estimate_bandwidth(self, X, quantile):
        dist = pairwise_distance(X, None, metric=self.metric_dist)
        dist = np.triu(dist, 1)
        dist_sorted = np.sort(dist[dist > 0])
        bandwidth = dist_sorted[floor(quantile * len(dist_sorted))]
        print(f"MeanShift bandwidth={bandwidth:.3f}")
        return bandwidth

    def _seek_mode(self, X, mean):
        for _ in range(self.max_iter):
            T = log_map(X, mean, metric=self.metric_map)
            dist2 = distance(X, mean, metric=self.metric_dist, squared=True)
            weights = self._kernel_fun(dist2[:, 0] / self._bandwidth2)
            meanshift = np.einsum("a,abc->bc", weights, T) / np.sum(weights)
            mean = exp_map(meanshift, mean, metric=self.metric_map)
            if np.linalg.norm(meanshift) <= self.tol:
                break
        else:
            warnings.warn("Convergence not reached")

        return mean

    def _fuse_mode(self, in_modes):
        out_modes = in_modes.copy()
        in_modes = np.stack(in_modes, axis=0)
        dist = pairwise_distance(in_modes, None, metric=self.metric_dist)
        np.fill_diagonal(dist, self._bandwidth + 1)
        for i in range(dist.shape[0] - 1, -1, -1):
            if np.min(dist[i]) < self._bandwidth:
                del out_modes[i]
                dist[:, i] = self._bandwidth + 1

        if len(out_modes) == 0:
            raise ValueError(
                "No mode found, try other parameters (Got "
                f"kernel={self.kernel} and bandwith={self._bandwidth:.3f})"
            )

        return out_modes

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Prediction for each matrix according to the closest mode.
        """
        dist = Parallel(n_jobs=self.n_jobs)(
            delayed(distance)(X, mode, self.metric_dist)
            for mode in self.modes_
        )
        dist = np.concatenate(dist, axis=1)
        return dist.argmin(axis=1)


class Potato(TransformerMixin, SpdClassifMixin, BaseEstimator):
    """Artifact detection with the Riemannian Potato.

    The Riemannian Potato [1]_ is a clustering method used to detect artifact
    in multichannel signals. Processing SPD/HPD matrices,
    the algorithm iteratively estimates the centroid of clean
    matrices by rejecting every matrix that is too far from it.

    Parameters
    ----------
    metric : string | dict, default="riemann"
        Metric used for mean estimation (for the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`) and
        for distance estimation
        (see :func:`pyriemann.utils.distance.distance`).
        The metric can be a dict with two keys, "mean" and "distance"
        in order to pass different metrics.
    threshold : float, default=3
        Threshold on z-score of distance to reject artifacts. It is the number
        of standard deviations from the mean of distances to the centroid.
    n_iter_max : int, default=100
        The maximum number of iteration to reach convergence.
    pos_label : int, default=1
        The positive label corresponding to clean data.
    neg_label : int, default=0
        The negative label corresponding to artifact data.

    Notes
    -----
    .. versionadded:: 0.2.3

    Attributes
    ----------
    covmean_ : ndarray, shape (n_channels, n_channels)
        Centroid of potato.

    See Also
    --------
    MDM

    References
    ----------
    .. [1] `The Riemannian Potato: an automatic and adaptive artifact detection
        method for online experiments using Riemannian geometry
        <https://hal.archives-ouvertes.fr/hal-00781701>`_
        A. Barachant, A Andreev, and M. Congedo. TOBI Workshop lV, Jan 2013,
        Sion, Switzerland. pp.19-20.
    .. [2] `The Riemannian Potato Field: A Tool for Online Signal Quality Index
        of EEG
        <https://hal.archives-ouvertes.fr/hal-02015909>`_
        Q. Barthélemy, L. Mayaud, D. Ojeda, and M. Congedo. IEEE Transactions
        on Neural Systems and Rehabilitation Engineering, IEEE Institute of
        Electrical and Electronics Engineers, 2019, 27 (2), pp.244-255
    """

    def __init__(
        self,
        metric="riemann",
        threshold=3,
        n_iter_max=100,
        pos_label=1,
        neg_label=0,
    ):
        """Init."""
        self.metric = metric
        self.threshold = threshold
        self.n_iter_max = n_iter_max
        self.pos_label = pos_label
        self.neg_label = neg_label

    def fit(self, X, y=None, sample_weight=None):
        """Fit the potato.

        Fit the potato from SPD/HPD matrices, with an iterative outlier
        removal to obtain a reliable potato.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : None | ndarray, shape (n_matrices,), default=None
            Labels corresponding to each matrix: positive (resp. negative)
            label corresponds to a clean (resp. artifact) matrix.
            If None, all matrices are considered as clean.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : Potato instance
            The Potato instance.
        """
        if self.pos_label == self.neg_label:
            raise ValueError("Positive and negative labels must be different")

        n_matrices, _, _ = X.shape
        y_old = self._check_labels(X, y)

        if sample_weight is None:
            sample_weight = np.ones(n_matrices)

        self._mdm = MDM(metric=self.metric)

        for _ in range(self.n_iter_max):
            ix = (y_old == 1)
            if not any(ix):
                raise ValueError("Iterative outlier removal has rejected all "
                                 "matrices. Choose a higher threshold.")
            self._mdm.fit(X[ix], y_old[ix], sample_weight=sample_weight[ix])
            y = np.zeros(n_matrices)
            d = np.squeeze(np.log(self._mdm.transform(X[ix])))
            self._mean = np.mean(d)
            self._std = np.std(d)
            y[ix] = self._get_z_score(d) < self.threshold

            if np.array_equal(y, y_old):
                break
            else:
                y_old = y

        self.covmean_ = self._mdm.covmeans_[0]
        return self

    def partial_fit(self, X, y=None, *, sample_weight=None, alpha=0.1):
        """Partially fit the potato.

        This partial fit can be used to update dynamic or semi-dymanic online
        potatoes with clean matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : None | ndarray, shape (n_matrices,), default=None
            Labels corresponding to each matrix: positive (resp. negative)
            label corresponds to a clean (resp. artifact) matrix.
            If None, all matrices are considered as clean.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.
        alpha : float, default=0.1
            Update rate in [0, 1] for the centroid, and mean and standard
            deviation of log-distances: 0 for no update, 1 for full update.

        Returns
        -------
        self : Potato instance
            The Potato instance.
        """
        if not hasattr(self, "_mdm"):
            raise ValueError(
                "partial_fit can be called only on an already fitted potato."
            )

        n_matrices, n_channels, _ = X.shape
        if n_channels != self._mdm.covmeans_[0].shape[0]:
            raise ValueError(
                "X does not have the good number of channels. Should be %d but"
                " got %d." % (self._mdm.covmeans_[0].shape[0], n_channels)
            )

        y = self._check_labels(X, y)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        if not 0 <= alpha <= 1:
            raise ValueError("Parameter alpha must be in [0, 1]")
        if alpha == 0:
            return self

        Xm = mean_covariance(
            X[y == self.pos_label],
            metric=self.metric,
            sample_weight=sample_weight[y == self.pos_label],
        )
        self._mdm.covmeans_[0] = geodesic(
            self._mdm.covmeans_[0], Xm, alpha, metric=self.metric
        )

        d = np.squeeze(np.log(self._mdm.transform(Xm[np.newaxis, ...])))
        self._mean = (1 - alpha) * self._mean + alpha * d
        self._std = np.sqrt(
            (1 - alpha) * self._std**2 + alpha * (d - self._mean)**2
        )

        self.covmean_ = self._mdm.covmeans_[0]
        return self

    def transform(self, X):
        """Return the standardized log-distance to the centroid.

        Return the standardized log-distances to the centroids, ie geometric
        z-scores of distances.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        z : ndarray, shape (n_matrices,)
            Standardized log-distance to the centroid.
        """
        d = np.squeeze(np.log(self._mdm.transform(X)), axis=1)
        z = self._get_z_score(d)
        return z

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : None | ndarray, shape (n_matrices,), default=None
            Labels corresponding to each matrix: positive (resp. negative)
            label corresponds to a clean (resp. artifact) matrix.
            If None, all matrices are considered as clean.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        z : ndarray, shape (n_matrices,)
            Standardized log-distance to the centroid.
        """
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def predict(self, X):
        """Predict artifact from data.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        pred : ndarray of bool, shape (n_matrices,)
            The artifact detection: True if the matrix is clean, and False if
            the matrix contains an artifact.
        """
        z = self.transform(X)
        pred = z < self.threshold
        out = np.zeros_like(z) + self.neg_label
        out[pred] = self.pos_label
        return out

    def predict_proba(self, X):
        """Return probability of belonging to the potato / being clean.

        It is the probability to reject the null hypothesis "clean data",
        computing the right-tailed probability from z-score.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        proba : ndarray, shape (n_matrices,)
            Matrix is considered as normal/clean for high value of proba.
            It is considered as abnormal/artifacted for low value of proba.
        """
        z = self.transform(X)
        proba = self._get_proba(z)
        return proba

    def _check_labels(self, X, y):
        """Check validity of labels."""
        if y is not None:
            if len(y) != len(X):
                raise ValueError("y must be the same length of X")

            classes = np.int32(np.unique(y))

            if len(classes) > 2:
                raise ValueError("number of classes must be maximum 2")

            if self.pos_label not in classes:
                raise ValueError("y must contain a positive class")

            y = np.int32(np.array(y) == self.pos_label)

        else:
            y = np.ones(len(X))

        return y

    def _get_z_score(self, d):
        """Get z-score from distance."""
        z = (d - self._mean) / self._std
        return z

    def _get_proba(self, z):
        """Get right-tailed proba from z-score."""
        proba = 1 - norm.cdf(z)
        return proba


def _check_n_matrices(X, n_matrices):
    """Check number of matrices in ndarray."""
    if X.shape[0] != n_matrices:
        raise ValueError(
            "Unequal n_matrices between ndarray of X. Should be %d but"
            " got %d." % (n_matrices, X.shape[0])
        )


class PotatoField(TransformerMixin, SpdClassifMixin, BaseEstimator):
    """Artifact detection with the Riemannian Potato Field.

    The Riemannian Potato Field [1]_ is a clustering method used to detect
    artifact in multichannel signals. Processing SPD/HPD matrices,
    the algorithm combines several potatoes of low dimension,
    each one being designed to capture specific artifact typically
    affecting specific subsets of channels and/or specific frequency bands.

    Parameters
    ----------
    n_potatoes : int, default=1
        Number of potatoes in the field.
    p_threshold : float, default=0.01
        Threshold on probability to being clean, in (0, 1), combining
        probabilities of potatoes using Fisher's method.
    z_threshold : float, default=3
        Threshold on z-score of distance to reject artifacts. It is the number
        of standard deviations from the mean of distances to the centroid.
    metric : string | dict, default="riemann"
        Metric used for mean estimation (for the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`) and
        for distance estimation
        (see :func:`pyriemann.utils.distance.distance`).
        The metric can be a dict with two keys, "mean" and "distance"
        in order to pass different metrics.
    n_iter_max : int, default=10
        The maximum number of iteration to reach convergence.
    pos_label : int, default=1
        The positive label corresponding to clean data.
    neg_label : int, default=0
        The negative label corresponding to artifact data.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    Potato

    References
    ----------
    .. [1] `The Riemannian Potato Field: A Tool for Online Signal Quality Index
        of EEG
        <https://hal.archives-ouvertes.fr/hal-02015909>`_
        Q. Barthélemy, L. Mayaud, D. Ojeda, and M. Congedo. IEEE Transactions
        on Neural Systems and Rehabilitation Engineering, IEEE Institute of
        Electrical and Electronics Engineers, 2019, 27 (2), pp.244-255
    """

    def __init__(
        self,
        n_potatoes=1,
        p_threshold=0.01,
        z_threshold=3,
        metric="riemann",
        n_iter_max=10,
        pos_label=1,
        neg_label=0,
    ):
        """Init."""
        self.n_potatoes = int(n_potatoes)
        self.p_threshold = p_threshold
        self.metric = metric
        self.z_threshold = z_threshold
        self.n_iter_max = n_iter_max
        self.pos_label = pos_label
        self.neg_label = neg_label

    def fit(self, X, y=None, sample_weight=None):
        """Fit the potato field.

        Fit the potato field from SPD/HPD matrices, with iterative
        outlier removal to obtain reliable potatoes.

        Parameters
        ----------
        X : list of n_potatoes ndarrays of shape (n_matrices, n_channels, \
                n_channels) with same n_matrices but potentially different \
                n_channels
            List of sets of SPD/HPD matrices, each corresponding to a different
            subset of channels and/or filtering with a specific frequency band.
        y : None | ndarray, shape (n_matrices,), default=None
            Labels corresponding to each matrix: positive (resp. negative)
            label corresponds to a clean (resp. artifact) matrix.
            If None, all matrices are considered as clean.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : PotatoField instance
            The PotatoField instance.
        """
        if self.n_potatoes < 1:
            raise ValueError("Parameter n_potatoes must be at least 1")
        if not 0 < self.p_threshold < 1:
            raise ValueError("Parameter p_threshold must be in (0, 1)")
        self._check_length(X)
        n_matrices = X[0].shape[0]

        pt = Potato(
            metric=self.metric,
            threshold=self.z_threshold,
            n_iter_max=self.n_iter_max,
            pos_label=self.pos_label,
            neg_label=self.neg_label,
        )
        self._potatoes = []
        for i in range(self.n_potatoes):
            _check_n_matrices(X[i], n_matrices)
            self._potatoes.append(clone(pt))
            self._potatoes[i].fit(X[i], y, sample_weight=sample_weight)

        return self

    def partial_fit(self, X, y=None, *, sample_weight=None, alpha=0.1):
        """Partially fit the potato field.

        This partial fit can be used to update dynamic or semi-dymanic online
        potatoes with clean matrices.

        Parameters
        ----------
        X : list of n_potatoes ndarrays of shape (n_matrices, n_channels, \
                n_channels) with same n_matrices but potentially different \
                n_channels
            List of sets of SPD/HPD matrices, each corresponding to a different
            subset of channels and/or filtering with a specific frequency band.
        y : None | ndarray, shape (n_matrices,), default=None
            Labels corresponding to each matrix: positive (resp. negative)
            label corresponds to a clean (resp. artifact) matrix.
            If None, all matrices are considered as clean.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.
        alpha : float, default=0.1
            Update rate in [0, 1] for the centroid, and mean and standard
            deviation of log-distances: 0 for no update, 1 for full update.

        Returns
        -------
        self : PotatoField instance
            The PotatoField instance.
        """
        if not hasattr(self, "_potatoes"):
            raise ValueError("partial_fit can be called only on an already "
                             "fitted potato field.")

        self._check_length(X)
        n_matrices = X[0].shape[0]

        for i in range(self.n_potatoes):
            _check_n_matrices(X[i], n_matrices)
            self._potatoes[i].partial_fit(
                X[i],
                y,
                sample_weight=sample_weight,
                alpha=alpha,
            )
        return self

    def transform(self, X):
        """Return the standardized log-distances to the centroids.

        Return the standardized log-distances to the centroids, ie geometric
        z-scores of distances.

        Parameters
        ----------
        X : list of n_potatoes ndarrays of shape (n_matrices, n_channels, \
                n_channels) with same n_matrices but potentially different \
                n_channels
            List of sets of SPD/HPD matrices, each corresponding to a different
            subset of channels and/or filtering with a specific frequency band.

        Returns
        -------
        z : ndarray, shape (n_matrices, n_potatoes)
            Standardized log-distances to the centroids.
        """
        self._check_length(X)
        n_matrices = X[0].shape[0]

        z = np.zeros((n_matrices, self.n_potatoes))
        for i in range(self.n_potatoes):
            _check_n_matrices(X[i], n_matrices)
            z[:, i] = self._potatoes[i].transform(X[i])
        return z

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : list of n_potatoes ndarrays of shape (n_matrices, n_channels, \
                n_channels) with same n_matrices but potentially different \
                n_channels
            List of sets of SPD/HPD matrices, each corresponding to a different
            subset of channels and/or filtering with a specific frequency band.
        y : None | ndarray, shape (n_matrices,), default=None
            Labels corresponding to each matrix: positive (resp. negative)
            label corresponds to a clean (resp. artifact) matrix.
            If None, all matrices are considered as clean.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        z : ndarray, shape (n_matrices, n_potatoes)
            Standardized log-distances to the centroids.
        """
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def predict(self, X):
        """Predict artifact from data.

        Parameters
        ----------
        X : list of n_potatoes ndarrays of shape (n_matrices, n_channels, \
                n_channels) with same n_matrices but potentially different \
                n_channels
            List of sets of SPD/HPD matrices, each corresponding to a different
            subset of channels and/or filtering with a specific frequency band.

        Returns
        -------
        pred : ndarray of bool, shape (n_matrices,)
            The artifact detection: True if the matrix is clean, and False if
            the matrix contains an artifact.
        """
        p = self.predict_proba(X)
        pred = p > self.p_threshold
        out = np.zeros_like(p) + self.neg_label
        out[pred] = self.pos_label
        return out

    def predict_proba(self, X):
        """Predict probability obtained combining probabilities of potatoes.

        Predict probability obtained combining probabilities of potatoes using
        Fisher's method. A threshold of 0.01 can be used.

        Parameters
        ----------
        X : list of n_potatoes ndarrays of shape (n_matrices, n_channels, \
                n_channels) with same n_matrices but potentially different \
                n_channels
            List of sets of SPD/HPD matrices, each corresponding to a different
            subset of channels and/or filtering with a specific frequency band.

        Returns
        -------
        proba : ndarray, shape (n_matrices,)
            Matrix is considered as normal/clean for high value of proba.
            It is considered as abnormal/artifacted for low value of proba.
        """
        self._check_length(X)
        n_matrices = X[0].shape[0]

        p = np.zeros((self.n_potatoes, n_matrices))
        for i in range(self.n_potatoes):
            _check_n_matrices(X[i], n_matrices)
            p[i] = self._potatoes[i].predict_proba(X[i])
        p[p < 1e-10] = 1e-10  # avoid trouble with log
        q = - 2 * np.sum(np.log(p), axis=0)
        proba = self._get_proba(q)
        return proba

    def _check_length(self, X):
        """Check validity of input length."""
        if len(X) != self.n_potatoes:
            raise ValueError(
                "Length of X is not equal to n_potatoes. Should be %d but got "
                "%d." % (self.n_potatoes, len(X))
            )

    def _get_proba(self, q):
        """Get proba from a chi-squared value q."""
        proba = 1 - chi2.cdf(q, df=2 * self.n_potatoes)
        return proba
