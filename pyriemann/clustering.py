"""Clustering."""
from math import floor
import warnings

from joblib import Parallel, delayed
import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans as sklearnKMeans
from sklearn.utils.validation import check_random_state

from .classification import MDM
from .datasets import sample_gaussian_spd
from .geometry.covariance import covariance_scm
from .geometry.distance import (
    distance,
    pairwise_distance,
    distance_mahalanobis,
)
from .geometry.mean import gmean
from .geometry.tangentspace import exp_map, log_map, tangent_space
from .utils._base import SpdClassifMixin, SpdClustMixin, SpdTransfMixin
from .utils._check import check_metric, check_function, check_weights


def _init_centroids(X, n_clusters, init, random_state, x_squared_norms):
    if random_state is not None:
        random_state = np.random.RandomState(random_state)
    if sklearn.__version__ < "1.3.0":
        return sklearnKMeans(n_clusters=n_clusters, init=init)._init_centroids(
            X,
            x_squared_norms,
            init,
            random_state,
        )
    else:
        n_matrices = X.shape[0]
        return sklearnKMeans(n_clusters=n_clusters, init=init)._init_centroids(
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
    mdm._metric_mean, mdm._metric_dist = check_metric(metric)
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


class Kmeans(SpdClassifMixin, SpdClustMixin, SpdTransfMixin, BaseEstimator):
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
        see :func:`pyriemann.geometry.mean.gmean`) and for distance estimation
        (see :func:`pyriemann.geometry.distance.distance`).
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
        self.random_state = random_state
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
            np.random.seed(self.random_state)
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
                random_state=self.random_state,
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

    def centroids(self):
        """Helper for fast access to the centroids.

        Returns
        -------
        centroids : ndarray, shape (n_clusters, n_channels, n_channels)
            Centroids of each cluster.
        """
        return self.mdm_.covmeans_


class KmeansPerClassTransform(SpdTransfMixin, BaseEstimator):
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
        mdm._metric_mean, mdm._metric_dist = check_metric(self.metric)
        mdm.covmeans_ = self.covmeans_
        return mdm._predict_distances(X)


###############################################################################


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
        see :func:`pyriemann.geometry.tangentspace.log_map`) and
        for distance estimation
        (see :func:`pyriemann.geometry.distance.distance`).
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
        self._metric_map, self._metric_dist = check_metric(
            self.metric, ["map", "dist"]
        )
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
        dist = pairwise_distance(X, None, metric=self._metric_dist)
        dist = np.triu(dist, 1)
        dist_sorted = np.sort(dist[dist > 0])
        bandwidth = dist_sorted[floor(quantile * len(dist_sorted))]
        print(f"MeanShift bandwidth={bandwidth:.3f}")
        return bandwidth

    def _seek_mode(self, X, mean):
        for _ in range(self.max_iter):
            T = log_map(X, mean, metric=self._metric_map)
            dist2 = distance(X, mean, metric=self._metric_dist, squared=True)
            weights = self._kernel_fun(dist2[:, 0] / self._bandwidth2)
            meanshift = np.einsum("a,abc->bc", weights, T) / np.sum(weights)
            mean = exp_map(meanshift, mean, metric=self._metric_map)
            if np.linalg.norm(meanshift) <= self.tol:
                break
        else:
            warnings.warn("Convergence not reached")

        return mean

    def _fuse_mode(self, in_modes):
        out_modes = in_modes.copy()
        in_modes = np.stack(in_modes, axis=0)
        dist = pairwise_distance(in_modes, None, metric=self._metric_dist)
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
            delayed(distance)(X, mode, self._metric_dist)
            for mode in self.modes_
        )
        dist = np.concatenate(dist, axis=1)
        return dist.argmin(axis=1)


###############################################################################


class Gaussian():
    """Gaussian model.

    Gaussian model for Riemannian manifold of SPD matrices,
    defined with a mean in manifold and a covariance in tangent space [1]_.

    Parameters
    ----------
    n : integer
        Dimension of the matrices.
    mu : ndarray, shape (n, n)
        Mean of the Gaussian, in manifold.
    sigma : None | ndarray, shape (n * (n + 1) / 2, n * (n + 1) / 2), \
            default=None
        Covariance of the Gaussian, in tangent space.
        If None, it uses identity matrix.
    metric : string | dict, default="riemann"
        Metric used for mean update (for the list of supported metrics,
        see :func:`pyriemann.geometry.mean.gmean`) and for tangent space map
        (see :func:`pyriemann.geometry.tangent_space.tangent_space`).
        The metric can be a dict with two keys, "mean" and "map"
        in order to pass different metrics.

    Notes
    -----
    .. versionadded:: 0.11

    References
    ----------
    .. [1] `Intrinsic statistics on Riemannian manifolds: Basic tools for
        geometric measurements
        <https://www.cis.jhu.edu/~tingli/App_of_Lie_group/Intrinsic%20Statistics%20on%20Riemannian%20Manifolds.pdf>`_
        X. Pennec. Journal of Mathematical Imaging and Vision, 2006
    """  # noqa
    def __init__(self, n, mu, sigma=None, metric="riemann"):
        self.n = n
        self.mu = mu
        if sigma is None:
            sigma = np.eye(n * (n + 1) // 2)
        self.sigma = sigma
        self.metric = metric
        self._metric_mean, self._metric_map = check_metric(
            metric, ["mean", "map"]
        )

    def pdf(self, X, *, reg=1e-16, use_pi=True):
        """Compute approximate probability density function (pdf) of matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n, n)
            Set of SPD matrices.
        reg : float, default=1e-16
            Regularization parameter for pdf normalization term.
        use_pi : bool, default=True
            If true, use (2 pi)^n to compute the full denominator.
            If false, do not use (2 pi)^n, because will be simplified with
            upcoming normalizations.

        Returns
        -------
        pdf : ndarray, shape (n_matrices,)
            Probability density function of each matrix.
        """
        TangVec = tangent_space(X, self.mu, metric=self._metric_map)
        dist = distance_mahalanobis(TangVec.T, self.sigma, squared=True)
        num = np.exp(-0.5 * dist)
        det = np.linalg.det(self.sigma)
        if use_pi:
            denom = np.sqrt(((2 * np.pi) ** self.n) * det)
        else:
            denom = np.sqrt(det)
        return num / (denom + reg)

    def update_mean(self, X, sample_weight):
        """Update mean in manifold.

        Compute weighted mean of matrices, initialized on previous mean.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n, n)
            Set of SPD matrices.
        sample_weight : ndarray, shape (n_matrices,)
            Weights for each matrix.
        """
        self.mu = gmean(
            X,
            metric=self._metric_mean,
            sample_weight=sample_weight,
            init=self.mu,
        )

    def update_covariance(self, X, sample_weight):
        """Update covariance in tangent space.

        Compute weighted covariance of tangent vectors.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n, n)
            Set of SPD matrices.
        sample_weight : ndarray, shape (n_matrices,)
            Weights for each matrix.
        """
        TangVec = tangent_space(X, self.mu, metric=self._metric_map)
        self.sigma = covariance_scm(
            TangVec.T,
            assume_centered=True,
            weights=sample_weight,
        )


class GaussianMixture(SpdClustMixin, BaseEstimator):
    """Gaussian mixture model.

    Representation of a Gaussian mixture model (GMM) probability distribution
    for SPD matrices by expectation-maximization (EM) algorithm [1]_.

    Parameters
    ----------
    n_components : integer, default=1
        Number of mixture components.
    metric : string | dict, default="riemann"
        Metric used for mean update (for the list of supported metrics,
        see :func:`pyriemann.geometry.mean.gmean`) and for tangent space map
        (see :func:`pyriemann.geometry.tangent_space.tangent_space`).
        The metric can be a dict with two keys, "mean" and "map"
        in order to pass different metrics.
    weights_init : None | ndarray, shape (n_components,), defaut=None
        Initial weights. If None, it uses equal weights.
    means_init : None | ndarray, shape (n_components,), defaut=None
        Initial means of Gaussians. If None, it randomly selects training
        matrices.
    tol : float, default=1e-5
        Tolerance to stop the EM algorithm.
    maxiter : int, default=100
        Maximum number of iterations of EM algorithm.
    random_state : None | integer | np.RandomState, default=None
        The generator used to initialize the Gaussian models. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    verbose : bool, default=False
        Verbose flag.

    Attributes
    ----------
    weights_ : ndarray, shape (n_components,)
        Weight of each mixture component.
    means_ : ndarray, shape (n_components, n_channels, n_channels)
        Mean of each mixture component.
    covariances_ : ndarray, shape (n_components, n_ts, n_ts)
        Covariance of each mixture component.

    Notes
    -----
    .. versionadded:: 0.11

    References
    ----------
    .. [1] `Gaussian mixture regression on symmetric positive definite matrices
        manifolds: Application to wrist motion estimation with sEMG
        <https://calinon.ch/papers/Jaquier-IROS2017.pdf>`_
        N. Jacquier & S. Calinon. IEEE IROS, 2017
    """
    def __init__(
        self,
        n_components=1,
        metric="riemann",
        weights_init=None,
        means_init=None,
        tol=1e-5,
        maxiter=100,
        random_state=None,
        verbose=False,
    ):
        """Init."""
        self.n_components = n_components
        self.metric = metric
        self.weights_init = weights_init
        self.means_init = means_init
        self.tol = tol
        self.maxiter = maxiter
        self.random_state = random_state
        self.verbose = verbose

    @property
    def means_(self):
        return np.stack([component.mu for component in self._components])

    @property
    def covariances_(self):
        return np.stack([component.sigma for component in self._components])

    def _get_wlik(self, X, use_pi=True):
        """Compute weighted likelihoods.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        use_pi : bool, default=True
            If true, use (2 pi)^n to compute the full denominator of pdf.
            If false, do not use (2 pi)^n, because will be simplified with
            upcoming normalizations.

        Returns
        -------
        wlik : ndarray, shape (n_matrices, n_components)
            Weighted likelihood of each matrix given component.
        """
        wlik = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            lik = self._components[k].pdf(X, use_pi=use_pi)
            wlik[:, k] = self.weights_[k] * lik
        return wlik

    def _get_proba(self, X, reg=1e-16):
        """Compute posterior probabilities.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        reg : float, default=1e-16
            Regularization parameter for probabilities normalization.

        Returns
        -------
        prob : ndarray, shape (n_matrices, n_components)
            Posterior probability of each component given matrix.
        """
        num = self._get_wlik(X, use_pi=False)
        prob = num / (np.sum(num, axis=1, keepdims=True) + reg)
        return prob

    def _log(self, X):
        """Log after clip."""
        return np.log(np.clip(X, a_min=1e-10, a_max=None))

    def fit(self, X, y=None):
        """Fit the mixture with EM.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : GaussianMixture instance
            The GaussianMixture instance.
        """
        n_matrices, n_channels, _ = X.shape
        if (n_channels * (n_channels + 1) // 2 > n_matrices):
            raise ValueError("Not enough matrices for training GMM.")

        # initialization
        self.random_state = check_random_state(self.random_state)

        if isinstance(self.means_init, np.ndarray) and self.means_init.shape \
                == (self.n_components, n_channels, n_channels):
            means_init = self.means_init
        else:
            inds = self.random_state.randint(
                n_matrices,
                size=(self.n_components,)
            )
            means_init = X[inds]

        self._components = []
        for k in range(self.n_components):
            self._components.append(
                Gaussian(
                    n_channels,
                    mu=means_init[k],
                    sigma=None,
                    metric=self.metric,
                )
            )

        self.weights_ = check_weights(self.weights_init, self.n_components)

        # expectation-maximization
        crit = 0
        for _ in range(self.maxiter):
            # e-step
            prob = self._get_proba(X)

            # m-step
            self.weights_ = np.sum(prob, axis=0) / n_matrices
            # re-normalization (necessary because of approx Gaussian pdf?)
            self.weights_ = self.weights_ / self.weights_.sum()
            for k in range(self.n_components):
                self._components[k].update_mean(X, prob[:, k])
                self._components[k].update_covariance(X, prob[:, k])

            # check convergence
            crit_new = -np.sum(self._log(np.sum(self._get_wlik(X), axis=1)))
            if self.verbose:
                print(f"neg log-likelihood = {crit_new}")
            if abs(crit - crit_new) < self.tol:
                break
            crit = crit_new
        else:
            warnings.warn("EM convergence not reached")

        return self

    def predict_proba(self, X):
        """Predict probabilities.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_matrices, n_components)
            Probabilities for each component.
        """
        return self._get_proba(X)

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix.
        """
        prob = self._get_proba(X)
        return np.argmax(prob, axis=1)

    def score(self, X, y=None):
        """Compute the average log-likelihood of the given matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        score : float
            Log-likelihood of matrices under the Gaussian mixture model.
        """
        lik = np.sum(self._get_wlik(X), axis=1)
        return np.mean(self._log(lik))

    def sample(self, n_matrices=1):
        """Generate random matrices from the fitted Gaussian distribution.

        Warning: GMM is calibrated using the Gaussian model [1]_,
        while this sampling uses the wrapped Gaussian model [2]_.

        Parameters
        ----------
        n_matrices : int, default=1
            Number of matrices to generate.

        Returns
        -------
        X : array, shape (n_matrices, n_channels, n_channels)
            Randomly generated matrices.
        y : array, shape (n_matrices,)
            Component labels.

        References
        ----------
        .. [1] `Intrinsic statistics on Riemannian manifolds: Basic tools for
            geometric measurements
            <https://www.cis.jhu.edu/~tingli/App_of_Lie_group/Intrinsic%20Statistics%20on%20Riemannian%20Manifolds.pdf>`_
            X. Pennec. Journal of Mathematical Imaging and Vision, 2006
        .. [2] `Wrapped gaussian on the manifold of symmetric positive
            definite matrices
            <https://openreview.net/pdf?id=EhStXG4dCS>`_
            T. de Surrel, F. Lotte, S. Chevallier, and F. Yger. ICML, 2025
        """  # noqa
        y = self.random_state.randint(self.n_components, size=(n_matrices,))

        means, covariances = self.means_, self.covariances_
        n_channels = means.shape[-1]

        X = np.zeros((n_matrices, means.shape[-1], n_channels))
        for i in np.unique(y):
            X[y == i] = sample_gaussian_spd(
                np.count_nonzero(y == i),
                mean=means[i],
                sigma=covariances[i],
                random_state=self.random_state
            )

        return X, y


###############################################################################


def __getattr__(name):
    if name == "Potato":
        warnings.warn(
            "clustering.Potato is deprecated and will be removed in 0.14.0; "
            "use artifactdetection.Potato instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from .artifactdetection import Potato
        return Potato
    elif name == "PotatoField":
        warnings.warn(
            "clustering.PotatoField is deprecated and will be removed in "
            "0.14.0; use artifactdetection.PotatoField instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from .artifactdetection import PotatoField
        return PotatoField
