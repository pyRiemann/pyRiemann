"""Code for channel selection."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .utils.distance import distance
from .classification import MDM


class ElectrodeSelection(TransformerMixin, BaseEstimator):

    """Channel selection based on a Riemannian geometry criterion.

    For each class, a centroid is estimated, and the channel selection is based
    on the maximization of the distance between centroids. This is done by a
    backward elimination where the electrode that carries the less distance is
    removed from the subset at each iteration [1]_.

    Parameters
    ----------
    nelec : int, default=16
        The number of electrode to keep in the final subset.
    metric : string | dict, default="riemann"
        Metric used for mean estimation (for the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`) and
        for distance estimation
        (see :func:`pyriemann.utils.distance.distance`).
        The metric can be a dict with two keys, "mean" and "distance"
        in order to pass different metrics.
    n_jobs : int, default=1
        Number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    covmeans_ : ndarray, shape (n_classes, n_channels, n_channels)
        Centroids for each class.
    dist_ : list
        Distance at each iteration.
    self.subelec_ : list
        Indices of selected channels.

    See Also
    --------
    Kmeans
    FgMDM

    References
    ----------
    .. [1] `Channel selection procedure using riemannian distance for BCI
        applications
        <https://hal.archives-ouvertes.fr/hal-00602707>`_
        A. Barachant and S. Bonnet. The 5th International IEEE EMBS Conference
        on Neural Engineering, Apr 2011, Cancun, Mexico.
    """

    def __init__(self, nelec=16, metric="riemann", n_jobs=1):
        """Init."""
        self.nelec = nelec
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y=None, sample_weight=None):
        """Find the optimal subset of electrodes.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None | ndarray, shape (n_matrices,), default=None
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : ElectrodeSelection instance
            The ElectrodeSelection instance.
        """
        if y is None:
            y = np.ones((X.shape[0]))

        mdm = MDM(metric=self.metric, n_jobs=self.n_jobs)
        mdm.fit(X, y, sample_weight=sample_weight)
        self.covmeans_ = mdm.covmeans_

        n_classes, n_channels, _ = self.covmeans_.shape

        self.dist_ = []
        self.subelec_ = list(range(n_channels))
        while (len(self.subelec_)) > self.nelec:
            di = np.zeros((len(self.subelec_), 1))
            for idx in range(len(self.subelec_)):
                sub = self.subelec_[:]
                sub.pop(idx)
                di[idx] = 0
                for i in range(n_classes):
                    for j in range(i + 1, n_classes):
                        di[idx] += distance(
                            self.covmeans_[i][:, sub][sub, :],
                            self.covmeans_[j][:, sub][sub, :],
                            metric=mdm.metric_dist,
                        )

            torm = di.argmax()
            self.dist_.append(di.max())
            self.subelec_.pop(torm)
        return self

    def transform(self, X):
        """Return reduced matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_elec, n_elec)
            Set of SPD matrices after reduction of the number of channels.
        """
        return X[:, self.subelec_, :][:, :, self.subelec_]

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None | ndarray, shape (n_matrices,), default=None
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_elec, n_elec)
            Set of SPD matrices after reduction of the number of channels.
        """
        return self.fit(X, y, sample_weight=sample_weight).transform(X)


class FlatChannelRemover(TransformerMixin, BaseEstimator):
    """Flat channel removal.

    Attributes
    ----------
    channels_ : ndarray, shape (n_good_channels,)
        Indices of the non-flat channels.
    """

    def fit(self, X, y=None):
        """Find flat channels.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : FlatChannelRemover instance
            The FlatChannelRemover instance.
        """
        std = np.mean(np.std(X, axis=2) ** 2, 0)
        self.channels_ = np.where(std)[0]
        return self

    def transform(self, X):
        """Remove flat channels.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_good_channels, n_times)
            Multi-channel time-series without flat channels.
        """
        return X[:, self.channels_, :]

    def fit_transform(self, X, y=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_good_channels, n_times)
            Multi-channel time-series without flat channels.
        """
        return self.fit(X, y).transform(X)
