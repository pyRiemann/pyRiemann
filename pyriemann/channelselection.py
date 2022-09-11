"""Code for channel selection."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .utils.distance import distance
from .classification import MDM


class ElectrodeSelection(BaseEstimator, TransformerMixin):

    """Channel selection based on a Riemannian geometry criterion.

    For each class, a centroid is estimated, and the channel selection is based
    on the maximization of the distance between centroids. This is done by a
    backward elimination where the electrode that carries the less distance is
    removed from the subset at each iteration.
    This algorithm is described in [1]_.

    Parameters
    ----------
    nelec : int, default=16
        The number of electrode to keep in the final subset.
    metric : string | dict, default='riemann'
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the selection.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    covmeans_ : list
        The class centroids.
    dist_ : list
        List of distance at each interation.

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

    def __init__(self, nelec=16, metric='riemann', n_jobs=1):
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

        n_channels, _ = self.covmeans_[0].shape

        self.dist_ = []
        self.subelec_ = list(range(0, n_channels, 1))
        while (len(self.subelec_)) > self.nelec:
            di = np.zeros((len(self.subelec_), 1))
            for idx in range(len(self.subelec_)):
                sub = self.subelec_[:]
                sub.pop(idx)
                di[idx] = 0
                for i in range(len(self.covmeans_)):
                    for j in range(i + 1, len(self.covmeans_)):
                        di[idx] += distance(self.covmeans_[i][:, sub][sub, :],
                                            self.covmeans_[j][:, sub][sub, :],
                                            metric=mdm.metric_dist)
            # print di
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
        covs : ndarray, shape (n_matrices, n_elec, n_elec)
            Set of SPD matrices after reduction of the number of channels.
        """
        return X[:, self.subelec_, :][:, :, self.subelec_]


class FlatChannelRemover(BaseEstimator, TransformerMixin):
    """Finds and removes flat channels.

    Attributes
    ----------
    channels_ : ndarray, shape (n_good_channels)
        The indices of the non-flat channels.
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
        X : ndarray, shape (n_matrices, n_good_channels, n_times)
            Multi-channel time-series without flat channels.
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
        X : ndarray, shape (n_matrices, n_good_channels, n_times)
            Multi-channel time-series without flat channels.
        """
        return X[:, self.channels_, :]

    def fit_transform(self, X, y=None):
        """Find and remove flat channels.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X : ndarray, shape (n_matrices, n_good_channels, n_times)
            Multi-channel time-series without flat channels.
        """
        self.fit(X, y)
        return self.transform(X)
