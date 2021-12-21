import numpy as np
from joblib import Parallel, delayed

from .classification import MDM
from .utils.mean import mean_covariance
from .utils.geodesic import geodesic


class MDWM (MDM):
    """Classification by Minimum Distance to Weighted Mean.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated, according to the chosen metric, as a weighted mean
    of covariance matrices from the source domain, combined with
    the class centroid of the target domain [1]_ [2]_.
    For classification, a given new point is attibuted to the class whose
    centroid is the nearest according to the chosen metric.

    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    Lambda : float, (default: 0)
        Transfer coefficient in [0,1], controlling the trade-off between
        source and target data. At 0, there is no transfer, only the data
        acquired from the source are used. At 1, this is a calibration-free
        system as no data are required from the source.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    covmeans_ : list
        the class centroids.
    classes_ : list
        list of classes.

    See Also
    --------
    MDM

    References
    ----------
    .. [1] E. Kalunga, S. Chevallier and Q. Barthélemy, "Transfer learning for
        SSVEP-based BCI using Riemannian similarities between users", in 26th
        European Signal Processing Conference (EUSIPCO), pp. 1685-1689. IEEE,
        2018.

    .. [2] S. Khazem, S. Chevallier, Q. Barthélemy, K. Haroun and C. Noûs,
        "Minimizing Subject-dependent Calibration for BCI with Riemannian
        Transfer Learning", in 10th International IEEE/EMBS Conference on
        Neural Engineering (NER), pp. 523-526. IEEE, 2021.
    """

    def __init__(self, metric='riemann', Lambda=0, n_jobs=1):
        """Init."""
        self.metric = metric
        self.n_jobs = n_jobs
        self.Lambda = Lambda
        self.target_means_ = None
        self.domain_means_ = None
        self.classes_ = None

        if isinstance(metric, str):
            self.metric_mean = metric
            self.metric_dist = metric

        elif isinstance(metric, dict):
            # check keys
            for key in ['mean', 'distance']:
                if key not in metric.keys():
                    raise KeyError('metric must contain "mean" and "distance"')

            self.metric_mean = metric['mean']
            self.metric_dist = metric['distance']

        else:
            raise TypeError('metric must be dict or str')

    def fit(self, X, y, X_source, y_source, sample_weight=None):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Covariance matrices from target domain
        y : ndarray shape (n_matrices, 1)
            labels corresponding to each matrix of target domain
        X_source : ndarray, shape (n_matrices, n_channels, n_channels)
            Covariance matrices from source domain
        y_source : ndarray shape (n_matrices, 1)
            labels corresponding to each matrix of source domain
        sample_weight : None | ndarray shape (n_matrices, 1)
            the weights of each sample from the domain. if None, each sample
            is treated with equal weights.

        Returns
        -------
        self : MDWM instance
            The MDWM instance.
        """

        if not 0 <= self.Lambda <= 1:
            raise ValueError(
                'Value Lambda must be included in [0, 1] (Got %d)'
                % self.Lambda)

        if set(y) != set(y_source):
            raise Exception(f"classes in source domain must match classes in \
                target domain. Classes in source are {np.unique(y_source)} \
                while classes in target are {np.unique(y)}")

        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X_source.shape[0])

        if self.n_jobs == 1:
            self.target_means_ = [
                mean_covariance(X[y == ll], metric=self.metric_mean)
                for ll in self.classes_]
            print(f"[DEBUG] self.classes_ {self.classes_}")
            print(f"[DEBUG] y_source {y_source}")
            print(f"[DEBUG] X_source.shape {X_source.shape}")
            self.domain_means_ = [
                mean_covariance(
                    X_source[y_source == ll],
                    metric=self.metric_mean,
                    sample_weight=sample_weight[y_source == ll]
                    )
                for ll in self.classes_]
        else:
            self.target_means_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(X[y == ll], metric=self.metric_mean)
                for ll in self.classes_)
            self.domain_means_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(
                    X_source[y_source == ll],
                    metric=self.metric_mean,
                    sample_weight=sample_weight[y_source == ll])
                for ll in self.classes_)

        self.covmeans_ = [geodesic(self.target_means_[i],
                                   self.domain_means_[i],
                                   self.Lambda, self.metric)
                          for i, _ in enumerate(self.classes_)]
        return self
