"""Artifact detection."""
import numpy as np
from scipy.stats import combine_pvalues, norm
from sklearn.base import BaseEstimator, TransformerMixin

from .classification import MDM
from .geometry.geodesic import geodesic
from .geometry.mean import gmean
from .utils._base import SpdClassifMixin
from .utils._check import check_metric


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
        see :func:`pyriemann.geometry.mean.gmean`) and for distance estimation
        (see :func:`pyriemann.geometry.distance.distance`).
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

    Attributes
    ----------
    covmean_ : ndarray, shape (n_channels, n_channels)
        Centroid of potato.

    Notes
    -----
    .. versionadded:: 0.2.3
    .. versionchanged:: 0.12
        Move from clustering to artifactdetection.

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

        self._metric_mean, _ = check_metric(self.metric)
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

        Notes
        -----
        .. versionadded:: 0.3
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

        Xm = gmean(
            X[y == self.pos_label],
            metric=self._metric_mean,
            sample_weight=sample_weight[y == self.pos_label],
        )
        self._mdm.covmeans_[0] = geodesic(
            self._mdm.covmeans_[0], Xm, alpha, metric=self._metric_mean
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

        Notes
        -----
        .. versionadded:: 0.2.7
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
        probabilities of potatoes using ``method_combination``.
    z_threshold : float, default=3
        Threshold on z-score of distance to reject artifacts. It is the number
        of standard deviations from the mean of distances to the centroid.
    metric : string | dict | list, default="riemann"
        Metric used for mean estimation (for the list of supported metrics,
        see :func:`pyriemann.geometry.mean.gmean`) and for distance estimation
        (see :func:`pyriemann.geometry.distance.distance`).
        The metric can be a single str;
        or a dict with two keys, "mean" and "distance",
        in order to pass different metrics for mean and distance;
        or a list of ``n_potatoes`` str or dict,
        in order to pass different metrics for each potato [2]_.

        .. versionchanged:: 0.11
            Allow a different metric per potato.
    n_iter_max : int, default=10
        The maximum number of iteration to reach convergence.
    pos_label : int, default=1
        The positive label corresponding to clean data.
    neg_label : int, default=0
        The negative label corresponding to artifact data.
    method_combination : {"fisher", "stouffer"} | callable, default="fisher"
        Method to combine probabilities from the different potatoes:

        * fisher: Fisher's method;
        * stouffer: Stouffer's z-score method;
        * callable: for a custom combination function, with an axis argument.

        .. versionadded:: 0.11

    Notes
    -----
    .. versionadded:: 0.3
    .. versionchanged:: 0.12
        Move from clustering to artifactdetection.

    See Also
    --------
    Potato

    References
    ----------
    .. [1] `The Riemannian Potato Field: A Tool for Online Signal Quality Index
        of EEG
        <https://hal.archives-ouvertes.fr/hal-02015909>`_
        Q. Barthélemy, L. Mayaud, D. Ojeda, and M. Congedo. IEEE
        Transactions on Neural Systems and Rehabilitation Engineering, 2019
    .. [2] `Improved Riemannian potato field: an Automatic Artifact Rejection
        Method for EEG
        <https://arxiv.org/pdf/2509.09264>`_
        D. Hajhassani, Q. Barthélemy, J. Mattout & M. Congedo.
        Biomedical Signal Processing and Control, 2026
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
        method_combination="fisher",
    ):
        """Init."""
        self.n_potatoes = int(n_potatoes)
        self.p_threshold = p_threshold
        self.metric = metric
        self.z_threshold = z_threshold
        self.n_iter_max = n_iter_max
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.method_combination = method_combination

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

        if isinstance(self.metric, (str, dict)):
            metric = [self.metric] * self.n_potatoes
        elif isinstance(self.metric, list):
            if len(self.metric) == self.n_potatoes:
                metric = self.metric
            else:
                raise ValueError(
                    f"Metric must be a list with {self.n_potatoes} elements."
                )
        else:
            raise TypeError(
                "Metric must be a str, a dict or a list, "
                f"but got {type(self.metric)}."
            )

        self._potatoes = []
        for i in range(self.n_potatoes):
            _check_n_matrices(X[i], n_matrices)
            pt = Potato(
                metric=metric[i],
                threshold=self.z_threshold,
                n_iter_max=self.n_iter_max,
                pos_label=self.pos_label,
                neg_label=self.neg_label,
            )
            self._potatoes.append(pt)
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
        """Predict probability combining probabilities of potatoes.

        Predict probability combining probabilities of the different potatoes
        using ``method_combination``.
        With Fisher's method, a threshold of 0.01 can be used.

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

        probas = np.zeros((self.n_potatoes, n_matrices))
        for i in range(self.n_potatoes):
            _check_n_matrices(X[i], n_matrices)
            probas[i] = self._potatoes[i].predict_proba(X[i])
        probas = np.clip(probas, a_min=1e-10, a_max=1)  # avoid trouble w. log

        if isinstance(self.method_combination, str):
            _, proba = combine_pvalues(
                probas,
                method=self.method_combination,
                axis=0,
            )
        elif callable(self.method_combination):
            proba = self.method_combination(probas, axis=0)
        else:
            raise TypeError(
                "method_combination must be a str or a callable, "
                f"but got {type(self.method_combination)}."
            )

        return proba

    def _check_length(self, X):
        """Check validity of input length."""
        if len(X) != self.n_potatoes:
            raise ValueError(
                "Length of X is not equal to n_potatoes. Should be %d but got "
                "%d." % (self.n_potatoes, len(X))
            )
