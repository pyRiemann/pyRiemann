import numpy as np
from joblib import Parallel, delayed
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    is_classifier,
    is_regressor
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score

from ..utils.mean import mean_covariance, mean_riemann
from ..utils.distance import distance
from ..utils.base import invsqrtm, powm, sqrtm
from ..utils.geodesic import geodesic
from ..utils.utils import check_weights, check_metric
from ._rotate import _get_rotation_matrix
from ..classification import MDM
from ..preprocessing import Whitening
from ._tools import decode_domains


class TLDummy(BaseEstimator, TransformerMixin):
    """No transformation on data for transfer learning.

    No transformation of the data points between the domains.
    This is what we call the Direct Center Transfer (DCT) method.

    Notes
    -----
    .. versionadded:: 0.4
    """

    def __init__(self):
        pass

    def fit(self, X, y_enc):
        """Do nothing.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.

        Returns
        -------
        self : TLDummy instance
            The TLDummy instance.
        """
        return self

    def transform(self, X, y_enc=None):
        """Do nothing.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X : ndarray, shape (n_matrices, n_classes)
            Same set of SPD matrices as in the input.
        """
        return X

    def fit_transform(self, X, y_enc):
        """Do nothing.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.

        Returns
        -------
        X : ndarray, shape (n_matrices, n_classes)
            Set of SPD matrices with mean in the Identity.
        """
        return self.fit(X, y_enc).transform(X, y_enc)


class TLCenter(BaseEstimator, TransformerMixin):
    """Recenter data for transfer learning.

    Recenter the data points from each domain to the Identity on manifold, ie
    make the mean of the datasets become the identity. This operation
    corresponds to a whitening step if the SPD matrices represent the spatial
    covariance matrices of multivariate signals.

    .. note::
       Using .fit() and then .transform() will give different results than
       .fit_transform(). In fact, .fit_transform() should be applied on the
       training dataset (target and source) and .transform() on the test
       partition of the target dataset.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target in ``transform()`` function:

        * if not empty, ``transform()`` recenters matrices to the specified
          target domain;
        * else, ``transform()`` recenters matrices to the last fitted domain.
    metric : str, default="riemann"
        Metric used for mean estimation. For the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`.
        Note, however, that only when using the "riemann" metric that we are
        ensured to re-center the matrices precisely to the Identity.

    Attributes
    ----------
    recenter_ : dict
        If fit, dictionary with key=domain_name and value=domain_mean.

    References
    ----------
    .. [1] `Transfer Learning: A Riemannian Geometry Framework With
        Applications to Brain–Computer Interfaces
        <https://hal.archives-ouvertes.fr/hal-01923278/>`_
        P Zanini et al, IEEE Transactions on Biomedical Engineering, vol. 65,
        no. 5, pp. 1107-1116, August, 2017
    .. [2] `Transfer Learning for Brain-Computer Interfaces:
        A Euclidean Space Data Alignment Approach
        <https://arxiv.org/abs/1808.05464>`_
        He He and Dongrui Wu, IEEE Transactions on Biomedical Engineering, 2019

    Notes
    -----
    .. versionadded:: 0.4
    """

    def __init__(self, target_domain, metric="riemann"):
        """Init"""
        self.target_domain = target_domain
        self.metric = metric

    def fit(self, X, y_enc, sample_weight=None):
        """Fit TLCenter.

        For each domain, calculates the mean of matrices of this domain.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : TLCenter instance
            The TLCenter instance.
        """
        _, _, domains = decode_domains(X, y_enc)
        n_matrices, _, _ = X.shape
        sample_weight = check_weights(sample_weight, n_matrices)

        self.recenter_ = {}
        for d in np.unique(domains):
            idx = domains == d
            self.recenter_[d] = Whitening(metric=self.metric).fit(
                X[idx], sample_weight=sample_weight[idx]
            )
        return self

    def transform(self, X, y_enc=None):
        """Re-center matrices in the target domain.

        .. note::
           This method is designed for using at test time,
           recentering all matrices in target domain.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X : ndarray, shape (n_matrices, n_classes)
            Set of recentered SPD matrices.
        """
        # if target domain is specified, use it
        if self.target_domain != "":
            target_domain = self.target_domain
        # else, use last calibrated domain as target domain
        else:
            keys = list(self.recenter_.keys())
            target_domain = keys[-1]

        X_rct = self.recenter_[target_domain].transform(X)
        return X_rct

    def fit_transform(self, X, y_enc, sample_weight=None):
        """Fit TLCenter and then transform matrices.

        For each domain, calculates the mean of matrices of this domain and
        then recenters them to Identity.

        .. note::
           This method is designed for using at training time. The output for
           .fit_transform() will be different than using .fit() and
           .transform() separately.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        X : ndarray, shape (n_matrices, n_classes)
            Set of recentered SPD matrices.
        """
        self.fit(X, y_enc, sample_weight)
        _, _, domains = decode_domains(X, y_enc)

        X_rct = np.zeros_like(X)
        for d in np.unique(domains):
            idx = domains == d
            X_rct[idx] = self.recenter_[d].transform(X[idx])
        return X_rct


class TLStretch(BaseEstimator, TransformerMixin):
    """Stretch data for transfer learning.

    Change the dispersion of the datapoints around their geometric mean
    for each dataset so that they all have the same desired value.

    .. note::
       Using .fit() and then .transform() will give different results than
       .fit_transform(). In fact, .fit_transform() should be applied on the
       training dataset (target and source) and .transform() on the test
       partition of the target dataset.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    dispersion : float, default=1.0
        Target value for the dispersion of the data points.
    centered_data : bool, default=False
        Whether the data has been re-centered to the Identity beforehand.
    metric : str, default="riemann"
        Metric used for calculating the dispersion.
        For the list of supported metrics,
        see :func:`pyriemann.utils.distance.distance`.

    Attributes
    ----------
    dispersions_ : dict
        Dictionary with key=domain_name and value=domain_dispersion.

    References
    ----------
    .. [1] `Riemannian Procrustes analysis: transfer learning for
        brain-computer interfaces
        <https://hal.archives-ouvertes.fr/hal-01971856>`_
        PLC Rodrigues et al, IEEE Transactions on Biomedical Engineering,
        vol. 66, no. 8, pp. 2390-2401, December, 2018

    Notes
    -----
    .. versionadded:: 0.4
    """

    def __init__(self, target_domain, final_dispersion=1.0,
                 centered_data=False, metric="riemann"):
        """Init"""
        self.target_domain = target_domain
        self.final_dispersion = final_dispersion
        self.centered_data = centered_data
        self.metric = metric

    def fit(self, X, y_enc, sample_weight=None):
        """Fit TLStretch.

        Calculate the dispersion around the mean for each domain.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : TLStretch instance
            The TLStretch instance.
        """
        _, _, domains = decode_domains(X, y_enc)
        n_matrices, n_channels, _ = X.shape
        sample_weight = check_weights(sample_weight, n_matrices)

        self._means, self.dispersions_ = {}, {}
        for d in np.unique(domains):
            idx = domains == d
            sample_weight_d = check_weights(sample_weight[idx], np.sum(idx))
            if self.centered_data:
                self._means[d] = np.eye(n_channels)
            else:
                self._means[d] = mean_riemann(
                    X[idx], sample_weight=sample_weight_d
                )
            dist = distance(
                X[idx],
                self._means[d],
                metric=self.metric,
                squared=True,
            )
            self.dispersions_[d] = np.sum(sample_weight_d * np.squeeze(dist))

        return self

    def _center(self, X, mean):
        Mean_isqrt = invsqrtm(mean)
        return Mean_isqrt @ X @ Mean_isqrt

    def _uncenter(self, X, mean):
        Mean_sqrt = sqrtm(mean)
        return Mean_sqrt @ X @ Mean_sqrt

    def _strech(self, X, dispersion_in, dispersion_out):
        return powm(X, np.sqrt(dispersion_out / dispersion_in))

    def transform(self, X, y_enc=None):
        """Stretch the data points in the target domain.

        .. note::
           The stretching operation is properly defined only for the riemann
           metric.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X : ndarray, shape (n_matrices, n_classes)
            Set of SPD matrices with desired final dispersion.
        """

        if not self.centered_data:
            # center matrices to Identity
            X = self._center(X, self._means[self.target_domain])

        # stretch
        X_str = self._strech(
            X, self.dispersions_[self.target_domain], self.final_dispersion
        )

        if not self.centered_data:
            # re-center back to previous mean
            X_str = self._uncenter(X_str, self._means[self.target_domain])

        return X_str

    def fit_transform(self, X, y_enc, sample_weight=None):
        """Fit TLStretch and then transform data points.

        Calculate the dispersion around the mean for each domain and then
        stretch the data points to the desired final dispersion.

        .. note::
           This method is designed for using at training time. The output for
           .fit_transform() will be different than using .fit() and
           .transform() separately.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        X : ndarray, shape (n_matrices, n_classes)
            Set of SPD matrices with desired final dispersion.
        """

        # used during fit, in pipeline
        self.fit(X, y_enc, sample_weight)
        _, _, domains = decode_domains(X, y_enc)

        X_str = np.zeros_like(X)
        for d in np.unique(domains):
            idx = domains == d

            if not self.centered_data:
                # re-center matrices to Identity
                X[idx] = self._center(X[idx], self._means[d])

            # stretch
            X_str[idx] = self._strech(
                X[idx], self.dispersions_[d], self.final_dispersion
            )

            if not self.centered_data:
                # re-center back to previous mean
                X_str[idx] = self._uncenter(X_str[idx], self._means[d])

        return X_str


class TLRotate(BaseEstimator, TransformerMixin):
    """Rotate data for transfer learning.

    Rotate the data points from each source domain so to match its class means
    with those from the target domain. The loss function for this matching was
    first proposed in [1]_ and the optimization procedure for mininimizing it
    follows the presentation from [2]_.

    .. note::
       The data points from each domain must have been re-centered to the
       identity before calculating the rotation.

    .. note::
       Using .fit() and then .transform() will give different results than
       .fit_transform(). In fact, .fit_transform() should be applied on the
       training dataset (target and source) and .transform() on the test
       partition of the target dataset.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    weights : None | array, shape (n_classes,), default=None
        Weights to assign for each class. If None, then give the same weight
        for each class.
    metric : {"euclid", "riemann"}, default="euclid"
        Metric for the distance to minimize between class means.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        the rotation matrix for each source domain in parallel. If -1 all CPUs
        are used.

    Attributes
    ----------
    rotations_ : dict
        Dictionary with key=domain_name and value=domain_rotation_matrix.

    References
    ----------
    .. [1] `Riemannian Procrustes analysis: transfer learning for
        brain-computer interfaces
        <https://hal.archives-ouvertes.fr/hal-01971856>`_
        PLC Rodrigues et al, IEEE Transactions on Biomedical Engineering,
        vol. 66, no. 8, pp. 2390-2401, December, 2018
    .. [2] `An introduction to optimization on smooth manifolds
        <https://www.nicolasboumal.net/book/>`_
        N. Boumal. To appear with Cambridge University Press. June, 2022

    Notes
    -----
    .. versionadded:: 0.4
    """

    def __init__(self, target_domain, weights=None, metric='euclid', n_jobs=1):
        """Init"""
        self.target_domain = target_domain
        self.weights = weights
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y_enc, sample_weight=None):
        """Fit TLRotate.

        Calculate the rotations matrices to transform each source domain into
        the target domain.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : TLRotate instance
            The TLRotate instance.
        """

        _, _, domains = decode_domains(X, y_enc)
        n_matrices, _, _ = X.shape
        sample_weight = check_weights(sample_weight, n_matrices)

        idx = domains == self.target_domain
        X_target, y_target = X[idx], y_enc[idx]
        M_target = np.stack([
            mean_riemann(X_target[y_target == label],
                         sample_weight=sample_weight[idx][y_target == label])
            for label in np.unique(y_target)
        ])

        source_domains = np.unique(domains)
        source_domains = source_domains[source_domains != self.target_domain]
        rotations = Parallel(n_jobs=self.n_jobs)(
            delayed(_get_rotation_matrix)(
                np.stack([
                    mean_riemann(
                        X[domains == d][y_enc[domains == d] == label],
                        sample_weight=sample_weight[domains == d][
                            y_enc[domains == d] == label
                        ]
                    ) for label in np.unique(y_enc[domains == d])
                ]),
                M_target,
                self.weights,
                metric=self.metric,
            ) for d in source_domains
        )

        self.rotations_ = {}
        for d, rot in zip(source_domains, rotations):
            self.rotations_[d] = rot

        return self

    def transform(self, X, y_enc=None):
        """Rotate the data points in the target domain.

        The rotations are done from source to target, so in this step the data
        points suffer no transformation at all.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X : ndarray, shape (n_matrices, n_classes)
            Same set of SPD matrices as in the input.
        """

        # used during inference on target domain
        return X

    def fit_transform(self, X, y_enc, sample_weight=None):
        """Fit TLRotate and then transform data points.

        Calculate the rotation matrix for matching each source domain to the
        target domain.

        .. note::
           This method is designed for using at training time. The output for
           .fit_transform() will be different than using .fit() and
           .transform() separately.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        X : ndarray, shape (n_matrices, n_classes)
            Set of SPD matrices after rotation step.
        """

        # used during fit in pipeline, rotate each source domain
        self.fit(X, y_enc, sample_weight)
        _, _, domains = decode_domains(X, y_enc)

        X_rot = np.zeros_like(X)
        for d in np.unique(domains):
            idx = domains == d
            if d != self.target_domain:
                X_rot[idx] = self.rotations_[d] @ X[idx] @ self.rotations_[d].T
            else:
                X_rot[idx] = X[idx]
        return X_rot


class TLEstimator(BaseEstimator):
    """Transfer learning wrapper for estimators.

    This is a wrapper for any BaseEstimator (i.e. classifier or regressor) that
    converts extended labels used in Transfer Learning into the usual y array
    to train a classifier/regressor of choice.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    estimator : BaseEstimator
        The estimator to apply on matrices. It can be any regressor or
        classifier from pyRiemann.
    domain_weight : None | dict, default=None
        Weights to combine matrices from each domain to train the estimator.
        The dict contains key=domain_name and value=weight_to_assign.
        If None, it uses equal weights.

    Notes
    -----
    .. versionadded:: 0.4
    """

    def __init__(self, target_domain, estimator, domain_weight=None):
        """Init."""
        self.target_domain = target_domain
        self.domain_weight = domain_weight
        self.estimator = estimator

    def fit(self, X, y_enc):
        """Fit TLEstimator.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.

        Returns
        -------
        self : TLEstimator instance
            The TLEstimator instance.
        """
        if not (is_regressor(self.estimator) or is_classifier(self.estimator)):
            raise TypeError(
                'Estimator has to be either a classifier or a regressor.')

        X_dec, y_dec, domains = decode_domains(X, y_enc)

        if is_regressor(self.estimator):
            y_dec = y_dec.astype(float)

        if self.domain_weight is not None:
            w = np.zeros(len(X_dec))
            for d in np.unique(domains):
                w[domains == d] = self.domain_weight[d]
        else:
            w = None

        if isinstance(self.estimator, Pipeline):
            sample_weight = {}
            for step in self.estimator.steps:
                step_name = step[0]
                sample_weight[step_name + '__sample_weight'] = w
            self.estimator.fit(X_dec, y_dec, **sample_weight)
        else:
            self.estimator.fit(X_dec, y_dec, sample_weight=w)

        return self

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray, shape (n_matrices,)
            Predictions for each matrix according to the estimator.
        """
        return self.estimator.predict(X)


class TLClassifier(TLEstimator):
    """Transfer learning wrapper for classifiers.

    This is a wrapper for any classifier that converts extended labels used in
    Transfer Learning into the usual y array to train a classifier of choice.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    estimator : BaseClassifier
        The classifier to apply on matrices.
    domain_weight : None | dict, default=None
        Weights to combine matrices from each domain to train the classifier.
        The dict contains key=domain_name and value=weight_to_assign.
        If None, it uses equal weights.

    Notes
    -----
    .. versionadded:: 0.4
    """

    def fit(self, X, y_enc):
        """Fit TLClassifier.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.

        Returns
        -------
        self : TLClassifier instance
            The TLClassifier instance.
        """
        if not is_classifier(self.estimator):
            raise TypeError('Estimator has to be a classifier.')

        return super().fit(X, y_enc)

    def predict_proba(self, X):
        """Get the probability.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray, shape (n_matrices, n_classes)
            Predictions for each matrix.
        """
        return self.estimator.predict_proba(X)

    def score(self, X, y_enc):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Test set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended true labels for each matrix.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        _, y_true, _ = decode_domains(X, y_enc)
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)


class TLRegressor(TLEstimator):
    """Transfer learning wrapper for regressors.

    This is a wrapper for any regressor that converts extended labels used in
    Transfer Learning into the usual y array to train a regressor of choice.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    estimator : BaseRegressor
        The regressor to apply on matrices.
    domain_weight : None | dict, default=None
        Weights to combine matrices from each domain to train the regressor.
        The dict contains key=domain_name and value=weight_to_assign.
        If None, it uses equal weights.

    Notes
    -----
    .. versionadded:: 0.4
    """

    def fit(self, X, y_enc):
        """Fit TLRegressor.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.

        Returns
        -------
        self : TLRegressor instance
            The TLRegressor instance.
        """
        if not is_regressor(self.estimator):
            raise TypeError('Estimator has to be a regressor.')

        return super().fit(X, y_enc)

    def score(self, X, y_enc):
        """Return the coefficient of determination of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Test set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended true values for each matrix.

        Returns
        -------
        score : float
            R2 of self.predict(X) wrt. y.
        """
        _, y_true, _ = decode_domains(X, y_enc)
        y_pred = self.predict(X)
        return r2_score(y_true.astype(float), y_pred)


class MDWM(MDM):
    """Classification by Minimum Distance to Weighted Mean.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated, according to the chosen metric, as a weighted mean
    of SPD matrices from the source domain, combined with the class centroid of
    the target domain [1]_ [2]_.
    For classification, a given new matrix is attibuted to the class whose
    centroid is the nearest according to the chosen metric.

    Parameters
    ----------
    domain_tradeoff : float
        Coefficient in [0,1] controlling the transfer, ie the trade-off between
        source and target domains.
        At 0, there is no transfer, only matrices acquired from the source
        domain are used.
        At 1, this is a calibration-free system as no matrices are required
        from the source domain.
    target_domain : string
        Name of the target domain in extended labels.
    metric : string | dict, default="riemann"
        Metric used for mean estimation (for the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`) and
        for distance estimation
        (see :func:`pyriemann.utils.distance.distance`).
        The metric can be a dict with two keys, "mean" and "distance"
        in order to pass different metrics.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.
    covmeans_ : ndarray, shape (n_classes, n_channels, n_channels)
        Centroids for each class.

    See Also
    --------
    MDM

    References
    ----------
    .. [1] `Transfer learning for SSVEP-based BCI using Riemannian similarities
        between users
        <https://hal.archives-ouvertes.fr/hal-01911092/>`_
        E. Kalunga, S. Chevallier and Q. Barthelemy, in 26th European Signal
        Processing Conference (EUSIPCO), pp. 1685-1689. IEEE, 2018.
    .. [2] `Minimizing Subject-dependent Calibration for BCI with Riemannian
        Transfer Learning
        <https://hal.archives-ouvertes.fr/hal-03202360/>`_
        S. Khazem, S. Chevallier, Q. Barthelemy, K. Haroun and C. Nous, 10th
        International IEEE/EMBS Conference on Neural Engineering (NER), pp.
        523-526. IEEE, 2021.

    Notes
    -----
    .. versionadded:: 0.4
    """

    def __init__(
            self,
            domain_tradeoff,
            target_domain,
            metric="riemann",
            n_jobs=1):
        """Init."""
        self.domain_tradeoff = domain_tradeoff
        self.target_domain = target_domain
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y_enc, sample_weight=None):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices from source and target domain.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices_source,), \
                default=None
            Weights for each matrix from the source domains.
            If None, it uses equal weights.

        Returns
        -------
        self : MDWM instance
            The MDWM instance.
        """
        self.metric_mean, self.metric_dist = check_metric(self.metric)

        if not 0 <= self.domain_tradeoff <= 1:
            raise ValueError(
                "Value domain_tradeoff must be included in [0, 1] (Got %d)"
                % self.domain_tradeoff)

        X_dec, y_dec, domains = decode_domains(X, y_enc)
        X_src = X_dec[domains != self.target_domain]
        y_src = y_dec[domains != self.target_domain]
        X_tgt = X_dec[domains == self.target_domain]
        y_tgt = y_dec[domains == self.target_domain]

        self.classes_ = np.unique(y_src)

        if self.domain_tradeoff != 0 and set(y_tgt) != set(y_src):
            raise ValueError(
                "Classes in source domain must match classes in target domain."
                f"Classes in source are {self.classes_} while classes in "
                f"target are {np.unique(y_tgt)}"
            )

        sample_weight = check_weights(sample_weight, X_src.shape[0])

        self.source_means_ = np.stack(
            Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(
                    X_src[y_src == ll],
                    metric=self.metric_mean,
                    sample_weight=sample_weight[y_src == ll],
                ) for ll in self.classes_
            )
        )

        self.target_means_ = np.stack(
            Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(
                    X_tgt[y_tgt == ll],
                    metric=self.metric_mean,
                ) for ll in self.classes_
            )
        )

        self.covmeans_ = geodesic(
            self.source_means_,
            self.target_means_,
            self.domain_tradeoff,
            metric=self.metric_mean,
        )
        return self

    def score(self, X, y_enc, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.

        Returns
        -------
        score : float
            Mean accuracy of clf.predict(X) wrt. y_enc.
        """
        _, y_true, _ = decode_domains(X, y_enc)
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)
