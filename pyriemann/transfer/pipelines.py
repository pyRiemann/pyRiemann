import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score
from ..utils.mean import mean_covariance, mean_riemann
from ..utils.distance import distance
from ..utils.base import invsqrtm, powm, sqrtm
from ..utils.geodesic import geodesic
from ._rotate import get_rotation_matrix
from ..classification import MDM, _check_metric
from ..preprocessing import Whitening
from .methods import decode_domains


class TLDummy(BaseEstimator, TransformerMixin):
    """No transformation on data for transfer learning.

    No transformation of the data points between the domains.
    This is what we call the Direct Center Transfer (DCT) method.
    """

    def __init__(self):
        pass

    def fit(self, X, y_enc):
        return self

    def transform(self, X, y_enc=None):
        return X

    def fit_transform(self, X, y_enc):
        return self.fit(X, y_enc).transform(X, y_enc)


class TLCenter(BaseEstimator, TransformerMixin):
    """Recenter data for transfer learning.

    Recenter the data points from each domain to the Identity on manifold, ie
    make the mean of the datasets become the identity. This operation
    corresponds to a whitening step if the SPD matrices represent the spatial
    covariance matrices of multivariate signals.

    Important: note that using .fit() and then .transform() will give different
    results than .fit_transform(). In fact, .fit_transform() should be applied
    on the training dataset (target and source) and .transform() on the test
    partition of the target dataset.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    metric : str, default='riemann'
        The metric for mean, can be: 'ale', 'alm', 'euclid', 'harmonic',
        'identity', 'kullback_sym', 'logdet', 'logeuclid', 'riemann',
        'wasserstein', or a callable function.

    Attributes
    ----------
    recenter_ : dict
        Dictionary with key=domain_name and value=domain_mean
    """

    def __init__(self, target_domain, metric='riemann'):
        """Init"""
        self.target_domain = target_domain
        self.metric = metric

    def fit(self, X, y_enc):
        _, _, domains = decode_domains(X, y_enc)
        self.recenter_ = {}
        for d in np.unique(domains):
            idx = domains == d
            self.recenter_[d] = Whitening(metric=self.metric).fit(X[idx])
        return self

    def transform(self, X, y_enc=None):
        # Used during inference, apply recenter from specified target domain.
        return self.recenter_[self.target_domain].transform(X)

    def inverse_transform(self, X, y_enc=None):
        return self.recenter_[self.target_domain].inverse_transform(X)

    def fit_transform(self, X, y_enc):
        # used during fit, in pipeline
        self.fit(X, y_enc)
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

    Important: note that using .fit() and then .transform() will give different
    results than .fit_transform(). In fact, .fit_transform() should be applied
    on the training dataset (target and source) and .transform() on the test
    partition of the target dataset.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    dispersion : float, default=1.0
        Target value for the dispersion of the data points.
    centered_data : bool, default=False
        Whether the data has been re-centered to the Identity beforehand.
    metric : str, default='riemann'
        The metric for calculating the dispersion can be: 'ale', 'alm',
        'euclid', 'harmonic', 'identity', 'kullback_sym', 'logdet',
        'logeuclid', 'riemann', 'wasserstein', or a callable function.

    Attributes
    ----------
    dispersions_ : dict
        Dictionary with key=domain_name and value=domain_dispersion
    """

    def __init__(self, target_domain, final_dispersion=1.0,
                 centered_data=False, metric='riemann'):
        """Init"""
        self.target_domain = target_domain
        self.final_dispersion = final_dispersion
        self.centered_data = centered_data
        self.metric = metric

    def fit(self, X, y_enc):
        _, _, domains = decode_domains(X, y_enc)
        n_dim = X[0].shape[1]
        self._means = {}
        self.dispersions_ = {}
        for d in np.unique(domains):
            if self.centered_data:
                self._means[d] = np.eye(n_dim)
            else:
                self._means[d] = mean_riemann(X[domains == d])
            disp_domain = np.sum([distance(
                Xi, self._means[d], metric=self.metric)**2
                                 for Xi in X[domains == d]])
            self.dispersions_[d] = disp_domain

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
        """Used during inference, apply recenter from specified target domain.
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

    def fit_transform(self, X, y_enc):
        # used during fit, in pipeline
        self.fit(X, y_enc)
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

    Important 1: the data points from each domain must have been re-centered
    to the identity before calculating the rotation.

    Important 2: note that using .fit() and then .transform() will give
    different results than .fit_transform(). In fact, .fit_transform() should
    be applied on the training dataset (target and source) and .transform() on
    the test partition of the target dataset.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    weights : None | array, shape (n_classes,), default=None
        Weights to assign for each class. If None, then give the same weight
        for each class.
    metric : {'euclid', 'riemann'}, default='euclid'
        Metric for the distance to minimize between class means. Options are
        either the Euclidean ('euclid') or Riemannian ('riemann') distance.

    Attributes
    ----------
    rotations_ : dict
        Dictionary with key=domain_name and value=domain_rotation_matrix

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
    """

    def __init__(self, target_domain, weights=None, metric='euclid'):
        """Init"""
        self.target_domain = target_domain
        self.weights = weights
        self.metric = metric

    def fit(self, X, y_enc):

        _, _, domains = decode_domains(X, y_enc)

        idx = domains == self.target_domain
        X_target, y_target = X[idx], y_enc[idx]
        M_target = [mean_riemann(X_target[y_target == label])
                    for label in np.unique(y_target)]

        self.rotations_ = {}
        for d in np.unique(domains):
            if d != self.target_domain:
                idx = domains == d
                X_source, y_source = X[idx], y_enc[idx]
                M_source = [mean_riemann(X_source[y_source == label])
                            for label in np.unique(y_source)]
                self.rotations_[d] = get_rotation_matrix(
                    M_source,
                    M_target,
                    self.weights,
                    metric=self.metric)

        return self

    def transform(self, X, y_enc=None):
        # used during inference on target domain
        return X

    def fit_transform(self, X, y_enc):
        # used during fit in pipeline, rotate each source domain
        self.fit(X, y_enc)
        _, _, domains = decode_domains(X, y_enc)
        X_rot = np.zeros_like(X)
        for d in np.unique(domains):
            idx = domains == d
            if d != self.target_domain:
                X_rot[idx] = self.rotations_[d] @ X[idx] @ self.rotations_[d].T
            else:
                X_rot[idx] = X[idx]
        return X_rot


class TLClassifier(BaseEstimator, ClassifierMixin):
    """Classification with extended labels.

    This is a wrapper that converts extended labels into class labels to
    train a classifier of choice.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    clf : BaseClassifier, default=MDM()
        The classifier to apply on the data points.
    """

    def __init__(self, target_domain, clf=MDM()):
        """Init."""
        self.target_domain = target_domain
        self.clf = clf

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
        X_dec, y_dec, _ = decode_domains(X, y_enc)

        select = np.where(y_dec != -1)[0]
        X_train = X_dec[select]
        y_train = y_dec[select]

        self.clf.fit(X_train, y_train)
        return self

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the classifier
        """
        return self.clf.predict(X)

    def predict_proba(self, X):
        """Get the probability.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of ifloat, shape (n_matrices, n_classes)
            Predictions for each matrix according to the classifier
        """
        return self.clf.predict_proba(X)

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


class TLMDM(MDM):
    """Classification by Minimum Distance to Weighted Mean.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated, according to the chosen metric, as a weighted mean
    of SPD matrices from the source domain, combined with
    the class centroid of the target domain [1]_ [2]_.
    For classification, a given new matrix is attibuted to the class whose
    centroid is the nearest according to the chosen metric.

    Parameters
    ----------
    transfer_coef : float
        Transfer coefficient in [0,1], controlling the trade-off between
        source and target data. At 0, there is no transfer, only the data
        acquired from the source are used. At 1, this is a calibration-free
        system as no data are required from the source.
    target_domain : string
        Name of the target domain in extended labels
    metric : string | dict, default='riemann'
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
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
        Class centroids, estimated after fit.
    classes_ : list
        List of classes, obtained after fit

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
    """

    def __init__(
            self,
            transfer_coef,
            target_domain,
            metric='riemann',
            n_jobs=1):
        """Init."""
        self.transfer_coef = transfer_coef
        self.target_domain = target_domain
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y_enc, sample_weight=None):
        """Fit (estimates) the centroids.
        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices from source and target domain
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix
        sample_weight : None | ndarray, shape (n_matrices_source,), \
            default=None
            Weights for each matrix from the source domains.
            If None, it uses equal weights.
        Returns
        -------
        self : TLMDM instance
            The TLMDM instance.
        """
        self.metric_mean, self.metric_dist = _check_metric(self.metric)
        if not 0 <= self.transfer_coef <= 1:
            raise ValueError(
                'Value transfer_coef must be included in [0, 1] (Got %d)'
                % self.transfer_coef)

        X_dec, y_dec, domains = decode_domains(X, y_enc)
        X_src = X_dec[domains != self.target_domain]
        y_src = y_dec[domains != self.target_domain]
        X_tgt = X_dec[domains == self.target_domain]
        y_tgt = y_dec[domains == self.target_domain]

        if self.transfer_coef != 0:
            if set(y_tgt) != set(y_src):
                raise ValueError(
                    f"classes in source domain must match classes in target \
                    domain. Classes in source are {np.unique(y_src)} while \
                    classes in target are {np.unique(y_enc)}")

        if sample_weight is not None:
            if (sample_weight.shape != (X_src.shape[0], 1)) and \
                                (sample_weight.shape != (X_src.shape[0],)):
                raise ValueError("Parameter sample_weight should either be \
                    None or an ndarray shape (n_matrices, 1)")

        # if X.shape[0] != y.shape[0]:
        #     raise ValueError("X and y must be for the same number of \
        #         matrices i.e. n_matrices")

        if sample_weight is None:
            sample_weight = np.ones(X_src.shape[0])

        # if not (X_source.shape[0] == y_source.shape[0]):
        #     raise ValueError("X and y must be for the same number of \
        #         matrices i.e. n_matrices")

        self.classes_ = np.unique(y_src)

        self.target_means_ = Parallel(n_jobs=self.n_jobs)(
            delayed(mean_covariance)(
                X_tgt[y_tgt == ll],
                metric=self.metric_mean)
            for ll in self.classes_)
        self.source_means_ = Parallel(n_jobs=self.n_jobs)(
            delayed(mean_covariance)(
                X_src[y_src == ll],
                metric=self.metric_mean,
                sample_weight=sample_weight[y_src == ll])
            for ll in self.classes_)

        self.covmeans_ = geodesic(
            self.target_means_,
            self.source_means_,
            self.transfer_coef,
            self.metric
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
