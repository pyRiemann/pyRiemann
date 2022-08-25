import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import ShuffleSplit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score
from pyriemann.utils.mean import mean_covariance, mean_riemann
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.base import invsqrtm, powm, sqrtm
from pyriemann.utils.geodesic import geodesic
from pyriemann.classification import MDM

base_clf = MDM()


def encode_domains(X, y, domain):
    y_enc = []
    for n in range(len(y)):
        yn = y[n]
        dn = domain[n]
        yn_enc = str(yn) + '/' + dn
        y_enc.append(yn_enc)
    X_enc = X
    y_enc = np.array(y_enc)
    return X_enc, y_enc


def decode_domains(X_enc, y_enc):
    y = []
    domain = []
    for n in range(len(y_enc)):
        yn_enc = y_enc[n]
        try:
            yn = int(yn_enc.split('/')[0])
        except AttributeError:
            print(yn_enc)
            yn = 0
        y.append(yn)
        dn = yn_enc.split('/')[1]
        domain.append(dn)
    X = X_enc
    y = np.array(y)
    domain = np.array(domain)
    return X, y, domain


class TLSplitter():
    def __init__(self, target_domain, target_train_frac=0.80, n_splits=5):
        self.target_domain = target_domain
        self.target_train_frac = target_train_frac
        self.n_splits = n_splits

    def split(self, X, y):

        # decode the domains of the data points
        X, y, domain = decode_domains(X, y)

        # indentify the indices of the target dataset
        idx_source = np.where(domain != self.target_domain)[0]
        idx_target = np.where(domain == self.target_domain)[0]

        # index of training-split for the target data points
        ss_target = ShuffleSplit(
            n_splits=self.n_splits,
            train_size=self.target_train_frac).split(idx_target)
        for train_sub_idx_target, test_sub_idx_target in ss_target:
            train_idx = np.concatenate(
                [idx_source, idx_target[train_sub_idx_target]])
            test_idx = idx_target[test_sub_idx_target]
            yield train_idx, test_idx

    def get_n_splits(self, X, y):
        return self.n_splits


class TLDummy(BaseEstimator, TransformerMixin):
    """No transformation on data for transfer learning

    No transformation of the data points between the domains.
    This is what we call the Direct Center Transfer (DCT) method.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        return X

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)


class TLCenter(BaseEstimator, TransformerMixin):
    """Recenter data for transfer learning

    Recenter the data points from each domain to the Identity on manifold, ie
    make the geometric mean of the datasets become the identity.

    Parameters
    ----------
    target_domain : str
        Which domain to consider as target
    metric : str, default='riemann'
        The metric for mean, can be: 'ale', 'alm', 'euclid', 'harmonic',
        'identity', 'kullback_sym', 'logdet', 'logeuclid', 'riemann',
        'wasserstein', or a callable function.

    """

    def __init__(self, target_domain, metric='riemann'):
        """Init"""
        self.target_domain = target_domain
        self.metric = metric

    def fit(self, X, y):
        _, _, domains = decode_domains(X, y)
        self._Minvsqrt = {}
        for d in np.unique(domains):
            M = mean_covariance(X[domains == d], self.metric)
            self._Minvsqrt[d] = invsqrtm(M)
        return self

    def transform(self, X, y=None):
        # Used during inference, apply recenter from specified target domain.
        X_rct = np.zeros_like(X)
        Minvsqrt_target = self._Minvsqrt[self.target_domain]
        X_rct = Minvsqrt_target @ X @ Minvsqrt_target
        return X_rct

    def fit_transform(self, X, y):
        # used during fit, in pipeline
        self.fit(X, y)
        _, _, domains = decode_domains(X, y)
        X_rct = np.zeros_like(X)
        for d in np.unique(domains):
            idx = domains == d
            X_rct[idx] = self._Minvsqrt[d] @ X[idx] @ self._Minvsqrt[d].T
        return X_rct


class TLStretch(BaseEstimator, TransformerMixin):
    """Stretch data for transfer learning

    Change the dispersion of the datapoints around their geometric mean
    for each dataset so that they all have the same desired value.

    The dispersion is defined as the squared sum of the AIRM-induced distances
    of each point with respect to their geometric mean.

    Parameters
    ----------
    target_domain : str
        Which domain to consider as target
    dispersion : float, default=1.0
        Which target value for the dispersion of the data points
    centered_data : bool, default=False
        Whether the data has been re-centered to the Identity beforehand
    """

    def __init__(self, target_domain, final_dispersion=1.0,
                 centered_data=False):
        """Init"""
        self.target_domain = target_domain
        self.final_dispersion = final_dispersion
        self.centered_data = centered_data

    def fit(self, X, y):

        _, _, domains = decode_domains(X, y)
        m = X[0].shape[1]
        self._means = {}

        self._dispersions = {}
        for d in np.unique(domains):
            if self.centered_data:
                self._means[d] = np.eye(m)
            else:
                self._means[d] = mean_riemann(X[domains == d])
            disp_domain = np.sum([distance_riemann(Xi, self._means[d])**2
                                 for Xi in X[domains == d]])
            self._dispersions[d] = disp_domain

        return self

    def transform(self, X, y=None):
        # Used during inference, apply recenter from specified target domain.
        if self.centered_data:
            num = self.final_dispersion
            den = self._dispersions[self.target_domain]
            X_str = powm(X, np.sqrt(num / den))
        else:
            # first have to re-center the data to Identity
            Minvsqrt_target = invsqrtm(self._means[self.target_domain])
            X_rct = Minvsqrt_target @ X @ Minvsqrt_target
            # then do the re-stretching
            num = self.final_dispersion
            den = self._dispersions[self.target_domain]
            X_rct_str = powm(X_rct, np.sqrt(num / den))
            # and re-center back to previous mean
            Msqrt_target = sqrtm(self._means[self.target_domain])
            X_str = Msqrt_target @ X_rct_str @ Msqrt_target
        return X_str

    def fit_transform(self, X, y):
        # used during fit, in pipeline
        self.fit(X, y)
        _, _, domains = decode_domains(X, y)
        if self.centered_data:
            X_str = np.zeros_like(X)
            for d in np.unique(domains):
                idx = domains == d
                num = self.final_dispersion
                den = self._dispersions[d]
                X_str[idx] = powm(X[idx], np.sqrt(num / den))
        else:
            X_rct = np.zeros_like(X)
            X_rct_str = np.zeros_like(X)
            X_str = np.zeros_like(X)
            for d in np.unique(domains):
                idx = domains == d
                # first have to re-center the data to Identity
                Minvsqrt_target = invsqrtm(self._means[d])
                X_rct[idx] = Minvsqrt_target @ X[idx] @ Minvsqrt_target
                # then do the re-stretching
                num = self.final_dispersion
                den = self._dispersions[d]
                X_rct_str[idx] = powm(X_rct[idx], np.sqrt(num / den))
                # and re-center back to previous mean
                Msqrt_domain = sqrtm(self._means[d])
                X_str[idx] = Msqrt_domain @ X_rct_str[idx] @ Msqrt_domain
        return X_str


class TLClassifier(BaseEstimator, ClassifierMixin):
    """Classification with extended labels

    This is a wrapper that convert extended labels into class labels to
    train a classifier of choice

    Parameters
    ----------
    clf : pyriemann classifier, default=MDM()
        The classifier to apply on the manifold, with class label.
    """

    def __init__(self, target_domain, clf=base_clf):
        """Init."""
        self.target_domain = target_domain
        self.clf = clf

    def fit(self, X, y):
        """Fit DTClassifier.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Extended labels for each matrix.

        Returns
        -------
        self : TLClassifier instance
            The TLClassifier instance.
        """
        X_dec, y_dec, domains = decode_domains(X, y)

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

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.

        Returns
        -------
        score : float
            Mean accuracy of clf.predict(X) wrt. y.
        """
        _, y_true, _ = decode_domains(X, y)
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
    .. [1] E. Kalunga, S. Chevallier and Q. Barthelemy, "Transfer learning for
        SSVEP-based BCI using Riemannian similarities between users", in 26th
        European Signal Processing Conference (EUSIPCO), pp. 1685-1689. IEEE,
        2018.
    .. [2] S. Khazem, S. Chevallier, Q. Barthelemy, K. Haroun and C. Nous,
        "Minimizing Subject-dependent Calibration for BCI with Riemannian
        Transfer Learning", in 10th International IEEE/EMBS Conference on
        Neural Engineering (NER), pp. 523-526. IEEE, 2021.
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

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids.
        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices from source and target domain
        y : ndarray, shape (n_matrices,)
            Extended labels for each matrix
        sample_weight : None | ndarray, shape (n_matrices_source,), \
            default=None
            Weights for each matrix from the source domains.
            If None, it uses equal weights.
        Returns
        -------
        self : MDWM instance
            The MDWM instance.
        """

        if not 0 <= self.transfer_coef <= 1:
            raise ValueError(
                'Value transfer_coef must be included in [0, 1] (Got %d)'
                % self.transfer_coef)

        if isinstance(self.metric, str):
            self.metric_mean = self.metric
            self.metric_dist = self.metric
        elif isinstance(self.metric, dict):
            # check keys
            for key in ['mean', 'distance']:
                if key not in self.metric.keys():
                    raise KeyError('metric must contain "mean" and "distance"')

        X_dec, y_dec, domains = decode_domains(X, y)
        X_src = X_dec[domains != self.target_domain]
        y_src = y_dec[domains != self.target_domain]
        X_tgt = X_dec[domains == self.target_domain]
        y_tgt = y_dec[domains == self.target_domain]

        if self.transfer_coef != 0:
            if set(y_tgt) != set(y_src):
                raise ValueError(
                    f"classes in source domain must match classes in target \
                    domain. Classes in source are {np.unique(y_src)} while \
                    classes in target are {np.unique(y)}")

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

        self.covmeans_ = [geodesic(self.target_means_[i],
                                   self.source_means_[i],
                                   self.transfer_coef, self.metric)
                          for i in range(len(self.classes_))]
        return self

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.

        Returns
        -------
        score : float
            Mean accuracy of clf.predict(X) wrt. y.
        """
        _, y_true, _ = decode_domains(X, y)
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)
