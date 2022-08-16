import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.base import invsqrtm
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

    Recenter the data points from each domain to the Identity on manifold.
    This method is called Re-Center Transform (RCT).

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
        _, yd, domains = decode_domains(X, y)
        X_rct = np.zeros_like(X)
        for d in np.unique(domains):
            idx = domains == d
            X_rct[idx] = self._Minvsqrt[d] @ X[idx] @ self._Minvsqrt[d].T
        return X_rct


class TLClassifier(BaseEstimator, ClassifierMixin):
    """Classification with extended labels

    Convert extended labels into class label to train a classifier and choose
    how to join the data from source and target domains as training and testing
    partitions

    Parameters
    ----------
    clf : pyriemann classifier, default=MDM()
        The classifier to apply on the manifold, with class label.
    training_mode : int
        The are three modes for training the pipeline:
        (1) train clf only on source domain + training partition from target
        (2) train clf only on source domain
        (3) train clf only on training partition from target
        these different choices yield very different results in the
        classification.
    """

    def __init__(self, target_domain, clf=base_clf, training_mode=1):
        """Init."""
        self.target_domain = target_domain
        self.clf = clf
        self.training_mode = training_mode

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

        if self.training_mode == 1:
            X_train = X_dec
            y_train = y_dec
        elif self.training_mode == 2:
            X_train = X_dec[domains != self.target_domain]
            y_train = y_dec[domains != self.target_domain]
        elif self.training_mode == 3:
            X_train = X_dec[domains == self.target_domain]
            y_train = y_dec[domains == self.target_domain]

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
