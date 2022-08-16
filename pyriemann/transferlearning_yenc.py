import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score
from pyriemann.utils.mean import mean_riemann
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
    def __init__(self, target_domain, n_splits=5):
        self.n_splits = n_splits
        self.target_domain = target_domain

    def split(self, X, y):
        # decode the domains of the data points
        X, y, domain = decode_domains(X, y)

        # indentify the indices of the target dataset
        idx_source = np.where(domain != self.target_domain)[0]
        idx_target = np.where(domain == self.target_domain)[0]

        # index of training-split for the target data points
        kf_target = KFold(n_splits=self.n_splits).split(idx_target)
        for train_sub_idx_target, test_sub_idx_target in kf_target:
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
    infer_domain : str
        If known indicate target domain for inference, else last one is used
    """

    def __init__(self, infer_domain=None):
        """Init"""
        self.infer_domain = infer_domain

    def fit(self, X, y):
        _, _, domains = decode_domains(X, y)
        self._Minvsqrt = {}
        for d in np.unique(domains):
            M = mean_riemann(X[domains == d])
            self._Minvsqrt[d] = invsqrtm(M)
        if self.infer_domain is None:
            self.infer_domain = np.unique(domains)[-1]
        return self

    def transform(self, X, y=None):
        # Used during inference, apply recenter from specified target domain.
        # If no domain specified for inference, last one is used.
        X_rct = np.zeros_like(X)
        Minvsqrt_domain = self._Minvsqrt[self.infer_domain]
        X_rct = np.stack(
                [Minvsqrt_domain @ Xi @ Minvsqrt_domain.T for Xi in X])
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

    Convert extended labels into class label to train a classifier

    Parameters
    ----------
    clf : pyriemann classifier, default=MDM()
        The classifier to apply on the manifold, with class label.
    """

    def __init__(self, clf=base_clf):
        """Init."""
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
        _, y_dec, _ = decode_domains(X, y)
        self.clf.fit(X, y_dec)
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
