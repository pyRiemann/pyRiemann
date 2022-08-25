
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm


class TLSplitter():
    def __init__(self, target_train_frac=0.80, n_splits=5):
        self.target_train_frac = target_train_frac
        self.n_splits = n_splits

    def split(self, X, y, meta):

        # index of all source data points
        idx_source = np.where(~meta['target'])[0]

        # index of training-split for the target data points
        ss_target = ShuffleSplit(
            n_splits=self.n_splits,
            train_size=self.target_train_frac).split(X[meta['target']])
        for train_idx_target, test_idx_target in ss_target:
            train_idx_target = list(
                meta[meta['target']].index[train_idx_target])
            train_idx = np.concatenate([idx_source, train_idx_target])
            test_idx_target = list(meta[meta['target']].index[test_idx_target])
            test_idx = test_idx_target
            yield train_idx, test_idx

    def get_n_splits(self, X, y, meta):
        return self.n_splits


class DCT(BaseEstimator, TransformerMixin):
    '''
    No transformation of the data points between the domains.
    This is what we call the direct (DCT) method.
    '''

    def __init__(self):
        pass

    def fit(self, X, y, meta):
        return self

    def transform(self, X, y=None, meta=None):
        return X

    def fit_transform(self, X, y, meta):
        return self.fit(X, y, meta).transform(X, y, meta)


class RCT(BaseEstimator, TransformerMixin):
    '''
    Re-center (RCT) the data points from each domain to the Identity.
    '''

    def __init__(self):
        pass

    def fit(self, X, y, meta):
        self._Minvsqrt = {}
        for domain in np.unique(meta['domain']):
            domain_idx = meta['domain'] == domain
            M = mean_riemann(X[domain_idx])
            self._Minvsqrt[domain] = invsqrtm(M)
        return self

    def transform(self, X, y=None, meta=None):
        X_rct = np.zeros(X.shape)
        for d in np.unique(meta['domain']):
            idx = meta['domain'] == d
            X_rct[idx] = self._Minvsqrt[d] @ X[idx] @ self._Minvsqrt[d].T
        return X_rct

    def fit_transform(self, X, y, meta):
        return self.fit(X, y, meta).transform(X, y, meta)


class TLPipeline(BaseEstimator, ClassifierMixin):
    '''

    Pipeline for carrying out transfer learning in pyRiemann.

    When fitting the model we do:
        (1) Fit and transform the data points from each domain
        (2) Select which samples from each domain should be in the training set
        (3) Fit the classifier on the training set

    When predicting labels we do:
        (1) Transform the data points from each domain
        (2) Use trained classifier to predict labels of testing set

    The are three modes for training the pipeline:
        (1) train clf only on source domain + training partition from target
        (2) train clf only on source domain
        (3) train clf only on training partition from target
    these different choices yield very different results in the classification.

    '''

    def __init__(self, transformer, clf):
        self.transformer = transformer
        self.clf = clf

    def fit(self, X, y, meta):
        X_transf = self.transformer.fit_transform(X, y, meta)
        select = np.where(y != -1)[0]
        X_train = X_transf[select]
        y_train = y[select]
        self.clf.fit(X_train, y_train)
        return self

    def predict(self, X, meta):
        X_transf = self.transformer.transform(X, meta=meta)
        X_test = X_transf
        return self.clf.predict(X_test)

    def score(self, X, y, meta):
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X, meta)
        y_true = y
        return accuracy_score(y_true, y_pred)
