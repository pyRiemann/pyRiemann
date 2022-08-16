
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm


class TLSplitter():
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y, meta):

        # index of all source data points
        idx_source = np.where(~meta['target'])[0]

        # index of training-split for the target data points
        kf_target = KFold(n_splits=self.n_splits).split(X[meta['target']])
        for train_idx_target, test_idx_target in kf_target:
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
        for domain in np.unique(meta['domain']):
            idx = meta['domain'] == domain
            Minvsqrt_domain = self._Minvsqrt[domain]
            X_rct[idx] = Minvsqrt_domain @ X[idx] @ Minvsqrt_domain.T
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
        (1) train clf only on source domain
        (2) train clf only on source domain + training partition from target
        (3) train clf only on training partition from target
    these different choices yield very different results in the classification.

    '''

    def __init__(self, transformer, clf, training_mode=1):
        self.transformer = transformer
        self.clf = clf
        self._training_mode = training_mode

    def __select(self, X, y, meta):

        if self._training_mode == 1:
            # take only data from training source
            select_source = ~meta['target']
            X_select = X[select_source]
            y_select = y[select_source]

        elif self._training_mode == 2:
            # take data from training source and target
            X_select = X
            y_select = y

        elif self._training_mode == 3:
            # take data only from training target
            select_target = meta['target']
            X_select = X[select_target]
            y_select = y[select_target]

        return X_select, y_select

    def fit(self, X, y, meta):
        X_transf = self.transformer.fit_transform(X, y, meta)
        X_select, y_select = self.__select(X_transf, y, meta)
        X_train, y_train = X_select, y_select
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
