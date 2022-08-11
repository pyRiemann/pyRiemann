
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm
import pandas as pd


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
        yn = float(yn_enc.split('/')[0])
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

    def get_n_splits(self, X, y, meta):
        return self.n_splits


class DCT(BaseEstimator, TransformerMixin):
    '''
    No transformation of the data points between the domains.
    This is what we call the direct (DCT) method.
    '''

    def __init__(self, target_domain, training_mode):
        self.target_domain = target_domain
        self.training_mode = training_mode

    def fit(self, X_enc, y_enc):
        return self

    def transform(self, X_enc, y_enc=None):
        X, y, domain = decode_domains(X_enc, y_enc)
        return X

    def fit_transform(self, X_enc, y_enc):
        return self.fit(X_enc, y_enc).transform(X_enc, y_enc)
