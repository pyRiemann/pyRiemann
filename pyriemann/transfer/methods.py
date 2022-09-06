import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


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


# Define the new classes for Transfer Learning


class TLSplitter():
    def __init__(self,
                 target_domain,
                 target_train_frac=0.80,
                 n_splits=5,
                 random_state=None):

        self.target_domain = target_domain
        self.target_train_frac = target_train_frac
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y):

        # decode the domains of the data points
        X, y, domain = decode_domains(X, y)

        # indentify the indices of the target dataset
        idx_source = np.where(domain != self.target_domain)[0]
        idx_target = np.where(domain == self.target_domain)[0]
        y_target = y[idx_target]

        # index of training-split for the target data points
        ss_target = StratifiedShuffleSplit(
            n_splits=self.n_splits,
            train_size=self.target_train_frac,
            random_state=self.random_state).split(idx_target, y_target)
        for train_sub_idx_target, test_sub_idx_target in ss_target:
            train_idx = np.concatenate(
                [idx_source, idx_target[train_sub_idx_target]])
            test_idx = idx_target[test_sub_idx_target]
            yield train_idx, test_idx

    def get_n_splits(self, X, y):
        return self.n_splits
