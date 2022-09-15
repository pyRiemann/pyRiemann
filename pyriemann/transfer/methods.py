import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def encode_domains(X, y, domain):
    r"""Encode the domains of the data points in the labels

    We handle the possibility of having different domains for the datasets by
    extending the labels of the data points and including this information to
    them. For instance, if we have a datapoint x with class `left_hand` on the
    `domain_01` then its extended label will be `left_hand/domain_01`. Note
    that if the classes were integers at first, they will be converted to
    strings.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    y : ndarray, shape (n_matrices,)
        Labels for each matrix.
    domain : ndarray, shape (n_matrices,)
        Which domain each data point belongs to.

    Returns
    -------
    X_enc : ndarray, shape (n_matrices, n_channels, n_channels)
        The same set of SPD matrices given as input.
    y_enc : ndarray, shape (n_matrices,)
        Extended labels for each data point.

    Notes
    -----
    .. versionadded:: 0.3.1
    """
    y_enc = [str(y[n]) + '/' + domain[n] for n in range(len(y))]
    return X, np.array(y_enc)


def decode_domains(X_enc, y_enc):
    r"""Decode the domains of the data points in the labels

    We handle the possibility of having different domains for the datasets by
    encoding the domain information into the labels of the data points. This
    method converts the data into its original form, with a separate data
    structure for labels and for domains.

    Parameters
    ----------
    X_enc : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    y_enc : ndarray, shape (n_matrices,)
        Extended labels for each data point.

    Returns
    -------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    y : ndarray, shape (n_matrices,)
        Labels for each matrix.
    domain : ndarray, shape (n_matrices,)
        Which domain each data point belongs to.

    Notes
    -----
    .. versionadded:: 0.3.1
    """
    y, domain = [], []
    for n in range(len(y_enc)):
        yn_enc = y_enc[n]
        try:
            yn = int(yn_enc.split('/')[0])
        except AttributeError:
            print(yn_enc)
            yn = 0
        y.append(yn)
        domain.append(yn_enc.split('/')[1])
    return X_enc, np.array(y), np.array(domain)


class TLSplitter():
    r"""Class for handling the cross-validation splits of multi-domain data

    This is a wrapper to sklearn's StratifiedShuffleSplit which ensures the
    handling of domain information with the data points. In fact, the data
    coming from source domain is always fully available and the random splits
    are carried out on the data points from the target domain.

    Parameters
    ----------
    target_domain : str
        Which domain should be considered as target.
    target_train_frac : float
        The fraction of data points in the target domain that should be in the
        training split during cross-validation.
    n_splits : int, default=5
        How many splits to consider in the cross-validation scheme.
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling and
        splitting the data. Pass an int for reproducible output across multiple
        function calls.

    Notes
    -----
    .. versionadded:: 0.3.1
    """
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
