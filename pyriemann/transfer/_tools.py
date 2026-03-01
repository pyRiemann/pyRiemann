import numpy as np


def encode_domains(X, y, domain):
    """Encode the domains of the data in the labels.

    We handle the possibility of having different domains for the datasets by
    extending the labels of the data and including this information to them.
    For instance, if we have a datum X with class ``left_hand`` on the
    ``domain_01`` then its extended label will be ``domain_01/left_hand``.
    Note that if the classes were integers at first, they will be converted to
    strings.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels) or \
            shape (n_vectors, n_ts)
        Set of SPD matrices or tangent vectors.
    y : ndarray, shape (n_matrices,) or shape (n_vectors,)
        Labels for each matrix or vector.
    domain : ndarray, shape (n_matrices,) or shape (n_vectors,)
        Domains for each matrix or vector.

    Returns
    -------
    X_enc : ndarray, shape (n_matrices, n_channels, n_channels) or \
            shape (n_vectors, n_ts)
        The same data given as input.
    y_enc : ndarray, shape (n_matrices,) or shape (n_vectors,)
        Extended labels for each matrix or vector.

    See Also
    --------
    decode_domains

    Notes
    -----
    .. versionadded:: 0.4
    """
    if len(y) != len(domain):
        raise ValueError("Input lengths don't match")

    y_enc = [str(d_) + "/" + str(y_) for (d_, y_) in zip(domain, y)]
    return X, np.array(y_enc)


def decode_domains(X_enc, y_enc):
    """Decode the domains of the data in the labels.

    We handle the possibility of having different domains for the datasets by
    encoding the domain information into the labels of the data.
    This method converts the data into its original form, with a separate data
    structure for labels and for domains.

    Parameters
    ----------
    X_enc : ndarray, shape (n_matrices, n_channels, n_channels) or \
            shape (n_vectors, n_ts)
        Set of SPD matrices or tangent vectors.
    y_enc : ndarray, shape (n_matrices,) or shape (n_vectors,)
        Extended labels for each matrix or vector.

    Returns
    -------
    X : ndarray, shape (n_matrices, n_channels, n_channels) or \
            shape (n_vectors, n_ts)
        The same data given as input.
    y : ndarray, shape (n_matrices,) or shape (n_vectors,)
        Labels for each matrix or vector.
    domain : ndarray, shape (n_matrices,) or shape (n_vectors,)
        Domains for each matrix or vector.

    See Also
    --------
    encode_domains

    Notes
    -----
    .. versionadded:: 0.4
    """
    y, domain = [], []
    for y_enc_ in y_enc:
        y_dec_ = y_enc_.split("/")
        domain.append(y_dec_[-2])
        y.append(y_dec_[-1])
    return X_enc, np.array(y), np.array(domain)


class TLSplitter():
    """Class for handling the cross-validation splits of multi-domain data.

    This is a wrapper to sklearn's cross-validation iterators [1]_ which
    ensures the handling of domain information with the data. In fact,
    the data from source domain is always fully available in the training
    partition whereas the random splits are done on the data from the
    target domain.

    Parameters
    ----------
    target_domain : str
        Domain considered as target.
    cv : None | BaseCrossValidator | BaseShuffleSplit, default=None
        An instance of a cross-validation iterator from sklearn.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators
    """  # noqa
    def __init__(self, target_domain, cv):
        self.target_domain = target_domain
        self.cv = cv

    def split(self, X, y):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.
        y : ndarray, shape (n_matrices,) or shape (n_vectors,)
            Extended labels for each matrix or vector.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        # decode the domains of the data
        X, y, domain = decode_domains(X, y)

        # identify the indices of the target dataset
        idx_source = np.where(domain != self.target_domain)[0]
        idx_target = np.where(domain == self.target_domain)[0]
        y_target = y[idx_target]

        # index of training-split for the target data
        ss_target = self.cv.split(idx_target, y_target)
        for train_sub_idx_target, test_sub_idx_target in ss_target:
            train_idx = np.concatenate(
                [idx_source, idx_target[train_sub_idx_target]]
            )
            test_idx = idx_target[test_sub_idx_target]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Ignored, exists for compatibility.
        y : object
            Ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Number of splitting iterations in the cross-validator.
        """
        return self.cv.n_splits
