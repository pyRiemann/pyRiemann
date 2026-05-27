from sklearn.base import ClassifierMixin, ClusterMixin, TransformerMixin


class SpdTransfMixin(TransformerMixin):
    """TransformerMixin for SPD matrices"""

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None | ndarray, shape (n_matrices,), default=None
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_centroids)
            Distance to each centroid.
        """
        if sample_weight is None:
            return self.fit(X, y).transform(X)
        else:
            return self.fit(X, y, sample_weight=sample_weight).transform(X)


class SpdClassifMixin(ClassifierMixin):
    """ClassifierMixin for SPD matrices"""

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Test set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            True labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix.

        Returns
        -------
        score : float
            Mean accuracy of clf.predict(X) wrt. y.
        """
        return super().score(X, y, sample_weight)


class SpdClustMixin(ClusterMixin):
    """ClusterMixin for SPD matrices"""

    def fit_predict(self, X, y=None):
        """Fit and predict in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Prediction for each matrix according to the closest cluster.
        """
        return self.fit(X, y).predict(X)
