import functools

import numpy as np
from sklearn.base import ClassifierMixin, ClusterMixin, TransformerMixin

from ..geometry.mean import gmean


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


class SpdSvmMixin():
    """SvmMixin for SPD matrices"""

    def _set_cref(self, X):
        if self.Cref is None:
            self.Cref_ = gmean(X, metric=self.metric)
        elif callable(self.Cref):
            self.Cref_ = self.Cref(X)
        elif isinstance(self.Cref, np.ndarray):
            self.Cref_ = self.Cref
        else:
            raise TypeError(
                f"Cref must be np.ndarray, callable or None, is {self.Cref}."
            )

    def _set_kernel(self, kernel):
        if callable(self.kernel_fct):
            self.kernel = functools.partial(
                self.kernel_fct,
                Cref=self.Cref_,
                metric=self.metric
            )
        elif self.kernel_fct is None or (isinstance(self.kernel_fct, str) and
                                         self.kernel_fct == "precomputed"):
            self.kernel = functools.partial(
                kernel,
                Cref=self.Cref_,
                metric=self.metric
            )
        else:
            raise TypeError(
                "kernel_fct must be None, 'precomputed' or callable, is "
                f"{self.kernel}."
            )
