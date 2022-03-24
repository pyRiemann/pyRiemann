"""Module for regression functions."""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVR as sklearnSVR
from sklearn.utils.extmath import softmax

from .utils.kernel import kernel
from .classification import MDM


class SVR(BaseEstimator, ClassifierMixin):
    """Regression by Riemannian Support Vector Machine.

    Support vector machine regression with precomputed Riemannian kernel matrix
    according to different metrics, extending the idea described in [1]_ to
    regression.

    Parameters
    ----------
    metric : {'riemann', 'euclid', 'logeuclid'}
        Metric for kernel matrix computation.
    Cref : None | ndarray, shape (n_channels, n_channels)
        Reference point for kernel matrix computation. If None, the mean of
        the training data according to the metric is used.
    **kwargs
        Keyword arguments passed to svc_func.

    Attributes
    ----------
    svc_ : svc_func instance
        Fitted SVC with precomputed Riemannian kernel matrix.
    data_ : ndarray, shape (n_matrices, n_channels, n_channels)
        Training data.

    Notes
    -----
    .. versionadded:: 0.2.8

    References
    ----------
    .. [1] A. Barachant, S. Bonnet, M. Congedo, and C. Jutten.
        Classification of covariance matrices using a Riemannian-based kernel
        for BCI applications". In: Neurocomputing 112 (July 2013), pp. 172-178.
    """

    def __init__(self, metric='riemann', Cref=None,
                 **kwargs):
        """Init."""
        self.svr_func = sklearnSVR
        self.Cref = Cref
        self.metric = metric
        self.svr_params = kwargs
        self.svr_ = self.svr_func(kernel='precomputed', **self.svr_params)

    def __setattr__(self, name, value):
        """Enable setting attributes for SVC subclass."""
        if 'svr_' in self.__dict__.keys():
            if name in self.svr_.__dict__.keys():
                self.svr_.set_params(**{name: value})
                return
        super().__setattr__(name, value)

    def fit(self, X, y):
        """Fit.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices, 1)
            labels corresponding to each matrix.

        Returns
        -------
        self : Riemannian SVC instance
            The SVC instance.
        """
        kernelmat = kernel(X, Cref=self.Cref, metric=self.metric)
        self.data_ = X
        self.svr_ = self.svr_.fit(kernelmat, y)

        return self

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices, 1)
            Predictions for each matrix according to the SVC.
        """
        test_kernel_mat = kernel(X,
                                 self.data_,
                                 Cref=self.Cref,
                                 metric=self.metric)
        return self.svr_.predict(test_kernel_mat)


class KNNRegression(MDM):
    """Regression by K-Nearest-Neighbors.

    Reression by nearest Neighbors. For each point of the test set, the
    pairwise distance to each element of the training set is estimated. The
    value is calculated according to the softmax average w.r.t. distance of
    the k nearest neighbors.

    Parameters
    ----------
    n_neighbors : int, (default: 5)
        Number of neighbors.
    metric : string | dict (default: 'riemann')
        The type of metric used for distance estimation.
        see `distance` for the list of supported metric.

    Attributes
    ----------
    classes_ : list
        list of classes.

    """

    def __init__(self, n_neighbors=5, metric='riemann'):
        """Init."""
        # store params for cloning purpose
        self.n_neighbors = n_neighbors
        MDM.__init__(self, metric=metric)

    def fit(self, X, y):
        """Fit (store the training data).

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray shape (n_matrices, 1)
            labels corresponding to each matrix.

        Returns
        -------
        self : NearestNeighbor instance
            The NearestNeighbor instance.
        """
        self.values_ = y
        self.covmeans_ = X

        return self

    def predict(self, covtest):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices, 1)
            Predictions for each matrix according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        dist_sorted = np.sort(dist)
        neighbors_values = self.values_[np.argsort(dist)]
        softmax_dist = softmax(-dist_sorted[:, 0:self.n_neighbors])
        knn_values = neighbors_values[:, 0:self.n_neighbors]
        out = np.sum(knn_values*softmax_dist, axis=1)
        return out
