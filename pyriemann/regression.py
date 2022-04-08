"""Module for regression functions."""

import numpy as np

from sklearn.svm import SVR as sklearnSVR
from sklearn.utils.extmath import softmax

from .utils.kernel import kernel
from .classification import MDM


class SVR(sklearnSVR):
    """Regression by Riemannian Support Vector Machine.

    Support vector machine regression with precomputed Riemannian kernel matrix
    according to different metrics, extending the idea described in [1]_ to
    regression.

    Parameters
    ----------
    metric : {'riemann', 'euclid', 'logeuclid'}, default: 'riemann'
        Metric for kernel matrix computation.
    Cref : None | ndarray, shape (n_channels, n_channels)
        Reference point for kernel matrix computation. If None, the mean of
        the training data according to the metric is used.
    tol : float, default: 1e-3
        Tolerance for stopping criterion.
    C : float, default: 1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
        The penalty is a squared l2 penalty.
    epsilon : float, default: 0.1
         Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
         within which no penalty is associated in the training loss function
         with points predicted within a distance epsilon from the actual
         value.
    shrinking : bool, default: True
        Whether to use the shrinking heuristic.
    cache_size : float, default: 200
        Specify the size of the kernel cache (in MB).
    verbose : bool, default: False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.
    max_iter : int, default: -1
        Hard limit on iterations within solver, or -1 for no limit.

    Attributes
    ----------
    data_ : ndarray, shape (n_matrices, n_channels, n_channels)
        If fitted, training data.

    Notes
    -----
    .. versionadded:: 0.2.8

    References
    ----------
    .. [1] A. Barachant, S. Bonnet, M. Congedo, and C. Jutten.
        Classification of covariance matrices using a Riemannian-based kernel
        for BCI applications". In: Neurocomputing 112 (July 2013), pp. 172-178.
    """

    def __init__(self,
                 *,
                 metric='riemann',
                 Cref=None,
                 tol=1e-3,
                 C=1.0,
                 epsilon=0.1,
                 shrinking=True,
                 cache_size=200,
                 verbose=False,
                 max_iter=-1,
                 ):
        """Init."""
        self.Cref = Cref
        self.metric = metric
        super().__init__(kernel='precomputed',
                         tol=tol,
                         C=C,
                         epsilon=epsilon,
                         shrinking=shrinking,
                         cache_size=cache_size,
                         verbose=verbose,
                         max_iter=max_iter)

    def fit(self, X, y, sample_weight=None):
        """Fit.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels corresponding to each matrix.
        sample_weight : ndarray, shape (n_matrices,), default: None
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        Returns
        -------
        self : Riemannian SVR instance
            The SVR instance.
        """
        kernelmat = kernel(X, Cref=self.Cref, metric=self.metric)
        self.data_ = X
        super().fit(kernelmat, y)

        return self

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray, shape (n_matrices,)
            Predictions for each matrix according to the SVR.
        """
        test_kernel_mat = kernel(X,
                                 self.data_,
                                 Cref=self.Cref,
                                 metric=self.metric)
        return super().predict(test_kernel_mat)


class KNearestNeighborRegressor(MDM):
    """Regression by K-Nearest-Neighbors.

    Regression by nearest Neighbors. For each point of the test set, the
    pairwise distance to each element of the training set is estimated. The
    value is calculated according to the softmax average w.r.t. distance of
    the k nearest neighbors.

    DISCLAIMER: This is an unpublished algorithm.

    Parameters
    ----------
    n_neighbors : int, default : 5
        Number of neighbors.
    metric : string | dict, default: 'riemann'
        The type of metric used for distance estimation.
        see `distance` for the list of supported metric.

    Attributes
    ----------
    values_ : ndarray, shape (n_matrices,)
        List of training data target values.
    covmeans_ : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices of training data.

    Notes
    -----
    .. versionadded:: 0.2.8

    """

    def __init__(self, n_neighbors=5, metric='riemann'):
        """Init."""
        # store params for cloning purpose
        self.n_neighbors = n_neighbors
        super().__init__(metric=metric)

    def fit(self, X, y):
        """Fit (store the training data).

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Target value for each matrix.

        Returns
        -------
        self : NearestNeighbor instance
            The NearestNeighbor instance.
        """
        self.values_ = y
        self.covmeans_ = X

        return self

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray, shape (n_matrices,)
            Predictions for each matrix according to the closest neighbors.
        """
        dist = self._predict_distances(X)
        idx = np.argsort(dist)
        dist_sorted = np.take_along_axis(dist, idx, axis=1)
        neighbors_values = self.values_[idx]
        softmax_dist = softmax(-dist_sorted[:, 0:self.n_neighbors]**2)
        knn_values = neighbors_values[:, 0:self.n_neighbors]
        out = np.sum(knn_values*softmax_dist, axis=1)
        return out
