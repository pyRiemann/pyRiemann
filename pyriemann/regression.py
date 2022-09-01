"""Module for regression functions."""
import functools

import numpy as np

from sklearn.svm import SVR as sklearnSVR
from sklearn.utils.extmath import softmax

from .utils.kernel import kernel
from .classification import MDM, _check_metric
from .utils.mean import mean_covariance


class SVR(sklearnSVR):
    """Regression by support-vector machine.

    Support-vector machine (SVM) regression with precomputed Riemannian kernel
    matrix according to different metrics, extending the idea described in [1]_
    to regression.

    Parameters
    ----------
    metric : {'riemann', 'euclid', 'logeuclid'}, default='riemann'
        Metric for kernel matrix computation.
    Cref : None | ndarray, shape (n_channels, n_channels)
        Reference point for kernel matrix computation. If None, the mean of
        the training data according to the metric is used.
    kernel_fct : 'precomputed' | callable
        If 'precomputed', the kernel matrix for datasets X and Y is estimated
        according to `pyriemann.utils.kernel(X, Y, Cref, metric)`.
        If callable, the callable is passed as the kernel parameter to
        `sklearn.svm.SVC()` [2]_. The callable has to be of the form
        `kernel(X, Y, Cref, metric)`.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
        The penalty is a squared l2 penalty.
    epsilon : float, default=0.1
         Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
         within which no penalty is associated in the training loss function
         with points predicted within a distance epsilon from the actual
         value.
    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).
    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.
    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    Attributes
    ----------
    data_ : ndarray, shape (n_matrices, n_channels, n_channels)
        If fitted, training data.

    Notes
    -----
    .. versionadded:: 0.3

    References
    ----------
    .. [1] A. Barachant, S. Bonnet, M. Congedo, and C. Jutten.
        Classification of covariance matrices using a Riemannian-based kernel
        for BCI applications". In: Neurocomputing 112 (July 2013), pp. 172-178.
    .. [2]
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    """

    def __init__(self,
                 *,
                 metric='riemann',
                 kernel_fct=None,
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
        self.Cref_ = None
        self.kernel_fct = kernel_fct
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
            Target values for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. Rescale C per matrix. Higher weights
            force the classifier to put more emphasis on these matrices.
            If None, it uses equal weights.

        Returns
        -------
        self : SVR instance
            The SVR instance.
        """
        self._set_cref(X)
        self._set_kernel()
        super().fit(X, y)
        return self

    def _set_cref(self, X):
        if self.Cref is None:
            self.Cref_ = mean_covariance(X, metric=self.metric)
        elif callable(self.Cref):
            self.Cref_ = self.Cref(X)
        elif isinstance(self.Cref, np.ndarray):
            self.Cref_ = self.Cref
        else:
            raise TypeError(f'Cref has to be np.ndarray, callable or None. But'
                            f' has type {type(self.Cref)}.')

    def _set_kernel(self):
        if callable(self.kernel_fct):
            self.kernel = functools.partial(self.kernel_fct,
                                            Cref=self.Cref_,
                                            metric=self.metric)

        elif self.kernel_fct is None:
            self.kernel = functools.partial(kernel,
                                            Cref=self.Cref_,
                                            metric=self.metric)
        else:
            raise TypeError(f"kernel must be 'precomputed' or callable, is "
                            f"{self.kernel}.")


class KNearestNeighborRegressor(MDM):
    """Regression by k-nearest-neighbors.

    Regression by k-nearest neighbors (k-NN). For each point of the test set,
    the pairwise distance to each element of the training set is estimated. The
    value is calculated according to the softmax average w.r.t. distance of
    the k-nearest neighbors.

    DISCLAIMER: This is an unpublished algorithm.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors.
    metric : string | dict, default='riemann'
        The type of metric used for distance estimation.
        See `distance` for the list of supported metric.

    Attributes
    ----------
    values_ : ndarray, shape (n_matrices,)
        List of training data target values.
    covmeans_ : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices of training dataset.

    Notes
    -----
    .. versionadded:: 0.3

    """

    def __init__(self, n_neighbors=5, metric='riemann'):
        """Init."""
        self.n_neighbors = n_neighbors
        super().__init__(metric=metric)

    def fit(self, X, y):
        """Fit (store the training data).

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Target values for each matrix.

        Returns
        -------
        self : KNearestNeighborRegressor instance
            The KNearestNeighborRegressor instance.
        """
        self.metric_mean, self.metric_dist = _check_metric(self.metric)
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
