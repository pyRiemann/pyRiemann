"""Module for classification function."""
import functools

from joblib import Parallel, delayed
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.svm import SVC as sklearnSVC
from sklearn.utils.extmath import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .utils import deprecated
from .utils.kernel import kernel
from .utils.mean import mean_covariance
from .utils.distance import distance
from .utils.utils import check_metric
from .tangentspace import FGDA, TangentSpace


def _mode_1d(X):
    vals, counts = np.unique(X, return_counts=True)
    mode = vals[counts.argmax()]
    return mode


def _mode_2d(X, axis=1):
    mode = np.apply_along_axis(_mode_1d, axis, X)
    return mode


class SpdClassifMixin(ClassifierMixin):

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


class MDM(SpdClassifMixin, TransformerMixin, BaseEstimator):
    r"""Classification by Minimum Distance to Mean.

    For each of the given classes :math:`k = 1, \ldots, K`, a centroid
    :math:`\mathbf{M}^k` is estimated according to the chosen metric.

    Then, for each new matrix :math:`\mathbf{X}`, the class is affected
    according to the nearest centroid [1]_:

    .. math::
        \hat{k} = \arg \min_{k} d (\mathbf{X}, \mathbf{M}^k)

    Parameters
    ----------
    metric : string | dict, default="riemann"
        Metric used for mean estimation (for the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`) and
        for distance estimation
        (see :func:`pyriemann.utils.distance.distance`).
        The metric can be a dict with two keys, "mean" and "distance"
        in order to pass different metrics.
        Typical usecase is to pass "logeuclid" metric for the "mean" in order
        to boost the computional speed, and "riemann" for the "distance" in
        order to keep the good sensitivity for the classification.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.
    covmeans_ : ndarray, shape (n_classes, n_channels, n_channels)
        Centroids for each class.

    See Also
    --------
    Kmeans
    FgMDM
    KNearestNeighbor

    References
    ----------
    .. [1] `Multiclass Brain-Computer Interface Classification by Riemannian
        Geometry
        <https://hal.archives-ouvertes.fr/hal-00681328>`_
        A. Barachant, S. Bonnet, M. Congedo, and C. Jutten. IEEE Transactions
        on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.
    .. [2] `Riemannian geometry applied to BCI classification
        <https://hal.archives-ouvertes.fr/hal-00602700/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. 9th International
        Conference Latent Variable Analysis and Signal Separation
        (LVA/ICA 2010), LNCS vol. 6365, 2010, p. 629-636.
    """

    def __init__(self, metric="riemann", n_jobs=1):
        """Init."""
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : MDM instance
            The MDM instance.
        """
        self.metric_mean, self.metric_dist = check_metric(self.metric)
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        if self.n_jobs == 1:
            self.covmeans_ = [
                mean_covariance(
                    X[y == c],
                    metric=self.metric_mean,
                    sample_weight=sample_weight[y == c]
                ) for c in self.classes_
            ]
        else:
            self.covmeans_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(
                    X[y == c],
                    metric=self.metric_mean,
                    sample_weight=sample_weight[y == c]
                ) for c in self.classes_
            )

        self.covmeans_ = np.stack(self.covmeans_, axis=0)

        return self

    def _predict_distances(self, X):
        """Helper to predict the distance. Equivalent to transform."""

        if self.n_jobs == 1:
            dist = [
                distance(X, covmean, self.metric_dist)
                for covmean in self.covmeans_
            ]
        else:
            dist = Parallel(n_jobs=self.n_jobs)(
                delayed(distance)(
                    X, covmean, self.metric_dist
                ) for covmean in self.covmeans_
            )

        dist = np.concatenate(dist, axis=1)
        return dist

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the nearest centroid.
        """
        dist = self._predict_distances(X)
        return self.classes_[dist.argmin(axis=1)]

    def transform(self, X):
        """Get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            Distance to each centroid according to the metric.
        """
        return self._predict_distances(X)

    @deprecated(
        "fit_predict() is deprecated and will be removed in 0.10.0; "
        "please use fit().predict()."
    )
    def fit_predict(self, X, y, sample_weight=None):
        return self.fit(X, y, sample_weight=sample_weight).predict(X)

    def fit_transform(self, X, y, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            Distance to each centroid according to the metric.
        """
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def predict_proba(self, X):
        """Predict proba using softmax of negative squared distances.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_matrices, n_classes)
            Probabilities for each class.
        """
        return softmax(-self._predict_distances(X) ** 2)


class FgMDM(SpdClassifMixin, TransformerMixin, BaseEstimator):
    """Classification by Minimum Distance to Mean with geodesic filtering.

    Apply geodesic filtering described in [1]_, and classify using MDM.
    The geodesic filtering is achieved in tangent space with a Linear
    Discriminant Analysis, then data are projected back to the manifold and
    classifier with a regular MDM.
    This is basically a pipeline of FGDA and MDM.

    Parameters
    ----------
    metric : string | dict, default="riemann"
        Metric used for reference matrix estimation (for the list of supported
        metrics, see :func:`pyriemann.utils.mean.mean_covariance`),
        for distance estimation (see :func:`pyriemann.utils.distance.distance`)
        and for tangent space map
        (see :func:`pyriemann.utils.tangent_space.tangent_space`).
        The metric can be a dict with three keys, "mean", "dist" and "map" in
        order to pass different metrics.
    tsupdate : bool, default=False
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]_. This is not compatible with
        online implementation. Performance are better when the number of
        matrices for prediction is higher.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.

    See Also
    --------
    MDM
    FGDA
    TangentSpace

    References
    ----------
    .. [1] `Riemannian geometry applied to BCI classification
        <https://hal.archives-ouvertes.fr/hal-00602700/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. 9th International
        Conference Latent Variable Analysis and Signal Separation
        (LVA/ICA 2010), LNCS vol. 6365, 2010, p. 629-636.
    .. [2] `Classification of covariance matrices using a Riemannian-based
        kernel for BCI applications
        <https://hal.archives-ouvertes.fr/hal-00820475/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. Neurocomputing,
        Elsevier, 2013, 112, pp.172-178.
    """

    def __init__(self, metric="riemann", tsupdate=False, n_jobs=1):
        """Init."""
        self.metric = metric
        self.n_jobs = n_jobs
        self.tsupdate = tsupdate

    def fit(self, X, y, sample_weight=None):
        """Fit FgMDM.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : FgMDM instance
            The FgMDM instance.
        """
        self._mdm = MDM(metric=self.metric, n_jobs=self.n_jobs)
        self._fgda = FGDA(metric=self.metric, tsupdate=self.tsupdate)
        cov = self._fgda.fit_transform(X, y, sample_weight=sample_weight)
        self._mdm.fit(cov, y, sample_weight=sample_weight)
        self.classes_ = self._mdm.classes_
        return self

    def predict(self, X):
        """Get the predictions after FGDA filtering.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the nearest centroid.
        """
        cov = self._fgda.transform(X)
        return self._mdm.predict(cov)

    def predict_proba(self, X):
        """Predict proba using softmax after FGDA filtering.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_matrices, n_classes)
            The softmax probabilities for each class.
        """
        cov = self._fgda.transform(X)
        return self._mdm.predict_proba(cov)

    def transform(self, X):
        """Get the distance to each centroid after FGDA filtering.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_cluster)
            Distance to each centroid according to the metric.
        """
        cov = self._fgda.transform(X)
        return self._mdm.transform(cov)

    def fit_transform(self, X, y, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_cluster)
            Distance to each centroid according to the metric.
        """
        return self.fit(X, y, sample_weight=sample_weight).transform(X)


class TSClassifier(SpdClassifMixin, BaseEstimator):
    """Classification in the tangent space.

    Project SPD matrices in the tangent space and apply a classifier.
    This is a simple helper to pipeline the tangent space projection and
    a classifier.

    Parameters
    ----------
    metric : string | dict, default="riemann"
        The type of metric used
        for reference matrix estimation (for the list of supported metrics
        see :func:`pyriemann.utils.mean.mean_covariance`) and
        for tangent space map
        (see :func:`pyriemann.utils.tangent_space.tangent_space`).
        The metric can be a dict with two keys, "mean" and "map"
        in order to pass different metrics.
    tsupdate : bool, default=False
        Activate tangent space update for covariate shift correction between
        training and test, as described in [1]_. This is not compatible with
        online implementation. Performance are better when the number of
        matrices for prediction is higher.
    clf : sklearn classifier, default=LogisticRegression()
        The classifier to apply in the tangent space.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.

    See Also
    --------
    TangentSpace

    Notes
    -----
    .. versionadded:: 0.2.4

    References
    ----------
    .. [1] `Classification of covariance matrices using a Riemannian-based
        kernel for BCI applications
        <https://hal.archives-ouvertes.fr/hal-00820475/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. Neurocomputing,
        Elsevier, 2013, 112, pp.172-178.
    """

    def __init__(self, metric="riemann", tsupdate=False,
                 clf=LogisticRegression()):
        """Init."""
        self.metric = metric
        self.tsupdate = tsupdate
        self.clf = clf

    def fit(self, X, y, sample_weight=None):
        """Fit TsClassifier.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : TSClassifier instance
            The TSClassifier instance.
        """
        if not isinstance(self.clf, ClassifierMixin):
            raise TypeError("clf must be a ClassifierMixin")
        self.classes_ = np.unique(y)

        ts = TangentSpace(metric=self.metric, tsupdate=self.tsupdate)
        self._pipe = make_pipeline(ts, self.clf)
        sample_weight_dict = {}
        for step in self._pipe.steps:
            step_name = step[0]
            sample_weight_dict[step_name + "__sample_weight"] = sample_weight
        self._pipe.fit(X, y, **sample_weight_dict)
        return self

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix.
        """
        return self._pipe.predict(X)

    def predict_proba(self, X):
        """Get the probability.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of ifloat, shape (n_matrices, n_classes)
            Predictions for each matrix.
        """
        return self._pipe.predict_proba(X)


@deprecated(
    "TSclassifier is deprecated and will be removed in 0.10.0; "
    "please use TSClassifier."
)
class TSclassifier(TSClassifier):
    pass


class KNearestNeighbor(MDM):
    """Classification by k-nearest neighbors.

    Classification by k-nearest neighbors (k-NN). For each matrix of the test
    set, the pairwise distance to each element of the training set is
    estimated. The class is affected according to the majority class of the
    k-nearest neighbors.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors.
    metric : string | dict, default="riemann"
        Metric used for means estimation (for the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`) and
        for distance estimation
        (see :func:`pyriemann.utils.distance.distance`).
        The metric can be a dict with two keys, "mean" and "distance"
        in order to pass different metrics.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the distance to the training set in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.
    covmeans_ : ndarray, shape (n_matrices, n_channels, n_channels)
        Matrices of training set.
    classmeans_ : ndarray, shape (n_matrices,)
        Labels of training set.

    See Also
    --------
    Kmeans
    MDM

    """

    def __init__(self, n_neighbors=5, metric="riemann", n_jobs=1):
        """Init."""
        super().__init__(metric=metric, n_jobs=n_jobs)
        self.n_neighbors = n_neighbors

    def fit(self, X, y, sample_weight=None):
        """Fit (store the training data).

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : NearestNeighbor instance
            The NearestNeighbor instance.
        """
        self.metric_mean, self.metric_dist = check_metric(self.metric)
        self.covmeans_ = X
        self.classmeans_ = y
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the nearest neighbors.
        """
        dist = self._predict_distances(X)
        neighbors_classes = self.classmeans_[np.argsort(dist)]
        pred = _mode_2d(neighbors_classes[:, 0:self.n_neighbors], axis=1)
        return pred

    def predict_proba(self, X):
        """Predict proba using softmax.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_matrices, n_classes)
            Probabilities for each class.
        """
        n_matrices, _, _ = X.shape

        dist = self._predict_distances(X)
        idx = np.argsort(dist)
        dist_sorted = np.take_along_axis(dist, idx, axis=1)
        neighbors_classes = self.classmeans_[idx]
        probas = softmax(-dist_sorted[:, 0:self.n_neighbors] ** 2)

        prob = np.zeros((n_matrices, len(self.classes_)))
        for m in range(n_matrices):
            for ic, c in enumerate(self.classes_):
                prob[m, ic] = np.sum(
                    probas[m, neighbors_classes[m, 0:self.n_neighbors] == c]
                )

        return prob


class SVC(sklearnSVC):
    """Classification by support-vector machine.

    Support-vector machine (SVM) classification with precomputed Riemannian
    kernel matrix according to different metrics as described in [1]_.

    Parameters
    ----------
    metric : string, default="riemann"
        Metric for kernel matrix computation. For the list of supported metrics
        see :func:`pyriemann.utils.kernel.kernel`.
    Cref : None | callable | ndarray, shape (n_channels, n_channels), \
            default=None
        Reference matrix for kernel matrix computation.
        If None, the mean of the training matrices according to the metric is
        used.
        If callable, the function is called on the training matrices to
        calculate Cref.
    kernel_fct : None | "precomputed" | callable, default=None
        If None or "precomputed", the kernel matrix for datasets X and Y is
        estimated according to `pyriemann.utils.kernel(X, Y, Cref, metric)`.
        If callable, the callable is passed as the kernel parameter to
        `sklearn.svm.SVC()` [2]_. The callable has to be of the form
        `kernel(X, Y, Cref, metric)`.
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.
    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
    probability : bool, default=False
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, will slow down that method as it internally uses
        5-fold cross-validation, and `predict_proba` may be inconsistent with
        `predict`.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).
    class_weight : None | dict | "balanced", default=None
        Set the parameter C of class i to class_weight[i]*C for SVC. If not
        given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_matrices / (n_classes * np.bincount(y))``.
    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.
    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.
    decision_function_shape : {"ovo", "ovr"}, default="ovr"
        Whether to return a one-vs-rest ("ovr") decision function of shape
        (n_matrices, n_classes) as all other classifiers, or the original
        one-vs-one ("ovo") decision function of libsvm which has shape
        (n_matrices, n_classes * (n_classes - 1) / 2). However, note that
        internally, one-vs-one ("ovo") is always used as a multi-class strategy
        to train models; an ovr matrix is only constructed from the ovo matrix.
        The parameter is ignored for binary classification.
    break_ties : bool, default=False
        If true, ``decision_function_shape="ovr"``, and number of classes > 2,
        `predict` will break ties according to the confidence values of
        `decision_function`; otherwise the first class among the tied
        classes is returned. Please note that breaking ties comes at a
        relatively high computational cost compared to a simple predict.
    random_state : None | int | RandomState instance, default=None
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when `probability` is False.
        Pass an int for reproducible output across multiple function calls.

    Notes
    -----
    .. versionadded:: 0.3

    References
    ----------
    .. [1] `Classification of covariance matrices using a Riemannian-based
        kernel for BCI applications
        <https://hal.archives-ouvertes.fr/hal-00820475/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. Neurocomputing,
        Elsevier, 2013, 112, pp.172-178.
    .. [2]
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    def __init__(
        self,
        *,
        metric="riemann",
        kernel_fct=None,
        Cref=None,
        C=1.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None
    ):
        """Init."""
        self.Cref = Cref
        self.metric = metric
        self.Cref_ = None
        self.kernel_fct = kernel_fct
        super().__init__(
            kernel="precomputed",
            C=C,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state
        )

    def fit(self, X, y, sample_weight=None):
        """Fit.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. Rescale C per matrix. Higher weights
            force the classifier to put more emphasis on these matrices.
            If None, it uses equal weights.

        Returns
        -------
        self : SVC instance
            The SVC instance.
        """
        self._set_cref(X)
        self._set_kernel()
        super().fit(X, y, sample_weight)
        return self

    def _set_cref(self, X):
        if self.Cref is None:
            self.Cref_ = mean_covariance(X, metric=self.metric)
        elif callable(self.Cref):
            self.Cref_ = self.Cref(X)
        elif isinstance(self.Cref, np.ndarray):
            self.Cref_ = self.Cref
        else:
            raise TypeError(f"Cref must be np.ndarray, callable or None, is "
                            f"{self.Cref}.")

    def _set_kernel(self):
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


class MeanField(SpdClassifMixin, TransformerMixin, BaseEstimator):
    """Classification by Minimum Distance to Mean Field.

    Classification by Minimum Distance to Mean Field [1]_, defining several
    power means for each class.

    Parameters
    ----------
    power_list : list of float, default=[-1,0,+1]
        Exponents of power means.
    method_label : {"sum_means", "inf_means"}, default="sum_means"
        Method to combine labels:

        * sum_means: it assigns the matrix to the class whom the sum of
          distances to means of the field is the lowest;
        * inf_means: it assigns the matrix to the class of the nearest mean
          of the field.
    metric : string, default="riemann"
        Metric used for distance estimation during prediction.
        For the list of supported metrics,
        see :func:`pyriemann.utils.distance.distance`.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.
    covmeans_ : dict of ``n_powers`` dicts of ``n_classes`` ndarrays of shape \
            (n_channels, n_channels)
        Centroids for each power and each class.

    See Also
    --------
    MDM

    Notes
    -----
    .. versionadded:: 0.3

    References
    ----------
    .. [1] `The Riemannian Minimum Distance to Means Field Classifier
        <https://hal.archives-ouvertes.fr/hal-02315131>`_
        M Congedo, PLC Rodrigues, C Jutten. BCI 2019 - 8th International
        Brain-Computer Interface Conference, Sep 2019, Graz, Austria.
    """

    def __init__(
        self,
        power_list=[-1, 0, 1],
        method_label="sum_means",
        metric="riemann",
        n_jobs=1,
    ):
        """Init."""
        self.power_list = power_list
        self.method_label = method_label
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : MeanField instance
            The MeanField instance.
        """
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        self.covmeans_ = {}
        for p in self.power_list:
            means_p = {}
            for c in self.classes_:
                means_p[c] = mean_covariance(
                    X[y == c],
                    p,
                    metric="power",
                    sample_weight=sample_weight[y == c],
                )
            self.covmeans_[p] = means_p

        return self

    def _get_label(self, x):
        m = np.zeros((len(self.power_list), len(self.classes_)))
        for ip, p in enumerate(self.power_list):
            for ic, c in enumerate(self.classes_):
                m[ip, ic] = distance(
                    x,
                    self.covmeans_[p][c],
                    metric=self.metric,
                    squared=True,
                )

        if self.method_label == "sum_means":
            ipmin = np.argmin(np.sum(m, axis=1))
        elif self.method_label == "inf_means":
            ipmin = np.where(m == np.min(m))[0][0]
        else:
            raise TypeError("method_label must be sum_means or inf_means")

        y = self.classes_[np.argmin(m[ipmin])]
        return y

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the nearest means field.
        """
        pred = Parallel(n_jobs=self.n_jobs)(
            delayed(self._get_label)(x) for x in X
        )
        return np.array(pred)

    def _predict_distances(self, X):
        """Helper to predict the distance. Equivalent to transform."""

        dist = []
        for x in X:
            m = {}
            for p in self.power_list:
                m[p] = []
                for c in self.classes_:
                    m[p].append(
                        distance(
                            x,
                            self.covmeans_[p][c],
                            metric=self.metric,
                        )
                    )
            pmin = min(m.items(), key=lambda x: np.sum(x[1]))[0]
            dist.append(np.array(m[pmin]))

        return np.stack(dist)

    def transform(self, X):
        """Get the distance to each means field.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            Distance to each means field according to the metric.
        """
        return self._predict_distances(X)

    @deprecated(
        "fit_predict() is deprecated and will be removed in 0.10.0; "
        "please use fit().predict()."
    )
    def fit_predict(self, X, y, sample_weight=None):
        return self.fit(X, y, sample_weight=sample_weight).predict(X)

    def fit_transform(self, X, y, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            Distance to each means field according to the metric.
        """
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def predict_proba(self, X):
        """Predict proba using softmax of negative squared distances.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_matrices, n_classes)
            Probabilities for each class.
        """
        return softmax(-self._predict_distances(X) ** 2)


def class_distinctiveness(X, y, exponent=1, metric="riemann",
                          return_num_denom=False):
    r"""Measure class distinctiveness between classes of SPD/HPD matrices.

    For two class problem, the class distinctiveness between class :math:`K_1`
    and :math:`K_2` on the manifold of SPD/HPD matrices is quantified as [1]_:

    .. math::
        \mathrm{classDis}(K_1, K_2, p) =
        \frac{d \left( \mathbf{M}_{K_1}, \mathbf{M}_{K_2} \right)^p}
        {\frac{1}{2} \left( \sigma_{K_1}^p + \sigma_{K_2}^p \right)}

    where :math:`\mathbf{M}_K` is the center of class :math:`K`, ie the mean of
    matrices from class :math:`K`; and
    :math:`\sigma_K` is the class dispersion, ie the mean of distances between
    matrices from class :math:`K` and their center of class
    :math:`\mathbf{M}_K`:

    .. math::
        \sigma_K^p = \frac{1}{m} \sum_{i=1}^m d
        \left(X_i, \mathbf{M}_K \right)^p

    and :math:`p` is the exponentiation of the distance.

    For more than two classes, it is quantified as:

    .. math::
        \mathrm{classDis} \left( \left\{K_{j} \right\}_{j=1}^c, p \right) =
        \frac{\sum_{j=1}^c d\left(\mathbf{M}_{K_{j}},\bar{\mathbf{M}}\right)^p}
        {\sum_{j=1}^c \sigma_{K_{j}}^p}

    where :math:`\bar{\mathbf{M}}` is the mean of centers of class of all
    :math:`c` classes.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD/HPD matrices.
    y : ndarray, shape (n_matrices,)
        Labels for each matrix.
    exponent : int, default=1
        Parameter for exponentiation of distances, corresponding to p in the
        above equations:

        - exponent = 1 gives the formula originally defined in [1]_;
        - exponent = 2 gives the Fisher criterion generalized on the manifold,
          ie the ratio of the variance between the classes to the variance
          within the classes.
    metric : string | dict, default="riemann"
        Metric used for mean estimation (for the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`) and
        for distance estimation
        (see :func:`pyriemann.utils.distance.distance`).
        The metric can be a dict with two keys, "mean" and "distance"
        in order to pass different metrics.
    return_num_denom : bool, default=False
        Whether to return numerator and denominator of class_dis.

    Returns
    -------
    class_dis : float
        Class distinctiveness value.
    num : float
        Numerator value of class_dis. Returned only if return_num_denom is
        True.
    denom : float
        Denominator value of class_dis. Returned only if return_num_denom is
        True.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] `Defining and quantifying users’ mental imagery-based
       BCI skills: a first step
       <https://hal.archives-ouvertes.fr/hal-01846434/>`_
       F. Lotte, and C. Jeunet. Journal of neural engineering,
       15(4), 046030, 2018.
    """

    metric_mean, metric_dist = check_metric(metric)
    classes = np.unique(y)
    if len(classes) <= 1:
        raise ValueError("y must contain at least two classes")

    means = np.array([
        mean_covariance(X[y == c], metric=metric_mean) for c in classes
    ])

    if len(classes) == 2:
        num = distance(means[0], means[1], metric=metric_dist) ** exponent
        denom = 0.5 * _get_within(X, y, means, classes, exponent, metric_dist)

    else:
        mean_all = mean_covariance(means, metric=metric_mean)
        dists_between = [
            distance(m, mean_all, metric=metric_dist) ** exponent
            for m in means
        ]
        num = np.sum(dists_between)
        denom = _get_within(X, y, means, classes, exponent, metric_dist)

    class_dis = num / denom

    if return_num_denom:
        return class_dis, num, denom
    else:
        return class_dis


def _get_within(X, y, means, classes, exponent, metric):
    """Private function to compute within dispersion."""
    sigmas = []
    for ic, c in enumerate(classes):
        dists_within = [
            distance(x, means[ic], metric=metric) ** exponent
            for x in X[y == c]
        ]
        sigmas.append(np.mean(dists_within))
    sum_sigmas = np.sum(sigmas)
    return sum_sigmas

#temporal: imports MeanField_V2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from .utils.mean import mean_logeuclid
from .utils.mean import mean_power
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class MeanField(SpdClassifMixin, TransformerMixin, BaseEstimator):
    """Classification by Minimum Distance to Mean Field.

    Classification by Minimum Distance to Mean Field [1]_, defining several
    power means for each class.

    Parameters
    ----------
    power_list : list of float, default=[-1,0,+1]
        Exponents of power means.
    method_label : {"sum_means", "inf_means"}, default="sum_means"
        Method to combine labels:

        * sum_means: it assigns the matrix to the class whom the sum of
          distances to means of the field is the lowest;
        * inf_means: it assigns the matrix to the class of the nearest mean
          of the field.
    metric : string, default="riemann"
        Metric used for distance estimation during prediction.
        For the list of supported metrics,
        see :func:`pyriemann.utils.distance.distance`.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.
    covmeans_ : dict of ``n_powers`` dicts of ``n_classes`` ndarrays of shape \
            (n_channels, n_channels)
        Centroids for each power and each class.

    See Also
    --------
    MDM

    Notes
    -----
    .. versionadded:: 0.3

    References
    ----------
    .. [1] `The Riemannian Minimum Distance to Means Field Classifier
        <https://hal.archives-ouvertes.fr/hal-02315131>`_
        M Congedo, PLC Rodrigues, C Jutten. BCI 2019 - 8th International
        Brain-Computer Interface Conference, Sep 2019, Graz, Austria.
    """

    def __init__(
        self,
        power_list=[-1, 0, 1],
        method_label="sum_means",
        metric="riemann",
        n_jobs=1,
    ):
        """Init."""
        self.power_list = power_list
        self.method_label = method_label
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : MeanField instance
            The MeanField instance.
        """
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        self.covmeans_ = {}
        for p in self.power_list:
            means_p = {}
            for c in self.classes_:
                means_p[c] = mean_covariance(
                    X[y == c],
                    p,
                    metric="power",
                    sample_weight=sample_weight[y == c],
                )
            self.covmeans_[p] = means_p

        return self

    def _get_label(self, x):
        m = np.zeros((len(self.power_list), len(self.classes_)))
        for ip, p in enumerate(self.power_list):
            for ic, c in enumerate(self.classes_):
                m[ip, ic] = distance(
                    x,
                    self.covmeans_[p][c],
                    metric=self.metric,
                    squared=True,
                )

        if self.method_label == "sum_means":
            ipmin = np.argmin(np.sum(m, axis=1))
        elif self.method_label == "inf_means":
            ipmin = np.where(m == np.min(m))[0][0]
        else:
            raise TypeError("method_label must be sum_means or inf_means")

        y = self.classes_[np.argmin(m[ipmin])]
        return y

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the nearest means field.
        """
        pred = Parallel(n_jobs=self.n_jobs)(
            delayed(self._get_label)(x) for x in X
        )
        return np.array(pred)

    def _predict_distances(self, X):
        """Helper to predict the distance. Equivalent to transform."""

        dist = []
        for x in X:
            m = {}
            for p in self.power_list:
                m[p] = []
                for c in self.classes_:
                    m[p].append(
                        distance(
                            x,
                            self.covmeans_[p][c],
                            metric=self.metric,
                        )
                    )
            pmin = min(m.items(), key=lambda x: np.sum(x[1]))[0]
            dist.append(np.array(m[pmin]))

        return np.stack(dist)

    def transform(self, X):
        """Get the distance to each means field.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            Distance to each means field according to the metric.
        """
        return self._predict_distances(X)

    @deprecated(
        "fit_predict() is deprecated and will be removed in 0.10.0; "
        "please use fit().predict()."
    )
    def fit_predict(self, X, y, sample_weight=None):
        return self.fit(X, y, sample_weight=sample_weight).predict(X)

    def fit_transform(self, X, y, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            Distance to each means field according to the metric.
        """
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def predict_proba(self, X):
        """Predict proba using softmax of negative squared distances.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_matrices, n_classes)
            Probabilities for each class.
        """
        return softmax(-self._predict_distances(X) ** 2)


def class_distinctiveness(X, y, exponent=1, metric="riemann",
                          return_num_denom=False):
    r"""Measure class distinctiveness between classes of SPD/HPD matrices.

    For two class problem, the class distinctiveness between class :math:`K_1`
    and :math:`K_2` on the manifold of SPD/HPD matrices is quantified as [1]_:

    .. math::
        \mathrm{classDis}(K_1, K_2, p) =
        \frac{d \left( \mathbf{M}_{K_1}, \mathbf{M}_{K_2} \right)^p}
        {\frac{1}{2} \left( \sigma_{K_1}^p + \sigma_{K_2}^p \right)}

    where :math:`\mathbf{M}_K` is the center of class :math:`K`, ie the mean of
    matrices from class :math:`K`; and
    :math:`\sigma_K` is the class dispersion, ie the mean of distances between
    matrices from class :math:`K` and their center of class
    :math:`\mathbf{M}_K`:

    .. math::
        \sigma_K^p = \frac{1}{m} \sum_{i=1}^m d
        \left(X_i, \mathbf{M}_K \right)^p

    and :math:`p` is the exponentiation of the distance.

    For more than two classes, it is quantified as:

    .. math::
        \mathrm{classDis} \left( \left\{K_{j} \right\}_{j=1}^c, p \right) =
        \frac{\sum_{j=1}^c d\left(\mathbf{M}_{K_{j}},\bar{\mathbf{M}}\right)^p}
        {\sum_{j=1}^c \sigma_{K_{j}}^p}

    where :math:`\bar{\mathbf{M}}` is the mean of centers of class of all
    :math:`c` classes.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD/HPD matrices.
    y : ndarray, shape (n_matrices,)
        Labels for each matrix.
    exponent : int, default=1
        Parameter for exponentiation of distances, corresponding to p in the
        above equations:

        - exponent = 1 gives the formula originally defined in [1]_;
        - exponent = 2 gives the Fisher criterion generalized on the manifold,
          ie the ratio of the variance between the classes to the variance
          within the classes.
    metric : string | dict, default="riemann"
        Metric used for mean estimation (for the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`) and
        for distance estimation
        (see :func:`pyriemann.utils.distance.distance`).
        The metric can be a dict with two keys, "mean" and "distance"
        in order to pass different metrics.
    return_num_denom : bool, default=False
        Whether to return numerator and denominator of class_dis.

    Returns
    -------
    class_dis : float
        Class distinctiveness value.
    num : float
        Numerator value of class_dis. Returned only if return_num_denom is
        True.
    denom : float
        Denominator value of class_dis. Returned only if return_num_denom is
        True.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] `Defining and quantifying users’ mental imagery-based
       BCI skills: a first step
       <https://hal.archives-ouvertes.fr/hal-01846434/>`_
       F. Lotte, and C. Jeunet. Journal of neural engineering,
       15(4), 046030, 2018.
    """

    metric_mean, metric_dist = check_metric(metric)
    classes = np.unique(y)
    if len(classes) <= 1:
        raise ValueError("y must contain at least two classes")

    means = np.array([
        mean_covariance(X[y == c], metric=metric_mean) for c in classes
    ])

    if len(classes) == 2:
        num = distance(means[0], means[1], metric=metric_dist) ** exponent
        denom = 0.5 * _get_within(X, y, means, classes, exponent, metric_dist)

    else:
        mean_all = mean_covariance(means, metric=metric_mean)
        dists_between = [
            distance(m, mean_all, metric=metric_dist) ** exponent
            for m in means
        ]
        num = np.sum(dists_between)
        denom = _get_within(X, y, means, classes, exponent, metric_dist)

    class_dis = num / denom

    if return_num_denom:
        return class_dis, num, denom
    else:
        return class_dis


def _get_within(X, y, means, classes, exponent, metric):
    """Private function to compute within dispersion."""
    sigmas = []
    for ic, c in enumerate(classes):
        dists_within = [
            distance(x, means[ic], metric=metric) ** exponent
            for x in X[y == c]
        ]
        sigmas.append(np.mean(dists_within))
    sum_sigmas = np.sum(sigmas)
    return sum_sigmas

#temporal: imports MeanField_V2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from .utils.mean import mean_logeuclid
from .utils.mean import mean_power
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class MeanField_V2(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Minimum Distance to Mean Field.

    Classification by Minimum Distance to Mean Field [1]_, defining several
    power means for each class.

    Parameters
    ----------
    power_list : list of float, default=[-1,0,+1]
        Exponents of power means.
    method_label : {'sum_means', 'inf_means'}, default='sum_means'
        Method to combine labels:

        * sum_means: it assigns the covariance to the class whom the sum of
          distances to means of the field is the lowest;
        * inf_means: it assigns the covariance to the class of the closest mean
          of the field.
    metric : string, default="riemann"
        Metric used for distance estimation during prediction.
        For the list of supported metrics,
        see :func:`pyriemann.utils.distance.distance`.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.
    covmeans_ : dict of ``n_powers`` lists of ``n_classes`` ndarrays of shape \
            (n_channels, n_channels)
        Centroids for each power and each class.

    See Also
    --------
    MDM

    Notes
    -----
    .. versionadded:: 0.3

    References
    ----------
    .. [1] `The Riemannian Minimum Distance to Means Field Classifier
        <https://hal.archives-ouvertes.fr/hal-02315131>`_
        M Congedo, PLC Rodrigues, C Jutten. BCI 2019 - 8th International
        Brain-Computer Interface Conference, Sep 2019, Graz, Austria.
    """

    def __init__(self, power_list=[-1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1], 
                 method_label='lda',
                 metric="riemann",
                 power_mean_zeta = 1e-07, #stopping criterion for the mean calculation, bigger values help with speed
                 distance_squared = True,
                 n_jobs=1, 
                 euclidean_mean  = False,
                 distance_strategy = "power_distance",
                 remove_outliers = True,
                 outliers_th = 2.5,
                 outliers_depth = 4, #how many times to run the outliers detection on the same data
                 outliers_max_remove_th = 30, #default 30%, parameter is percentage
                 outliers_method = "zscore",
                 outliers_mean_init = True,
                 reuse_previous_mean = False,
                 outliers_single_zscore = True, #when false more outliers are removed. When True only the outliers further from the mean are removed
                 ):
        """Init."""
        self.power_list = power_list
        self.method_label = method_label
        self.metric = metric
        self.n_jobs = n_jobs
        self.euclidean_mean = euclidean_mean #if True sets LogEuclidian distance for LogEuclidian mean and Euclidian distance for power mean p=1
        self.distance_strategy = distance_strategy 
        self.remove_outliers = remove_outliers
        self.outliers_th = outliers_th
        self.outliers_depth = outliers_depth
        self.outliers_max_remove_th = outliers_max_remove_th
        self.outliers_method = outliers_method
        self.power_mean_zeta = power_mean_zeta
        self.outliers_mean_init = outliers_mean_init
        self.distance_squared = distance_squared
        self.reuse_previous_mean = reuse_previous_mean
        self.outliers_single_zscore = outliers_single_zscore
        
        '''
        "default_metric" - it uses "metric" (usually Riemann) for all distances 
        "power_mean"     - uses a modified power_distance function based riemann distance, which has an optimization that first calcualtes the inverse of the power mean
        '''
        if distance_strategy not in ["default_metric", "power_distance"]:
            raise Exception()("Invalid distance stategy!")
        
        if (outliers_max_remove_th > 100):
            raise Exception("outliers_max_remove_th is a %, it can not be > 100")
            
        if self.method_label == "lda":
            self.lda = LDA()
    
    def _calculate_mean(self,X, y, p, sample_weight):
        '''
        Calculates mean (and inv mean) for all classes for specific p

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.
        sample_weight : TYPE
            DESCRIPTION.

        Returns
        -------
        means_p : TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        '''
        means_p   = {} #keys are classes, values are means for this p and class
        inv_means = {}
        
        if p == 200: #adding an extra mean - this one is logeuclid and not power mean
            #print("euclidean mean")
            for ll in self.classes_:
                means_p[ll] = mean_logeuclid(
                    X[y == ll],
                    sample_weight=sample_weight[y == ll]
                )       
        else:
            for ll in self.classes_:
                
                init = None
                
                #use previous mean for this p
                #usually when calculating the new mean after outliers removal
                if self.outliers_mean_init and p in self.covmeans_:
                    init = self.covmeans_[p][ll] #use previous mean
                    #print("using init mean")
                
                #use the mean from the previous position in the power list
                elif self.reuse_previous_mean:
                    pos = self.power_list.index(p)
                    if pos>0:
                        prev_p = self.power_list[pos-1]
                        init = self.covmeans_[prev_p][ll]
                        #print(prev_p)
                        #print("using prev mean from the power list")
                 
                means_p[ll] = mean_power( #original is mean_power_custom
                    X[y == ll],
                    p,
                    sample_weight=sample_weight[y == ll],
                    zeta = self.power_mean_zeta,
                    init = init
                )
            
        if self.distance_strategy == "power_distance":
            inv_means= self.calculate_inv_mean_by_mean(means_p)
            
        return means_p,inv_means #contains means for all classes
    
    def _calcualte_mean_remove_outliers(self,X, y, p, sample_weight):
        '''
        Removes outliers and calculates the power mean p on the rest

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.
        sample_weight : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        means_p : TYPE
            DESCRIPTION.
        inv_means : TYPE
            DESCRIPTION.

        '''
        X_no_outliers = X.copy() #so that every power mean p start from the same data
        y_no_outliers = y.copy()
        
        total_outliers_removed_per_class = np.zeros(len(self.classes_))
        total_samples_per_class          = np.zeros(len(self.classes_))
        
        for ll in self.classes_:
            total_samples_per_class[ll] = len(y_no_outliers[y_no_outliers==ll])
        
        if self.outliers_method == "iforest":
            iso = IsolationForest(contamination='auto') #0.1
        elif self.outliers_method == "lof":
            lof = LocalOutlierFactor(contamination='auto', n_neighbors=2) #default = 2
        
        early_stop = False
        
        for i in range(self.outliers_depth):
            
            if early_stop:
                #print("Early stop")
                break
            
            #print("\nremove outliers iteration: ",i)
            
            #calculate/update the n means (one for each class)
            means_p,inv_means = self._calculate_mean(X_no_outliers, y_no_outliers, p, sample_weight)
            
            ouliers_per_iteration_count = {}
            
            #outlier removal is per class
            for ll in self.classes_:
                
                samples_before = X_no_outliers.shape[0]
                
                m = [] #each entry contains a distance to the power mean p for class ll
                
                #length includes all classes, not only the ll
                z_scores = np.zeros(len(y_no_outliers),dtype=float)
            
                # Calcualte all the distances only for class ll and power mean p
                for idx, x in enumerate (X_no_outliers[y_no_outliers==ll]):
                    
                    if self.distance_strategy == "power_distance":
                        #dist_p = self._calculate_distance(x, self.covmeans_inv_[p][ll], p)
                        dist_p = self._calculate_distance(x, inv_means[ll], p)
                    else:
                        #dist_p = self._calculate_distance(x, self.covmeans_[p][ll], p)
                        dist_p = self._calculate_distance(x, means_p[ll], p)
                    #dist_p = np.log(dist_p)
                    m.append(dist_p)
                
                m = np.array(m, dtype=float)
                
                if self.outliers_method == "zscore":
                    
                    m = np.log(m)
                    # Calculate Z-scores for each data point for the current ll class
                    # For the non ll the zscore stays 0, so they won't be removed
                    z_scores[y_no_outliers==ll] = zscore(m)
                
                    if self.outliers_single_zscore:
                        outliers = (z_scores > self.outliers_th)
                    else:
                        outliers = (z_scores > self.outliers_th) | (z_scores < -self.outliers_th)
                    
                elif self.outliers_method == "iforest":
                    
                    m1 = [[k] for k in m]
                    z_scores[y_no_outliers==ll] = iso.fit_predict(m1)
                    #outliers is designed to be the size with all classes
                    outliers = z_scores == -1
                    
                elif self.outliers_method == "lof":
                    
                    m1 = [[k] for k in m]
                    z_scores[y_no_outliers==ll] = lof.fit_predict(m1)
                    #outliers is designed to be the size with all classes
                    outliers = z_scores == -1
                    
                else:   
                    raise Exception("Invalid Outlier Removal Method")

                outliers_count = len(outliers[outliers==True])
                
                #check if too many samples are about to be removed
                #case 1 less than self.max_outliers_remove_th are to be removed
                if ((total_outliers_removed_per_class[ll] + outliers_count) / total_samples_per_class[ll]) * 100 < self.outliers_max_remove_th:
                    #print ("Removed for class ", ll ," ",  len(outliers[outliers==True]), " samples out of ", X_no_outliers.shape[0])
            
                    X_no_outliers = X_no_outliers[~outliers]
                    y_no_outliers = y_no_outliers[~outliers]
                    sample_weight = sample_weight[~outliers]
                
                    if X_no_outliers.shape[0] != (samples_before - outliers_count):
                        raise Exception("Error while removing outliers!")
                    
                    total_outliers_removed_per_class[ll] = total_outliers_removed_per_class[ll] + outliers_count
                
                else: #case 2 more than self.max_outliers_remove_th are to be removed
                
                    outliers_count = 0 #0 set outliers removed to 0
                    
                    print("WARNING: Skipped full outliers removal because too many samples were about to be removed.")
                
                ouliers_per_iteration_count[ll] = outliers_count
            
            #early stop: if no outliers were removed for both classes then we stop early
            if sum(ouliers_per_iteration_count.values()) == 0:
                early_stop = True
        
        total_outliers_removed = total_outliers_removed_per_class.sum()

        if total_outliers_removed > 0:
           
            #generate the final power mean (after outliers removal)
            means_p,inv_means = self._calculate_mean(X_no_outliers, y_no_outliers, p, sample_weight)
        
            outliers_removed_for_single_mean_gt = X.shape[0] - X_no_outliers.shape[0]
            
            if (total_outliers_removed != outliers_removed_for_single_mean_gt):
                raise Exception("Error outliers removal count!")
            
            print("Total outliers removed for mean p=",p," is: ",total_outliers_removed, " for all classes")
            
            if (outliers_removed_for_single_mean_gt / X.shape[0]) * 100 > self.outliers_max_remove_th:
                raise Exception("Outliers removal algorithm has removed too many samples: ", outliers_removed_for_single_mean_gt, " out of ",X.shape[0])
        else: 
            #print("No outliers removed")
            pass
        
        return means_p,inv_means
    
    def power_distance(self, trial, power_mean_inv, squared=False):
        '''
        A distance that requires inv power mean as second parameter

        Parameters
        ----------
        trial : TYPE
            DESCRIPTION.
        power_mean_inv : TYPE
            DESCRIPTION.
        squared : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        #_check_inputs(A, B)
        #_check_inputs(power_mean_inv, trial)

        d2 = (np.log( np.linalg.eigvals (power_mean_inv @ trial)) **2 ).sum(axis=-1)

        return d2 if squared else np.sqrt(d2)

    def _calculate_all_means(self,X,y,sample_weight):
        
        if self.n_jobs==-1 or self.n_jobs > 1:
            print("parallel means")
            if (self.remove_outliers):
                
                results = Parallel(n_jobs=self.n_jobs)(delayed(self._calcualte_mean_remove_outliers)(X, y, p, sample_weight)
                                      for p in self.power_list
                                  )
            else:
                results = Parallel(n_jobs=-1)(delayed(self._calculate_mean)(X, y, p, sample_weight)
                                        for p in self.power_list
                                    )
        else:
            print("NON parallel means")
            results = [] #per p for all classes
            for p in self.power_list:
                
                if (self.remove_outliers):
                    result_per_p = self._calcualte_mean_remove_outliers(X, y, p, sample_weight)
                else:
                    result_per_p = self._calculate_mean(X, y, p, sample_weight)
                results.append(result_per_p)
        
        for i, p in enumerate(self.power_list):
            self.covmeans_[p]     = results[i][0]
            self.covmeans_inv_[p] = results[i][1]
                
    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids. Calculates the power means.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : MeanField instance
            The MeanField instance.
        """
        
        if self.euclidean_mean:
            self.power_list.append(200)
            
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        
        # keys are p, each value is another dictionary over the classes and
        # values of this dictionary are the means for this p and a class
        self.covmeans_ = {}
        self.covmeans_inv_ = {}
        
        self._calculate_all_means(X,y,sample_weight)
        
        if len(self.power_list) != len(self.covmeans_.keys()):
            raise Exception("Problem with number of calculated means!",len(self.power_list),len(self.covmeans_.keys()))
            
        if self.distance_strategy == "power_distance" and len(self.covmeans_.keys()) != len(self.covmeans_inv_.keys()):
            raise Exception("Problem with the number of inverse matrices")
        
        if self.method_label == "lda":
            dists = self._predict_distances(X)
            self.lda.fit(dists,y)

        return self
    
    def calculate_inv_mean_by_mean(self,cov): #for all classes
        '''
        Calculates the inverse mean of a mean covariance matrix.

        Parameters
        ----------
        cov : TYPE
            DESCRIPTION.

        Returns
        -------
        inv_means_p : TYPE
            DESCRIPTION.

        '''
        inv_means_p = {}
        for ll in self.classes_:
            inv_means_p[ll] = np.linalg.inv(cov[ll])
        
        return inv_means_p           

    def _get_label(self, x, labs_unique):
        
        m = np.zeros((len(self.power_list), len(labs_unique)))
        
        for ip, p in enumerate(self.power_list):
            for ill, ll in enumerate(labs_unique):
                 m[ip, ill] = self._calculate_distance(x,self.covmeans_[p][ll],p)

        if self.method_label == 'sum_means':
            ipmin = np.argmin(np.sum(m, axis=1))
        elif self.method_label == 'inf_means':
            ipmin = np.where(m == np.min(m))[0][0]
        else:
            raise TypeError('method_label must be sum_means or inf_means')

        y = labs_unique[np.argmin(m[ipmin])]
        return y

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the closest means field.
        """
        
        #print("In predict")
        if self.method_label == "lda":

            dists = self._predict_distances(X)
            
            pred  = self.lda.predict(dists)
            
            return np.array(pred)
            
        else:
            
            labs_unique = sorted(self.covmeans_[self.power_list[0]].keys())
    
            pred = Parallel(n_jobs=self.n_jobs)(delayed(self._get_label)(x, labs_unique)
                 for x in X
                )
            
            return np.array(pred)

    def _calculate_distance(self,A,B,p):

        squared = self.distance_squared
        
        if len(A.shape) == 2:
        
            if self.distance_strategy == "default_metric":
                
                dist = distance(
                        A,
                        B,
                        metric=self.metric,
                        squared = squared,
                    )
            
            #same as "default_metric", but uses inverse mean
            elif self.distance_strategy == "power_distance":
                
                dist = self.power_distance(
                        A, #trial
                        B, #mean inverted
                        squared = squared,
                    )
            else:
                raise Exception("Invalid distance strategy")
                    
        else:
            raise Exception("Error size of input, not matrices?")
            
        return dist
    
    def _calucalte_distances_for_all_means(self,x):
        '''
        Calculates the distances to all power means 

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        combined : TYPE
            DESCRIPTION.

        '''
        m = {} #contains a distance to a power mean
        
        for p in self.power_list:
            m[p] = []
            
            for ll in self.classes_: #add all distances (1 per class) for m[p] power mean
                
                if self.distance_strategy == "power_distance":
                    dist_p = self._calculate_distance(x, self.covmeans_inv_[p][ll], p)
                else:
                    dist_p = self._calculate_distance(x, self.covmeans_[p][ll], p)
                
                m[p].append(dist_p)
                
        combined = [] #combined for all classes
        for v in m.values():
            combined.extend(v)
        
        #check combned = (number of classes) x (number of power means)
        if len(combined) != (len(self.power_list) * len(self.classes_)) :
            raise Exception("Not enough calculated distances!", len(combined),(len(self.power_list) * 2))
            
        return combined
        
    def _predict_distances(self, X):
        """Helper to predict the distance. Equivalent to transform."""
        
        #print("predict distances")
           
        if (self.n_jobs == 1):
            distances = []
            for x in X:
                distances_per_mean = self._calucalte_distances_for_all_means(x)
                distances.append(distances_per_mean)
        else:
            distances = Parallel(n_jobs=self.n_jobs)(delayed(self._calucalte_distances_for_all_means)(x)
                 for x in X
                )
            
        distances = np.array(distances)
        
        return distances

    def transform(self, X,):
        """Get the distance to each means field.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            Distance to each means field according to the metric.
        """
        return self._predict_distances(X)

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """Predict proba using softmax of negative squared distances.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_matrices, n_classes)
            Probabilities for each class.
        """
        if self.method_label == "lda":
            
            dists = self._predict_distances(X)
            
            return self.lda.predict_proba(dists)
            
        else:
            return softmax(-self._predict_distances(X) ** 2)
