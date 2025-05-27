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
        Number of jobs to use for the computation. This works by computing
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

        dist = Parallel(n_jobs=self.n_jobs)(
            delayed(distance)(X, covmean, self.metric_dist)
            for covmean in self.covmeans_
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
        Number of jobs to use for the computation. This works by computing
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
        Number of jobs to use for the computation. This works by computing
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
        estimated according to ``pyriemann.utils.kernel(X, Y, Cref, metric)``.
        If callable, the callable is passed as the kernel parameter to
        ``sklearn.svm.SVC()`` [2]_. The callable has to be of the form
        ``kernel(X, Y, Cref, metric)``.
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.
    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
    probability : bool, default=False
        Whether to enable probability estimates. This must be enabled prior
        to calling ``fit``, will slow down that method as it internally uses
        5-fold cross-validation, and ``predict_proba`` may be inconsistent with
        ``predict``.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).
    class_weight : None | dict | "balanced", default=None
        Set the parameter C of class i to class_weight[i]*C for SVC. If not
        given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as n_matrices / (n_classes * np.bincount(y)).
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
        If True, ``decision_function_shape`` ="ovr", and n_classes > 2,
        ``predict`` will break ties according to the confidence values of
        ``decision_function``; otherwise the first class among the tied
        classes is returned. Please note that breaking ties comes at a
        relatively high computational cost compared to a simple predict.
    random_state : None | int | RandomState instance, default=None
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when ``probability`` is False.
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
    """Classification by Mean Field.

    The Mean Field estimates several power means for each class.
    Then, it can be used as a classifier, which computes the minimum distance
    to the mean field [1]_;
    or as a feature extractor, which must be pipelined with another classifier
    [2]_.

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
    covmeans_ : dict of len(``power_list``) dicts of n_classes ndarrays of \
            shape (n_channels, n_channels)
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
    .. [2] `The Riemannian Means Field Classifier for EEG-Based BCI Data
        <https://www.mdpi.com/1424-8220/25/7/2305>`_
        A Andreev, G Cattan, M Congedo. MDPI Sensors journal, April 2025
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
