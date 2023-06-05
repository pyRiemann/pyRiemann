"""Module for classification function."""
import functools

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.svm import SVC as sklearnSVC
from sklearn.utils.extmath import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed

from .utils.kernel import kernel
from .utils.mean import mean_covariance, mean_power
from .utils.distance import distance
from .tangentspace import FGDA, TangentSpace


def _check_metric(metric):
    if isinstance(metric, str):
        metric_mean = metric
        metric_dist = metric

    elif isinstance(metric, dict):
        # check keys
        for key in ['mean', 'distance']:
            if key not in metric.keys():
                raise KeyError('metric must contain "mean" and "distance"')

        metric_mean = metric['mean']
        metric_dist = metric['distance']

    else:
        raise TypeError('metric must be dict or str')

    return metric_mean, metric_dist


class MDM(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Minimum Distance to Mean.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated according to the chosen metric. Then, for each new
    point, the class is affected according to the nearest centroid.

    Parameters
    ----------
    metric : string | dict, default='riemann'
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metrics for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
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
    covmeans_ : list of ``n_classes`` ndarrays of shape (n_channels, \
            n_channels)
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

    def __init__(self, metric='riemann', n_jobs=1):
        """Init."""
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids.

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
        self : MDM instance
            The MDM instance.
        """
        self.metric_mean, self.metric_dist = _check_metric(self.metric)
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        if self.n_jobs == 1:
            self.covmeans_ = [
                mean_covariance(X[y == ll], metric=self.metric_mean,
                                sample_weight=sample_weight[y == ll])
                for ll in self.classes_]
        else:
            self.covmeans_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(X[y == ll], metric=self.metric_mean,
                                         sample_weight=sample_weight[y == ll])
                for ll in self.classes_)

        return self

    def _predict_distances(self, X):
        """Helper to predict the distance. Equivalent to transform."""
        n_centroids = len(self.covmeans_)

        if self.n_jobs == 1:
            dist = [distance(X, self.covmeans_[m], self.metric_dist)
                    for m in range(n_centroids)]
        else:
            dist = Parallel(n_jobs=self.n_jobs)(delayed(distance)(
                X, self.covmeans_[m], self.metric_dist)
                for m in range(n_centroids))

        dist = np.concatenate(dist, axis=1)
        return dist

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the closest centroid.
        """
        dist = self._predict_distances(X)
        return self.classes_[dist.argmin(axis=1)]

    def transform(self, X):
        """Get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            The distance to each centroid according to the metric.
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
        return softmax(-self._predict_distances(X) ** 2)


class FgMDM(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Minimum Distance to Mean with geodesic filtering.

    Apply geodesic filtering described in [1]_, and classify using MDM.
    The geodesic filtering is achieved in tangent space with a Linear
    Discriminant Analysis, then data are projected back to the manifold and
    classifier with a regular MDM.
    This is basically a pipeline of FGDA and MDM.

    Parameters
    ----------
    metric : string | dict, default='riemann'
        The type of metric used for reference matrix estimation (see
        `mean_covariance` for the list of supported metric), for distance
        estimation, and for tangent space map (see `tangent_space` for the list
        of supported metric).
        The metric could be a dict with three keys, `mean`, `dist` and `map` in
        order to pass different metrics for the reference matrix estimation,
        the distance estimation, and the tangent space mapping.
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

    def __init__(self, metric='riemann', tsupdate=False, n_jobs=1):
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
        self.classes_ = np.unique(y)

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
            Predictions for each matrix according to the closest centroid.
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
            The distance to each centroid according to the metric.
        """
        cov = self._fgda.transform(X)
        return self._mdm.transform(cov)


class TSclassifier(BaseEstimator, ClassifierMixin):
    """Classification in the tangent space.

    Project data in the tangent space and apply a classifier on the projected
    data. This is a simple helper to pipeline the tangent space projection and
    a classifier. Default classifier is LogisticRegression

    Parameters
    ----------
    metric : string | dict, default='riemann'
        The type of metric used for reference matrix estimation (see
        `mean_covariance` for the list of supported metric) and for tangent
        space map (see `tangent_space` for the list of supported metric).
        The metric could be a dict with two keys, `mean` and `map` in
        order to pass different metrics for the reference matrix estimation
        and the tangent space mapping.
    tsupdate : bool, default=False
        Activate tangent space update for covariate shift correction between
        training and test, as described in [2]. This is not compatible with
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
    """

    def __init__(self, metric='riemann', tsupdate=False,
                 clf=LogisticRegression()):
        """Init."""
        self.metric = metric
        self.tsupdate = tsupdate
        self.clf = clf

    def fit(self, X, y, sample_weight=None):
        """Fit TSclassifier.

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
        self : TSclassifier instance
            The TSclassifier instance.
        """
        if not isinstance(self.clf, ClassifierMixin):
            raise TypeError('clf must be a ClassifierMixin')
        self.classes_ = np.unique(y)

        ts = TangentSpace(metric=self.metric, tsupdate=self.tsupdate)
        self._pipe = make_pipeline(ts, self.clf)
        sample_weight_dict = {}
        for step in self._pipe.steps:
            step_name = step[0]
            sample_weight_dict[step_name + '__sample_weight'] = sample_weight
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
            Predictions for each matrix according to the closest centroid.
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
            Predictions for each matrix according to the closest centroid.
        """
        return self._pipe.predict_proba(X)


class KNearestNeighbor(MDM):
    """Classification by k-nearest neighbors.

    Classification by k-nearest neighbors (k-NN). For each point of the test
    set, the pairwise distance to each element of the training set is
    estimated. The class is affected according to the majority class of the
    k-nearest neighbors.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors.
    metric : string | dict, default='riemann'
        The type of metric used for distance estimation.
        see `distance` for the list of supported metric.
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

    def __init__(self, n_neighbors=5, metric='riemann', n_jobs=1):
        """Init."""
        self.n_neighbors = n_neighbors
        MDM.__init__(self, metric=metric, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        """Fit (store the training data).

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : NearestNeighbor instance
            The NearestNeighbor instance.
        """
        self.metric_mean, self.metric_dist = _check_metric(self.metric)
        self.covmeans_ = X
        self.classmeans_ = y
        self.classes_ = np.unique(y)

        return self

    def predict(self, covtest):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        neighbors_classes = self.classmeans_[np.argsort(dist)]
        out, _ = stats.mode(neighbors_classes[:, 0:self.n_neighbors], axis=1)
        return out.ravel()

    def predict_proba(self, X):
        """Predict proba using softmax.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

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
            for il, ll in enumerate(self.classes_):
                prob[m, il] = np.sum(
                    probas[m, neighbors_classes[m, 0:self.n_neighbors] == ll]
                )

        return prob


class SVC(sklearnSVC):
    """Classification by support-vector machine.

    Support-vector machine (SVM) classification with precomputed Riemannian
    kernel matrix according to different metrics as described in [1]_.

    Parameters
    ----------
    metric : {'riemann', 'euclid', 'logeuclid'}, default='riemann'
        Metric for kernel matrix computation.
    Cref : None | callable | ndarray, shape (n_channels, n_channels)
        Reference point for kernel matrix computation.
        If None, the mean of the training data according to the metric is used.
        If callable, the function is called on the training data to calculate
        Cref.
    kernel_fct : None | 'precomputed' | callable
        If 'precomputed', the kernel matrix for datasets X and Y is estimated
        according to `pyriemann.utils.kernel(X, Y, Cref, metric)`.
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
    class_weight : dict or 'balanced', default=None
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
    decision_function_shape : {'ovo', 'ovr'}, default='ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_matrices, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_matrices, n_classes * (n_classes - 1) / 2). However, note that
        internally, one-vs-one ('ovo') is always used as a multi-class strategy
        to train models; an ovr matrix is only constructed from the ovo matrix.
        The parameter is ignored for binary classification.
    break_ties : bool, default=False
        If true, ``decision_function_shape='ovr'``, and number of classes > 2,
        `predict` will break ties according to the confidence values of
        `decision_function`; otherwise the first class among the tied
        classes is returned. Please note that breaking ties comes at a
        relatively high computational cost compared to a simple predict.
    random_state : int, RandomState instance or None, default=None
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

    def __init__(self,
                 *,
                 metric='riemann',
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
                 random_state=None):
        """Init."""
        self.Cref = Cref
        self.metric = metric
        self.Cref_ = None
        self.kernel_fct = kernel_fct
        super().__init__(kernel='precomputed',
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
            raise TypeError(f'Cref must be np.ndarray, callable or None, is'
                            f' {self.Cref}.')

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


class MeanField(BaseEstimator, ClassifierMixin, TransformerMixin):
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
    metric : string | dict, default='riemann'
        The type of metric used for distance estimation during prediction.
        See `distance` for the list of supported metric.

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

    def __init__(self, power_list=[-1, 0, 1], method_label='sum_means',
                 metric='riemann', n_jobs=1):
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
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        self.covmeans_ = {}
        for p in self.power_list:
            means_p = {}
            for ll in self.classes_:
                means_p[ll] = mean_power(
                    X[y == ll],
                    p,
                    sample_weight=sample_weight[y == ll]
                )
            self.covmeans_[p] = means_p

        return self

    def _get_label(self, x, labs_unique):
        m = np.zeros((len(self.power_list), len(labs_unique)))
        for ip, p in enumerate(self.power_list):
            for ill, ll in enumerate(labs_unique):
                m[ip, ill] = distance(
                    x, self.covmeans_[p][ll], metric=self.metric, squared=True)

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
        labs_unique = sorted(self.covmeans_[self.power_list[0]].keys())

        pred = Parallel(n_jobs=self.n_jobs)(delayed(self._get_label)(
            x, labs_unique)
            for x in X)
        return np.array(pred)

    def _predict_distances(self, X):
        """Helper to predict the distance. Equivalent to transform."""

        dist = []
        for x in X:
            m = {}
            for p in self.power_list:
                m[p] = []
                for ll in self.classes_:
                    m[p].append(
                        distance(
                            x,
                            self.covmeans_[p][ll],
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
        return softmax(-self._predict_distances(X) ** 2)


def class_distinctiveness(X, y, exponent=1, metric='riemann',
                          return_num_denom=False):
    r"""Measure class distinctiveness between classes of SPD matrices.

    For two class problem, the class distinctiveness between class A
    and B on the manifold of SPD matrices is quantified as [1]_:

    .. math::
        \mathrm{classDis}(A, B, p) = \frac{d \left(\bar{X}^{A},
        \bar{X}^{B}\right)^p}
        {\frac{1}{2} \left( \sigma_{X^{A}}^p + \sigma_{X^{B}}^p \right)}

    where :math:`\bar{X}^{K}` is the center of class K, ie the mean of matrices
    from class K (see :func:`pyriemann.utils.mean.mean_covariance`) and
    :math:`\sigma_{X^{K}}` is the class dispersion, ie the mean of distances
    between matrices from class K and their center of class
    :math:`\bar{X}^{K}`:

    .. math::
        \sigma_{X^{K}}^p = \frac{1}{m} \sum_{i=1}^m d
        \left(X_i, \bar{X}^{K}\right)^p

    For more than two classes, it is quantified as:

    .. math::
        \mathrm{classDis}\left(\left\{K_{j}\right\}, p\right) =
        \frac{\sum_{j=1}^{c} d\left(\bar{X}^{K_{j}}, \tilde{X}\right)^p}
        {\sum_{j=1}^{c} \sigma_{X^{K_{j}}}^p}

    where :math:`\tilde{X}` is the mean of centers of class of all :math:`c`
    classes and :math:`p` is the exponentiation of the distance measure
    named exponent at the input of this function.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    y : ndarray, shape (n_matrices,)
        Labels for each matrix.
    exponent : int, default=1
        Parameter for exponentiation of distances, corresponding to p in the
        above equations:

        - exponent = 1 gives the formula originally defined in [1]_;
        - exponent = 2 gives the Fisher criterion generalized on the manifold,
          ie the ratio of the variance between the classes to the variance
          within the classes.
    metric : string | dict, default='riemann'
        The type of metric used for centroid and distance estimation.
        See `mean_covariance` for the list of supported metric.
        The metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metrics for the centroid estimation and the
        distance estimation. The original equation of class distinctiveness
        in [1]_ uses 'riemann' for both the centroid estimation and the
        distance estimation but you can customize other metrics with your
        interests.
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

    metric_mean, metric_dist = _check_metric(metric)
    classes = np.unique(y)
    if len(classes) <= 1:
        raise ValueError('X must contain at least two classes')

    means = np.array([
        mean_covariance(X[y == ll], metric=metric_mean) for ll in classes
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


def _get_within(X, y, means, classes, exponent, metric_dist):
    """Private function to compute within dispersion."""
    sigmas = []
    for ii, ll in enumerate(classes):
        dists_within = [
            distance(x, means[ii], metric=metric_dist) ** exponent
            for x in X[y == ll]
        ]
        sigmas.append(np.mean(dists_within))
    sum_sigmas = np.sum(sigmas)
    return sum_sigmas
