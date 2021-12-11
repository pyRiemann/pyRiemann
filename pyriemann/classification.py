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
from .utils.mean import mean_covariance
from .utils.distance import distance
from .utils.geodesic import geodesic
from .tangentspace import FGDA, TangentSpace


class MDM(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Minimum Distance to Mean.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated according to the chosen metric. Then, for each new
    point, the class is affected according to the nearest centroid.

    Parameters
    ----------
    metric : string | dict, default: 'riemann'
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    n_jobs : int, default: 1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    covmeans_ : list
        The class centroids.
    classes_ : list
        List of classes.

    See Also
    --------
    Kmeans
    FgMDM
    KNearestNeighbor

    References
    ----------
    .. [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Multiclass
        Brain-Computer Interface Classification by Riemannian Geometry," in
        IEEE Transactions on Biomedical Engineering, vol. 59, no. 4,
        p. 920-928, 2012.

    .. [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian
        geometry applied to BCI classification", 9th International Conference
        Latent Variable Analysis and Signal Separation (LVA/ICA 2010),
        LNCS vol. 6365, 2010, p. 629-636.
    """

    def __init__(self, metric='riemann', n_jobs=1):
        """Init."""
        # store params for cloning purpose
        self.metric = metric
        self.n_jobs = n_jobs

        if isinstance(metric, str):
            self.metric_mean = metric
            self.metric_dist = metric

        elif isinstance(metric, dict):
            # check keys
            for key in ['mean', 'distance']:
                if key not in metric.keys():
                    raise KeyError('metric must contain "mean" and "distance"')

            self.metric_mean = metric['mean']
            self.metric_dist = metric['distance']

        else:
            raise TypeError('metric must be dict or str')

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels corresponding to each matrix.
        sample_weight : None | ndarray shape (n_matrices,)
            Weights of each matrix. If None, each matrix is treated with
            equal weights.

        Returns
        -------
        self : MDM instance
            The MDM instance.
        """
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

    def _predict_distances(self, covtest):
        """Helper to predict the distance. Equivalent to transform."""
        n_centroids = len(self.covmeans_)

        if self.n_jobs == 1:
            dist = [distance(covtest, self.covmeans_[m], self.metric_dist)
                    for m in range(n_centroids)]
        else:
            dist = Parallel(n_jobs=self.n_jobs)(delayed(distance)(
                covtest, self.covmeans_[m], self.metric_dist)
                for m in range(n_centroids))

        dist = np.concatenate(dist, axis=1)
        return dist

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
        return softmax(-self._predict_distances(X)**2)


class FgMDM(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Minimum Distance to Mean with geodesic filtering.

    Apply geodesic filtering described in [1]_, and classify using MDM.
    The geodesic filtering is achieved in tangent space with a Linear
    Discriminant Analysis, then data are projected back to the manifold and
    classifier with a regular MDM.
    This is basically a pipeline of FGDA and MDM.

    Parameters
    ----------
    metric : string | dict, default: 'riemann'
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    tsupdate : bool, default False
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]_. This is not compatible with
        online implementation. Performance are better when the number of
        matrices for prediction is higher.
    n_jobs : int, default: 1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    classes_ : list
        List of classes.

    See Also
    --------
    MDM
    FGDA
    TangentSpace

    References
    ----------
    .. [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian
        geometry applied to BCI classification", 9th International Conference
        Latent Variable Analysis and Signal Separation (LVA/ICA 2010),
        LNCS vol. 6365, 2010, p. 629-636.

    .. [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Classification
        of covariance matrices using a Riemannian-based kernel for BCI
        applications", in NeuroComputing, vol. 112, p. 172-178, 2013.
    """

    def __init__(self, metric='riemann', tsupdate=False, n_jobs=1):
        """Init."""
        self.metric = metric
        self.n_jobs = n_jobs
        self.tsupdate = tsupdate

        if isinstance(metric, str):
            self.metric_mean = metric

        elif isinstance(metric, dict):
            # check keys
            for key in ['mean', 'distance']:
                if key not in metric.keys():
                    raise KeyError('metric must contain "mean" and "distance"')

            self.metric_mean = metric['mean']

        else:
            raise TypeError('metric must be dict or str')

    def fit(self, X, y):
        """Fit FgMDM.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels corresponding to each matrix.

        Returns
        -------
        self : FgMDM instance
            The FgMDM instance.
        """
        self.classes_ = np.unique(y)
        self._mdm = MDM(metric=self.metric, n_jobs=self.n_jobs)
        self._fgda = FGDA(metric=self.metric_mean, tsupdate=self.tsupdate)
        cov = self._fgda.fit_transform(X, y)
        self._mdm.fit(cov, y)
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
    metric : string | dict, default: 'riemann'
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    tsupdate : bool, default: False
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]. This is not compatible with
        online implementation. Performance are better when the number of
        matrices for prediction is higher.
    clf: sklearn classifier, default: LogisticRegression
        The classifier to apply in the tangent space.

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

        if not isinstance(clf, ClassifierMixin):
            raise TypeError('clf must be a ClassifierMixin')

    def fit(self, X, y):
        """Fit TSclassifier.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels corresponding to each matrix.

        Returns
        -------
        self : TSclassifier instance
            The TSclassifier instance.
        """
        self.classes_ = np.unique(y)
        ts = TangentSpace(metric=self.metric, tsupdate=self.tsupdate)
        self._pipe = make_pipeline(ts, self.clf)
        self._pipe.fit(X, y)
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
    n_neighbors : int, default: 5
        Number of neighbors.
    metric : string | dict, default: 'riemann'
        The type of metric used for distance estimation.
        see `distance` for the list of supported metric.
    n_jobs : int, default: 1
        The number of jobs to use for the computation. This works by computing
        each of the distance to the training set in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    classes_ : list
        List of classes.
    covmeans_ : list
        The class centroids.
    classmeans_ : list
        List of classes of centroids.

    See Also
    --------
    Kmeans
    MDM

    """

    def __init__(self, n_neighbors=5, metric='riemann', n_jobs=1):
        """Init."""
        # store params for cloning purpose
        self.n_neighbors = n_neighbors
        MDM.__init__(self, metric=metric, n_jobs=n_jobs)

    def fit(self, X, y):
        """Fit (store the training data).

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels corresponding to each matrix.

        Returns
        -------
        self : NearestNeighbor instance
            The NearestNeighbor instance.
        """
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
        probas = softmax(-dist_sorted[:, 0:self.n_neighbors]**2)

        prob = np.zeros((n_matrices, len(self.classes_)))
        for m in range(n_matrices):
            for il, ll in enumerate(self.classes_):
                prob[m, il] = np.sum(
                    probas[m, neighbors_classes[m, 0:self.n_neighbors] == ll]
                )

        return prob


class SVC(sklearnSVC):
    """Classification by support-vector machine.

    Support-vector machine (SVM) with precomputed Riemannian kernel matrix
    according to different metrics as described in [1]_.

    Parameters
    ----------
    metric : {'riemann', 'euclid', 'logeuclid'}, default: 'riemann'
        Metric for kernel matrix computation.
    Cref : None | ndarray | callable, shape (n_channels, n_channels)
        Reference point for kernel matrix computation.
        If None, the mean of the training data according to the metric is used.
        If callable, the function is called on the training data to calculate
        Cref.
    kernel_fct : 'precomputed' | callable
        If 'precomputed', the kernel matrix for datasets X and Y is estimated
        according to pyriemann.utils.kernel(X, Y, Cref, metric).
        If callable, the callable is passed as the kernel parameter to
        sklearn.svm.SVC(). The callable has to be of the form
        kernel(X, Y, Cref, metric).
    C : float, default: 1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.
    shrinking : bool, default: True
        Whether to use the shrinking heuristic.
    probability : bool, default: False
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, will slow down that method as it internally uses
        5-fold cross-validation, and `predict_proba` may be inconsistent with
        `predict`. Read more in the :ref:`User Guide <scores_probabilities>`.
    tol : float, default: 1e-3
        Tolerance for stopping criterion.
    cache_size : float, default: 200
        Specify the size of the kernel cache (in MB).
    class_weight : dict or 'balanced', default: None
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_matrices / (n_classes * np.bincount(y))``.
    verbose : bool, default: False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.
    max_iter : int, default: -1
        Hard limit on iterations within solver, or -1 for no limit.
    decision_function_shape : {'ovo', 'ovr'}, default: 'ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_matrices, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_matrices, n_classes * (n_classes - 1) / 2). However, note that
        internally, one-vs-one ('ovo') is always used as a multi-class strategy
        to train models; an ovr matrix is only constructed from the ovo matrix.
        The parameter is ignored for binary classification.
    break_ties : bool, default: False
        If true, ``decision_function_shape='ovr'``, and number of classes > 2,
        :term:`predict` will break ties according to the confidence values of
        :term:`decision_function`; otherwise the first class among the tied
        classes is returned. Please note that breaking ties comes at a
        relatively high computational cost compared to a simple predict.
    random_state : int, RandomState instance or None, default: None
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when `probability` is False.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

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
            Labels corresponding to each matrix.
        sample_weight : ndarray, shape (n_matrices,), default: None
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        Returns
        -------
        self : SVC instance
            The SVC instance.
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


class MDWM (MDM):
    """Classification by Minimum Distance to Weighted Mean.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated, according to the chosen metric, as a weighted mean
    of point (i.e. covariance matrices) from the source domain, combined with
    the class centroid of the target domain.
    For classification, a given new point is attibuted to the class whose
    centroid is the nearest according to the chosen metric.

    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    L : float, (default: 0)
        Transfer coefficient in [0,1]. This parameter controls the trade-off 
        between source and target data.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    covmeans_ : list
        the class centroids.
    classes_ : list
        list of classes.

    See Also
    --------
    MDM
    Kmeans
    FgMDM
    KNearestNeighbor

    References
    ----------
    [1] E. Kalunga, S. Chevallier and Q. Barthélemy, "Transfer learning for
    SSVEP-based BCI using Riemannian similarities between users", in 26th
    European Signal Processing Conference (EUSIPCO), pp. 1685-1689. IEEE, 2018.

    [2] S. Khazem, S. Chevallier, Q. Barthélemy, K. Haroun and C. Noûs,
    "Minimizing Subject-dependent Calibration for BCI with Riemannian Transfer
    Learning", in 10th International IEEE/EMBS Conference on Neural
    Engineering (NER), pp. 523-526. IEEE, 2021.
    """

    def __init__(self, metric='riemann', L=0, n_jobs=1):
        """Init."""
        self.metric = metric
        self.n_jobs = n_jobs
        self.L = L

        if isinstance(metric, str):
            self.metric_mean = metric
            self.metric_dist = metric

        elif isinstance(metric, dict):
            # check keys
            for key in ['mean', 'distance']:
                if key not in metric.keys():
                    raise KeyError('metric must contain "mean" and "distance"')

            self.metric_mean = metric['mean']
            self.metric_dist = metric['distance']

        else:
            raise TypeError('metric must be dict or str')

    def fit(self, X, y, X_source, y_source, sample_weight=None):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            ndarray of SPD matrices from target subject
        y : ndarray shape (n_matrices, 1)
            labels corresponding to each trial of target subject
        X_source : ndarray, shape (n_matrices, n_channels, n_channels)
            ndarray of SPD matrices from source domain subjects
        y_source : ndarray shape (n_matrices, 1)
            labels corresponding to each trial.
        sample_weight : None | ndarray shape (n_matrices, 1)
            the weights of each sample from the domain. if None, each sample
            is treated with equal weights.

        Returns
        -------
        self : MDWM instance
            The MDWM instance.
        """

        if set(y) != set(y_source):
            raise Exception(f"classes in source domain must match classes in target \
                domain. Classes in source are {np.unique(y_source)} while \
                    classes in target are {np.unique(y)}")

        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X_source.shape[0])

        if self.n_jobs == 1:
            self.target_means_ = [
                mean_covariance(X[y == ll], metric=self.metric_mean)
                for ll in self.classes_]

            self.domain_means_ = [
                mean_covariance(
                    X_source[y_source == ll],
                    metric=self.metric_mean,
                    sample_weight=sample_weight[y_source == ll]
                    )
                for ll in self.classes_]
        else:
            self.target_means_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(X[y == ll], metric=self.metric_mean)
                for ll in self.classes_)
            self.domain_means_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(
                    X_source[y_source == ll],
                    metric=self.metric_mean,
                    sample_weight=sample_weight[y_source == ll])
                for ll in self.classes_)

        self.class_center_ = [geodesic(self.target_means_[i],
                                       self.domain_means_[i],
                                       self.L, self.metric)
                              for i, _ in enumerate(self.classes_)]
        return self
