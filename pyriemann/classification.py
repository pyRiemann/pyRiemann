"""Module for classification function."""
import numpy as np

from scipy import stats

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.svm import SVC as sklearnSVC
from sklearn.utils.extmath import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from joblib import Parallel, delayed

from .utils.kernel import kernel
from .utils.mean import mean_covariance
from .utils.distance import distance
from .tangentspace import FGDA, TangentSpace


class MDM(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Minimum Distance to Mean.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated according to the chosen metric. Then, for each new
    point, the class is affected according to the nearest centroid.

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
    Kmeans
    FgMDM
    KNearestNeighbor

    References
    ----------
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Multiclass
    Brain-Computer Interface Classification by Riemannian Geometry," in IEEE
    Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.

    [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian geometry
    applied to BCI classification", 9th International Conference Latent
    Variable Analysis and Signal Separation (LVA/ICA 2010), LNCS vol. 6365,
    2010, p. 629-636.
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
        y : ndarray, shape (n_matrices, 1)
            Labels corresponding to each matrix.
        sample_weight : None | ndarray shape (n_matrices, 1)
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
        """Helper to predict the distance. equivalent to transform."""
        Nc = len(self.covmeans_)

        if self.n_jobs == 1:
            dist = [distance(covtest, self.covmeans_[m], self.metric_dist)
                    for m in range(Nc)]
        else:
            dist = Parallel(n_jobs=self.n_jobs)(delayed(distance)(
                covtest, self.covmeans_[m], self.metric_dist)
                for m in range(Nc))

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
        pred : ndarray of int, shape (n_matrices, 1)
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
            the distance to each centroid according to the metric.
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
            the softmax probabilities for each class.
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
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]_. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    classes_ : list
        list of classes.

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
        y : ndarray, shape (n_matrices, 1)
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
        pred : ndarray of int, shape (n_matrices, 1)
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
            the softmax probabilities for each class.
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
            the distance to each centroid according to the metric.
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
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.
    clf: sklearn classifier (default LogisticRegression)
        The classifier to apply in the tangent space

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
        y : ndarray, shape (n_matrices, 1)
            Labels corresponding to each matrix.

        Returns
        -------
        self : TSclassifier. instance
            The TSclassifier. instance.
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
        pred : ndarray of int, shape (n_matrices, 1)
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
    """Classification by K-NearestNeighbor.

    Classification by nearest Neighbors. For each point of the test set, the
    pairwise distance to each element of the training set is estimated. The
    class is affected according to the majority class of the k nearest
    neighbors.

    Parameters
    ----------
    n_neighbors : int, (default: 5)
        Number of neighbors.
    metric : string | dict (default: 'riemann')
        The type of metric used for distance estimation.
        see `distance` for the list of supported metric.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the distance to the training set in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    classes_ : list
        list of classes.

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
        y : ndarray, shape (n_matrices, 1)
            Labels corresponding to each matrix.

        Returns
        -------
        self : NearestNeighbor instance
            The NearestNeighbor instance.
        """
        self.classes_ = y
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
        neighbors_classes = self.classes_[np.argsort(dist)]
        out, _ = stats.mode(neighbors_classes[:, 0:self.n_neighbors], axis=1)
        return out.ravel()


class SVC(BaseEstimator, ClassifierMixin):
    """Classification by Riemannian Support Vector Machine.

    Support vector machine with precomputed Riemannian kernel matrix
    according to different metrics as described in [1]_.

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

    def __init__(self, metric='riemann', Cref=None, **kwargs):
        """Init."""
        self.svc_func = sklearnSVC
        self.Cref = Cref
        self.metric = metric
        self.svc_params = kwargs
        self.svc_ = self.svc_func(kernel='precomputed', **self.svc_params)

    def __setattr__(self, name, value):
        """Enable setting attributes for SVC subclass."""
        if 'svc_' in self.__dict__.keys():
            if name in self.svc_.get_params():
                self.svc_.set_params(**{name: value})
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
        self.svc_ = self.svc_.fit(kernelmat, y)

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
            Predictions for each matrix according to the SVC.
        """
        test_kernel_mat = kernel(X,
                                 self.data_,
                                 Cref=self.Cref,
                                 metric=self.metric)
        return self.svc_.predict(test_kernel_mat)

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_matrices, n_classes)
            the probabilities for each class.
        """
        test_kernel_mat = kernel(X,
                                 self.data_,
                                 Cref=self.Cref,
                                 metric=self.metric)

        if self.probA_.size == 0 or self.probB_.size == 0:
            raise NotFittedError(
                "predict_proba is not available when fitted with "
                "probability=False"
            )

        return self.svc_.predict_proba(test_kernel_mat)
