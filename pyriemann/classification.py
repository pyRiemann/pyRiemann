"""Module for classification function."""
import numpy as np

from scipy import stats

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed

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
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        sample_weight : None | ndarray shape (n_trials, 1)
            the weights of each sample. if None, each sample is treated with
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
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        return self.classes_[dist.argmin(axis=1)]

    def transform(self, X):
        """get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_trials, n_classes)
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
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
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
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

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
        """get the predictions after FGDA filtering.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        cov = self._fgda.transform(X)
        return self._mdm.predict(cov)

    def predict_proba(self, X):
        """Predict proba using softmax after FGDA filtering.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
            the softmax probabilities for each class.
        """
        cov = self._fgda.transform(X)
        return self._mdm.predict_proba(cov)

    def transform(self, X):
        """get the distance to each centroid after FGDA filtering.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_trials, n_cluster)
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
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

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
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        return self._pipe.predict(X)

    def predict_proba(self, X):
        """get the probability.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of ifloat, shape (n_trials, n_classes)
            the prediction for each trials according to the closest centroid.
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
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

        Returns
        -------
        self : NearestNeighbor instance
            The NearestNeighbor instance.
        """
        self.classes_ = y
        self.covmeans_ = X

        return self

    def predict(self, covtest):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        neighbors_classes = self.classes_[np.argsort(dist)]
        out, _ = stats.mode(neighbors_classes[:, 0:self.n_neighbors], axis=1)
        return out.ravel()

class MDWM (MDM):
    """Classification by Minimum Distance to Weighted Mean.

    Classification by nearest centroid. For each of the given classes, a 
    centroid is estimated, according to the chosen metric, as a weighted mean
    of point (i.e. covariance matrices) from the source domain, combined with 
    the class centroid of the target domain. 
    For classification, a given new point is attibuted to the class whose centroid is 
    the nearest according to the chosen metric.

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
        Transfer coefficient. This parameter controls the trade-off between 
        source and target data.
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

    def __init__(self,metric='riemann', L=0, n_jobs=1):
        
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
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices from target subject
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial of target subject
        X_source : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices from source domain subjects
        y_source : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        sample_weight : None | ndarray shape (n_trials, 1)
            the weights of each sample from the domain. if None, each sample
            is treated with equal weights.

        Returns
        -------
        self : MDWM instance
            The MDWM instance.
        """
        
        if set(y) != set(y_source):
            raise Exception(f"classes in source domain must match classes in target \
                domain. Classes in source are {np.unique(y_source)} while classes \
                    in target are {np.unique(y)}")

        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X_source.shape[0])
             
        if self.n_jobs == 1:
            self.target_means_ = [mean_covariance(X[y == l], 
                                              metric=self.metric_mean)
                                        for l in self.classes_]

            self.domain_means_ = [mean_covariance(X_source[y_source == l], 
                                                  metric=self.metric_mean,
                                    sample_weight=sample_weight[y_source == l])
                                        for l in self.classes_]
        else:
            self.target_means_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(X[y == l], metric=self.metric_mean)
                for l in self.classes_) 
            self.domain_means_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(X_source[y_source == l],
                                metric=self.metric_mean,
                                sample_weight=sample_weight[y_source == l])
                for l in self.classes_)

        self.class_center_ = [geodesic(self.target_means_[i], 
                                       self.domain_means_[i],
                                       self.L, self.metric) 
                for i, _ in enumerate(self.classes_)]
        return self