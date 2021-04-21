"""Module for classification function."""
import numpy

from scipy import stats

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed

from .utils.mean import mean_covariance
from .utils.distance import distance
from .tangentspace import FGDA, TangentSpace

from qiskit import BasicAer, IBMQ
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.algorithms import SklearnSVM
from qiskit.ml.datasets import ad_hoc_data, sample_ad_hoc_data
from qiskit.providers.ibmq import least_busy

from datetime import datetime



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
        self.classes_ = numpy.unique(y)

        if sample_weight is None:
            sample_weight = numpy.ones(X.shape[0])

        if self.n_jobs == 1:
            self.covmeans_ = [mean_covariance(X[y == l], metric=self.metric_mean,
                                    sample_weight=sample_weight[y == l])
                                        for l in self.classes_]
            """
            for l in self.classes_:
                self.covmeans_.append(
                    mean_covariance(X[y == l], metric=self.metric_mean,
                                    sample_weight=sample_weight[y == l]))
            """
        else:
            self.covmeans_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(X[y == l], metric=self.metric_mean,
                                         sample_weight=sample_weight[y == l])
                for l in self.classes_)

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

        dist = numpy.concatenate(dist, axis=1)
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
        return softmax(-self._predict_distances(X))


class FgMDM(BaseEstimator, ClassifierMixin, TransformerMixin):

    """Classification by Minimum Distance to Mean with geodesic filtering.

    Apply geodesic filtering described in [1], and classify using MDM algorithm
    The geodesic filtering is achieved in tangent space with a Linear
    Discriminant Analysis, then data are projected back to the manifold and
    classifier with a regular mdm.
    This is basically a pipeline of FGDA and MDM

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
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    See Also
    --------
    MDM
    FGDA
    TangentSpace

    References
    ----------
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian geometry
    applied to BCI classification", 9th International Conference Latent
    Variable Analysis and Signal Separation (LVA/ICA 2010), LNCS vol. 6365,
    2010, p. 629-636.

    [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Classification of
    covariance matrices using a Riemannian-based kernel for BCI applications",
    in NeuroComputing, vol. 112, p. 172-178, 2013.
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
        self._mdm = MDM(metric=self.metric, n_jobs=self.n_jobs)
        self._fgda = FGDA(metric=self.metric_mean, tsupdate=self.tsupdate)
        cov = self._fgda.fit_transform(X, y)
        self._mdm.fit(cov, y)
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
        neighbors_classes = self.classes_[numpy.argsort(dist)]
        out, _ = stats.mode(neighbors_classes[:, 0:self.n_neighbors], axis=1)
        return out.ravel()

class QuanticSVM(BaseEstimator, ClassifierMixin):
  
    def __init__(self, target, qAccountToken = None, quantum = True, processVector=lambda v:v, verbose=True, **parameters):
        self.verbose = verbose
        self.log("Initializing Quantum SVM")
        self.set_params(**parameters)
        self.processVector = processVector
        self.qAccountToken = qAccountToken
        self.training_input = {}
        self.target = target
        self.quantum = quantum
        if quantum: 
          aqua_globals.random_seed = datetime.now().microsecond
          self.log("seed = ", aqua_globals.random_seed)
          if qAccountToken:
            self.log("Real quantum computation will be performed")
            IBMQ.save_account(qAccountToken)
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
          else:
            self.log("Quantum simulation will be performed")
            self.backend = BasicAer.get_backend('qasm_simulator')
        else:
          self.log("Classical SVM will be performed")
    
    def log(self, *values):
      if self.verbose:
        print("[QSVM] ", *values)

    def vectorize(self, X):
      vector = X.reshape(len(X), self.feature_dim)
      return [self.processVector(x) for x in vector]

    def splitTargetAndNonTarget(self, X, y):
        self.log("[Warning] Spitting target from non target. Only binary classification is supported.")
        nbSensor = len(X[0])
        nbSamples = len(X[0][0])
        self.feature_dim = nbSensor*nbSamples
        self.log("Feature dimension = ", self.feature_dim)
        Xta = X[y == self.target]
        
        nbXta = len(Xta)
        Xnt = X[numpy.logical_not(y == self.target)]
        nbXnt = len(Xnt)
        balanced = nbXnt == nbXta
        if(not balanced):
          self.log("Set is not balanced. Balancing...")
          Xnt = Xnt[range(nbXta)]
        
        VectorizedXta = self.vectorize(Xta)
        VectorizedXnt = self.vectorize(Xnt)
        
        self.log("Feature dimension after vector processing = ", len(VectorizedXta[0]))

        return (VectorizedXta, VectorizedXnt)

    def fit(self, X, y):
        self.log("Fitting: ", X.shape)
        self.classes_ = numpy.unique(y)
        VectorizedXta, VectorizedXnt = self.splitTargetAndNonTarget(X, y)

        vectorLen = len(VectorizedXta[0])

        self.training_input["Target"] = VectorizedXta
        self.training_input["NonTarget"] = VectorizedXnt
        self.feature_map = ZZFeatureMap(feature_dimension=vectorLen, reps=2, entanglement='linear')

        if self.quantum:
          if not self.backend:
            devices = provider.backends(filters=lambda x: x.configuration().n_qubits >= vectorLen
                                            and not x.configuration().simulator 
                                            and x.status().operational==True)
            self.backend = least_busy(devices)
            self.log("Quantum backend = ", self.backend)

          self.quantum_instance = QuantumInstance(self.backend, shots=1024, seed_simulator=aqua_globals.random_seed, seed_transpiler=aqua_globals.random_seed)
        return self

    def get_params(self, deep=True):
      # suppose this estimator has parameters "alpha" and "recursive"
      return {"target":self.target,
       "qAccountToken":self.qAccountToken,
        "quantum":self.quantum,
        "processVector":self.processVector,
        "verbose":self.verbose,
         "ba":self.ba,
         "test_input":self.test_input
         }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def run(self, predict_set=None):
      if self.quantum:
        qsvm = QSVM(self.feature_map, self.training_input, self.test_input, predict_set)
        result = qsvm.run(self.quantum_instance)
      else:
        result = SklearnSVM(self.training_input, self.test_input, predict_set).run()
      return result

    def predict(self, X):
      result = None
      predict_set = self.vectorize(X)
      self.log("Prediction: ", X.shape)
      result = self.run(predict_set)

      self.log("Prediction finished. Returning predicted labels")
      return result["predicted_labels"]

    def predict_proba(self, X):
      self.log("[WARNING] SVM prediction probabilities are not available. Results from predict will be used instead.")
      predicted_labels = self.predict(X)
      ret = [numpy.array([c == 0, c == 1]) for c in predicted_labels]
      return numpy.array(ret)

    def score(self, X, y):
        self.log("Scoring: ", X.shape)
        VectorizedXta, VectorizedXnt = self.splitTargetAndNonTarget(X, y)

        self.test_input = {}
        self.test_input["Target"] = VectorizedXta
        self.test_input["NonTarget"] = VectorizedXnt
        
        result = self.run()

        self.ba = result["testing_accuracy"]
        self.log("Balanced accuracy = ", self.ba)
           
        return result["testing_accuracy"]
