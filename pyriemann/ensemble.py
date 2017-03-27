"""
Ensemble Classificatiers

"""

# Authors: AJ Keller <aj@pushtheworld.us>
#
# License: BSD 3 clause
from __future__ import division

import numpy as np

from scipy.linalg import eigh

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from .utils.covariance import _check_est
from numpy.core.numerictypes import typecodes



def _parallel_fit_estimator(estimator, X, y, sample_weight):
    """Private function used to fit an estimator within a job."""
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight)
    else:
        estimator.fit(X, y)
    return estimator


class StigClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Spectral Meta-Learning Classifier for real-time classifier selection of fit estimators.

    .. versionadded:: 2.5

    Parameters
    ----------
    estimators : list of (string, estimator) tuples
        List of independent estimators.
        "STIG, which is based on the spectral-meta learning approach from Parisi et al 2014,
        requires that the classifiers in the ensemble make independent errors. It's a common
        assumption with EEG that data from different subjects trained using the same
        classifier will produce independent errors. Similarly, different classifiers
        trained on the same data will produce independent errors (for the most part).
        So using either classifiers trained on different data or using different classifiers
        on the same data (or some combination of both) to build you're ensemble should
        work fine for STIG." He also said six is a good minimum number of estimators.

        Nick Waytowich - https://github.com/alexandrebarachant/pyRiemann/issues/46#issuecomment-276787910

    cov_estimator : string (default: 'scm')
        covariance matrix estimator. For regularization consider 'lwf' or 'oas'
        For a complete list of estimator, see `utils.covariance`.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for ``fit``.
        If -1, then the number of jobs is set to the number of cores.

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : array-like, shape = [n_predictions]
        The classes labels.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from pyriemann.ensemble import StigClassifier
    >>> clf1 = LogisticRegression(random_state=1)
    >>> clf2 = RandomForestClassifier(random_state=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = StigClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)])
    >>> eclf1 = eclf1.fit(X)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> eclf2 = StigClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)])
    >>> eclf2 = eclf2.fit(X)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = StigClassifier(estimators=[
    ...        ('lr', clf1), ('rf', clf2), ('gnb', clf3)])
    >>> eclf3 = eclf3.fit(X)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>>

    Refernces
    ---------

    [1] Waytowich NR, Lawhern VJ, Bohannon AW, Ball KR and Lance BJ (2016)
    Spectral Transfer Learning Using Information Geometry for a
    User-Independent Brain-Computer Interface. Front. Neurosci.
    10:430. doi: 10.3389/fnins.2016.00430

    """

    def __init__(self, estimators, cov_estimator='scm', n_jobs=1):
        self.classes_ = []
        self.cov_estimator = cov_estimator
        self.estimators = estimators
        self.le_ = None
        self.n_jobs = n_jobs
        self.named_estimators = dict(estimators)
        self.weights_ = None

        self.estimators_ = []

        for _, clf in self.estimators:
            self.estimators_.append(clf)

        if len(self.estimators_) == 0:
            raise ValueError('Must use at least one estimator')
        else:
            try:
                self.classes_ = self.estimators_[0].classes_
                if len(self.classes_) != 2:
                    raise ValueError('StigClassifier only works with 2 classes, input estimator has %i classes' % len(self.classes_))
            except AttributeError:
                try:
                    self.classes_ = self.estimators_[0].clf.classes_
                except AttributeError:
                    raise AttributeError('Estimator does not have property _classes. Fit classifier first.')
                except BaseException as e:
                    raise e

    def fit(self, X, y=None):
        """ Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """
        self._fit(X, y)
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        ----------
        sml : array-like, shape = [n_trials]
            Predicted class labels.
        """
        tmp_scores = self._collect_probas(X)

        if self.weights_ is not None:
            need_sum = self.weights_ * tmp_scores.T
        else:
            weight = (1.0 / len(self.estimators_)) * np.ones((1, len(self.estimators_)))

            need_sum = weight * tmp_scores.T

        scores_proba = np.sum(need_sum, axis=1)

        scores_predict = np.asarray([self.classes_[0] if score < 0.5 else self.classes_[1] for score in scores_proba])

        return scores_predict

    def _collect_predicts(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators_])

    def _collect_probas(self, X):
        """Collect results from clf.predict_probas calls. """
        return np.asarray([clf.predict_proba(X)[:, 1] for clf in self.estimators_])

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        sml : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        tmp_scores = self._collect_probas(X)

        if self.weights_ is not None:
            need_sum = self.weights_ * tmp_scores.T
        else:
            weight = (1.0 / len(self.estimators_)) * np.ones((1, len(self.estimators_)))

            need_sum = weight * tmp_scores.T

        scores_proba = np.sum(need_sum, axis=1)

        out = np.zeros((len(scores_proba), len(self.classes_)))

        index = 0
        for score in scores_proba:
            out[index][0] = 1. - score
            out[index][1] = score
            index += 1

        return out

    def transform(self, X):
        """Return probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        array-like = [n_classifiers, n_samples, n_classes]
            Class probabilities calculated by each classifier.
        """
        check_is_fitted(self, 'estimators_')
        return self._collect_probas(X)

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support"""
        if not deep:
            return super(StigClassifier, self).get_params(deep=False)
        else:
            out = super(StigClassifier, self).get_params(deep=False)
            out.update(self.named_estimators.copy())
            for name, step in six.iteritems(self.named_estimators):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out

    def _get_principal_eig(self, hard_preds):
        """
        Used to get the principal eigen vector for the given hard_preds

        Parameters
        ----------
        hard_preds : ndarray, dtype=int shape (len(self.estimators_), n_trials)
            ndarray of class labels for each x for each classifier.

        Returns
        -------
        v : ndarray, dtype=complex shape (len(self.estimators_),)
            The principal eigenvector for the covariance matrix calculated from hard_preds
        """
        est = _check_est(self.cov_estimator)
        Q = est(hard_preds)
        if Q.dtype.char in typecodes['AllFloat'] and not np.isfinite(Q).all():
            raise ValueError("Covariance matrices must be positive definite. Add regularization to avoid this error with cov_estimator='lwf'")
        _, vs = eigh(Q)
        return vs[:, -1]

    def _apply_sml(self, hard_preds):
        """ Apply sml to hard label predictions of each x for each classifier.
        Adapted from [1].

        Parameters
        ----------
        hard_preds : ndarray, dtype=int shape (len(self.estimators_), n_trials)
            ndarray of class labels for each x for each classifier.

        Returns
        ----------
        weight : array-like, shape = [1, n_trials]
            Weight for each classifier

        References
        ----------
        [1] N. Waytowich, github.com/nwayt001/Transfer_Learning_Project, "Code for
        designing and implementing transfer learning for machine learning applications", 2016
        """
        weight = self._get_principal_eig(hard_preds)
        weight /= np.sum(weight)
        return np.atleast_2d(weight)

    def _balanced_accuracy(self, pseudo_labels, true_labels):
        """

        :param pseudo_labels:
        :param true_labels:
        :return: tuple (psi, eta, pi)
            psi: nd.array shape: (pseudo_labels.shape[0],) sensitivity
            eta: nd.array shape: (pseudo_labels.shape[0],) specificity
            pi: nd.array shape: (pseudo_labels.shape[0],) balanced accuracy
        """
        nClfs, _ = pseudo_labels.shape
        psi = np.zeros((nClfs,))
        eta = np.zeros((nClfs,))
        pi = np.zeros((nClfs,))
        for i in range(0, nClfs):
            pos = pseudo_labels[i][true_labels == 1]
            if len(pos) > 0:
                psi[i] = np.mean(pos == 1)
            else:
                psi[i] = 0.

            neg = pseudo_labels[i][true_labels == 0]
            if len(neg) > 0:
                eta[i] = np.mean(neg == 0)
            else:
                eta[i] = 0.

            pi[i] = 0.5 * (psi[i] + eta[i])

        return psi, eta, pi

    def _estimation_maximization(self, hard_preds, pred_label, max_iters=100):
        q = 0

        converged = False

        k = len(pred_label)
        m = len(self.estimators_)
        # we consider the pred_label to be pred 0
        # q = 1
        prev_y = pred_label
        pi = np.zeros((m,))
        while not converged and q < max_iters:
            psi, eta, pi = self._balanced_accuracy(hard_preds,
                                                   prev_y)
            y = np.zeros((k,), dtype=int)
            for i in range(0, k):
                sum = 0.0
                for j in range(0, m):
                    if eta[j] > 0.999999:
                        eta[j] = 0.999999
                    if psi[j] > 0.999999:
                        psi[j] = 0.999999
                    if eta[j] < 0.000001:
                        eta[j] = 0.000001
                    if psi[j] < 0.000001:
                        psi[j] = 0.000001

                    log_1 = np.log((psi[j] * eta[j]) / ((1 - psi[j]) * (1 - eta[j])))
                    log_2 = np.log((psi[j] * (1 - psi[j])) / (eta[j] * (1 - eta[j])))
                    log_sum = log_1 + log_2
                    res = log_sum if hard_preds[j][i] > 0 else -1. * log_sum
                    sum += res
                y[i] = self.classes_[0] if np.sign(sum) < 0 else self.classes_[1]

            # Store current value
            # y_.append(y)

            # Converged?
            if np.array_equal(y, prev_y):
                converged = True
                break

            prev_y = y
            # Increment counter
            q += 1

        new_v = 2*pi - 1

        return new_v

    def _fit(self, X, y=None):
        hard_preds = self._collect_predicts(X)

        # Apply sml
        self.weights_ = self._apply_sml(hard_preds)

        # Estimation maximization
        scores_predict = self.predict(X)
        v_em = self._estimation_maximization(hard_preds=hard_preds,
                                             pred_label=scores_predict)

        v_em /= np.sum(v_em)

        self.weights_ = np.atleast_2d(v_em)

        return self
