"""
Ensemble Classificatiers

"""

# Authors: AJ Keller <aj@pushtheworld.us>
#
# License: BSD 3 clause
from __future__ import division

import numpy as np

from scipy.linalg import eig

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import has_fit_parameter, check_is_fitted


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
    sml_limit : int
        The number of trials you want to keep in the buffer.
    sml_threshold : int
        The minimum number of trials to have in the buffer before applying sml. Nick said 32...
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

    def __init__(self, estimators, sml_limit=200, sml_threshold=32, n_jobs=1):
        self.classes_ = []
        self.estimators = estimators
        self.le_ = None
        self.n_jobs = n_jobs
        self.named_estimators = dict(estimators)
        self.pred_labels_ = None
        self.sml_limit = sml_limit
        self.sml_threshold = sml_threshold
        self.tmp_scores_ = None
        self.tmp_hard_preds_ = None
        self.v_ = None
        self.weights_ = None
        self.x_ = None
        self.x_in = 0

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

    def fit(self, X, y):
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
        if check_is_fitted(self, 'estimators_') is None:
            self._init_arrays(X)
            self._fit(X, y)
        else:
            for clf in self.estimators_:
                clf.fit(X, y)
        return self

    def partial_fit(self, X, y=None):
        if self.x_ is None:
            self._init_arrays(X)
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
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilities calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_samples, n_classifiers]
            Class labels predicted by each classifier.
        """
        check_is_fitted(self, 'estimators_')
        return self._predict(X)

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

    def _append_x(self):
        pass

    def _get_principal_eig(self, hard_preds):
        Q = np.cov(hard_preds)
        if Q.size > 1:
            v, _ = eig(Q)
            return v
        else:
            return None

    def _extract_class_one_scores(self, tmp_scores):
        """ Extract just the active probability

        Parameters
        ----------
        tmp_scores : ndarray, shape (len(self.estimators_), n_trials, n_classes)
            ndarray predictions for each classifier of each epoch. n_classes is
            always 2 because this is a binary classifier

        Returns
        ----------
        indexes : ndarray, shape (len(self.estimators_), n_trials)
            Soft score for each clf for each trial
        """
        nClfs, nTrials, _ = tmp_scores.shape
        extracted_tmp_score = np.zeros((nClfs, nTrials))
        for i in range(nClfs):
            for j in range(nTrials):
                extracted_tmp_score[i][j] = tmp_scores[i][j][1]
        return extracted_tmp_score

    def _find_indexes_of_maxes(self, tmp_scores):
        """ Get the index of the max (farthest from 0.5) classifer for each
            trial, epoch, etc...

        Parameters
        ----------
        tmp_scores : ndarray, shape (len(self.estimators_), n_trials, n_classes)
            ndarray predictions for each classifier of each epoch. n_classes is
            always 2 because this is a binary classifier

        Returns
        ----------
        indexes : ndarray, shape (n_trials,)
            Soft score for each clf for each trial
        """
        nClfs, nTrials = tmp_scores.shape
        idx = np.zeros((nTrials,), dtype=int)
        for i in range(0, nTrials):
            val = tmp_scores[:, i].view()
            upper_dist = val.max() - 0.5
            lower_dist = 0.5 - val.min()
            if upper_dist > lower_dist:
                idx[i] = np.argmax(val)
            else:
                idx[i] = np.argmin(val)
        return idx

    def _get_ensemble_scores_labels(self, tmp_scores, indexes):
        """ Get the (farthest from 0.5) score and label for each trial, epoch, etc...

        Parameters
        ----------
        tmp_scores : ndarray, shape (len(self.estimators_), n_trials, n_classes)
            ndarray predictions for each classifier of each epoch. n_classes is
            always 2 because this is a binary classifier
        indexes : ndarray, shape (n_trials,)
            Soft score for each clf for each trial

        Returns
        ----------
        ensemble_scores: ndarray, shape (n_trials,) dtype=float
            Soft scores of the max classifier for each trial
        ensemble_labels: ndarray, shape (n_trials,) dtype=int
            Hard scores (labels) of the max classifier for each trial
        """
        _, n_trials = tmp_scores.shape
        ensemble_scores = np.zeros((n_trials,), dtype=float)
        ensemble_labels = np.zeros((n_trials,), dtype=int)
        for i in range(n_trials):
            ind = indexes[i]
            ensemble_scores[i] = tmp_scores[ind][i]
            ensemble_labels[i] = int(self.classes_[0]) if tmp_scores[ind][i] < 0.5 else int(self.classes_[1])
        return ensemble_scores, ensemble_labels

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
        # These lines remove classifiers who make only positive predictions or
        # only negative predictions; it causes issues with the
        # eigendecomposition if they are not removed; we assign a weighting of
        # zero to these classifiers
        nClfs = len(self.estimators_)
        indexes = np.ones((nClfs,), dtype=bool)
        for i in range(0, nClfs):
            indexes[i] = np.unique(hard_preds[i]).size != 1

        if np.sum(indexes) == 1:
            indexes = np.ones((nClfs,), dtype=bool)

        v = self._get_principal_eig(hard_preds[indexes])

        weight = np.zeros((nClfs,))
        weight[indexes] = np.real(v)
        weight[np.logical_not(indexes)] = 1.0 / nClfs
        weight /= np.sum(weight)

        n_weight = np.zeros((1, nClfs))

        n_weight[:] = weight[:]

        return n_weight

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
            tn, fp, fn, tp = confusion_matrix(true_labels, pseudo_labels[i], labels=[0, 1]).astype(float).ravel()

            psi[i] = tp / (fp + tp) if (fp + tp) > 0 else 0.
            eta[i] = tn / (tn + fn) if (tn + fn) > 0 else 0.

            # There seems to be a lac of documentation but cnf_mat returns
            #   np.ndarray:
            #       ---------
            #      | TN | FP |
            #       ---------
            #      | FN | TP |
            #       ---------
            # psi[i] = precision_score(true_labels, pseudo_labels[i], labels=[0,1])
            # if cnf_mat[0][0] + cnf_mat[1][0] < 0.1:
            #     eta[i] = 0.
            # else:
            #     eta[i] = cnf_mat[0][0] / (cnf_mat[0][0] + cnf_mat[1][0])
            pi[i] = 0.5 * (psi[i] + eta[i])

        return psi, eta, pi

    def _estimation_maximization(self, principal_eig_v, hard_preds, pred_label, max_iters=100):
        q = 0
        y_ = [pred_label]

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
                if psi.max() > 0.9999:
                    indx = np.argmax(psi)
                    y[i] = hard_preds[indx][i]
                elif eta.max() > 0.9999:
                    indx = np.argmax(eta)
                    y[i] = hard_preds[indx][i]
                else:
                    for j in range(0, m):
                        if eta[j] < 0.001:
                            continue
                        log_1 = np.log((psi[j] * eta[j]) / ((1 - psi[j]) * (1 - eta[j])))
                        log_2 = np.log((psi[j] * (1 - psi[j])) / (eta[j] * (1 - eta[j])))
                        log_sum = log_1 + log_2
                        res = hard_preds[j][i] * log_sum if hard_preds[j][i] > 0 else -1. * log_sum
                        sum += res
                    y[i] = self.classes_[0] if np.sign(sum) < 0 else self.classes_[1]

            # Store current value
            y_.append(y)

            # Converged?
            if np.array_equal(y_[q], y_[q + 1]):
                converged = True
                break

            prev_y = y
            # Increment counter
            q += 1

        new_v = 2*pi - 1

        return new_v

    def _init_arrays(self, X):
        """ Initialize internal arrays to size dependent on input data

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        """
        self.x_in = 0
        new_shape = X[0].shape
        new_shape = (self.sml_limit,) + new_shape
        self.x_ = np.zeros(new_shape)
        self.pred_label_ = np.zeros((self.sml_limit,), dtype=int)
        self.pred_labels_ = np.zeros((self.sml_limit,), dtype=int)
        self.tmp_scores_ = np.zeros((len(self.estimators_), self.sml_limit,))
        self.tmp_hard_preds_ = np.zeros((self.sml_limit, len(self.estimators_),), dtype=int)
        self.weights_ = None

    def _fit(self, X, y=None):
        # Append the new x
        n = X.shape[0]
        if n > self.sml_limit:
            raise ValueError('Number of X must be less then or equal to %d got %d trials'
                             % (n, self.sml_limit))
        else:
            self.x_in = self.x_in + n
            self.x_in = self.x_in if self.x_in < self.sml_limit else self.sml_limit
            if self.sml_limit <= n:
                X = X[-self.sml_limit:]
                n = len(X)
            else:
                self.x_[:-n] = self.x_[n:]
            self.x_[-n:] = X

        # Shift array by n
        self.tmp_scores_[:, :-n] = self.tmp_scores_[:, n:]

        tmp_scores = self._collect_probas(self.x_[-n:])

        self.tmp_scores_[:, -n:] = tmp_scores

        if self.x_in > self.sml_threshold:
            if self.weights_ is None:
                n = self.x_in

            # Shift array by n
            self.pred_labels_[:-n] = self.pred_labels_[n:]
            self.tmp_hard_preds_[:-n] = self.tmp_hard_preds_[n:]

            indexes = self._find_indexes_of_maxes(self.tmp_scores_[:, -n:])
            ensemble_scores, ensemble_labels = self._get_ensemble_scores_labels(self.tmp_scores_[:, -n:], indexes)
            hard_preds = self._collect_predicts(self.x_[-n:])

            self.pred_labels_[-n:] = ensemble_labels
            self.tmp_hard_preds_[-n:] = hard_preds.T

            # Apply sml
            self.weights_ = self._apply_sml(self.tmp_hard_preds_[-self.x_in:].T)

            # Need to implement estimation maximization
            """
            scores_predict = self.predict(self.x_[-n:])
            self.pred_label_[:-n] = self.pred_label_[n:]
            self.pred_label_[-n:] = scores_predict
            v = self._get_principal_eig(hard_preds)
            v_em = self._estimation_maximization(principal_eig_v=v,
                                                 hard_preds=self.tmp_hard_preds_[-self.x_in:].T,
                                                 pred_label=self.pred_label_[-self.x_in:])

            v_em /= np.sum(v_em)
            """
        return self
