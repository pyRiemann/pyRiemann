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
from sklearn.metrics import confusion_matrix
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
    """Soft Voting/Majority Rule classifier for unfitted estimators.

    .. versionadded:: 2.5

    Read more in the :ref:`User Guide <voting_classifier>`.

    Parameters
    ----------
    estimators : list of (string, estimator) tuples
        Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        `self.estimators_`.

    voting : str, {'hard', 'soft'} (default='hard')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.

    weights : array-like, shape = [n_classifiers], optional (default=`None`)
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.

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
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> eclf2 = StigClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...         voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = StigClassifier(estimators=[
    ...        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...        voting='soft', weights=[2,1,1])
    >>> eclf3 = eclf3.fit(X, y)
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

    def __init__(self, estimators, weights=None, n_jobs=1, sml_limit=200):
        self.estimators = estimators
        self.n_jobs = n_jobs
        self.named_estimators = dict(estimators)
        self.pred_labels_ = None
        self.sml_limit = sml_limit
        self.sml_threshold = len(estimators)
        self.tmp_scores_ = None
        self.tmp_hard_preds_ = None
        self.v_ = None
        self.weights_ = weights
        self.x_ = None
        self.x_in = 0

        check_is_fitted(self, 'estimators')

        self.estimators_ = []

        for _, clf in self.estimators:
            self.estimators_.append(clf)

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
        self._predict(X)

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        ----------
        maj : array-like, shape = [n_trials]
            Predicted class labels.
        """
        return self._predict(X)
        # predictions = self._predict(X)
        # maj = np.apply_along_axis(lambda x:
        #                           np.argmax(np.bincount(x,
        #                                     weights=self.weights_)),
        #                           axis=1,
        #                           arr=predictions.astype('int'))
        #
        # maj = self.le_.inverse_transform(maj)
        #
        # return maj

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        out = []
        nt, _, _ = X.shape

        for i in range(0, nt):
            preds = []
            for clf in self.estimators_:
                preds.append(clf.predict_proba(np.array([X[i]]))[0])
            out.append(np.array(preds))

        return np.array(out)
        # return np.asarray([clf.predict_proba(X) for clf in self.estimators_])

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        check_is_fitted(self, 'estimators_')
        avg = np.average(self._collect_probas(X), axis=0, weights=self.weights)
        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        return self._predict_proba

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
        v, _ = eig(Q)
        return v

    def _apply_sml(self, hard_preds):
        # need to make an [n_clf, self.x_in_]

        # These lines remove classifiers who make only positive predictions or
        # only negative predictions; it causes issues with the
        # eigendecomposition if they are not removed; we assign a weighting of
        # zero to these classifiers
        nClfs = len(self.estimators_)
        indexes = np.ones((nClfs,), dtype=bool)
        for i in range(0, nClfs):
            indexes[i] = np.unique(hard_preds[i]).size != 1

        Q = np.cov(hard_preds[indexes])

        # solve for v, the principal eigen-vector of Q
        v, _ = eig(Q)

        weight = np.zeros((nClfs,))
        weight[indexes] = np.real(v)
        weight[np.logical_not(indexes)] = 1.0 / nClfs
        weight /= np.sum(weight)

        n_weight = np.zeros((1, nClfs))

        n_weight[:] = weight[:]

        return n_weight

    def _balenced_accuracy(self, pseudo_labels, true_labels):
        psi = []
        eta = []
        pi = []
        index = true_labels == 1
        for i in range(0, len(self.estimators_)):
            _psi, _eta = 0.0, 0.0
            cnf_mat = confusion_matrix(true_labels, pseudo_labels[i]).astype(float)
            _psi = cnf_mat[1][1] / (cnf_mat[1][1] + cnf_mat[0][1])
            _eta = cnf_mat[0][0] / (cnf_mat[0][0] + cnf_mat[1][0])
            _pi = 0.5 * (_psi + _eta)
            psi.append(_psi)
            eta.append(_eta)
            pi.append(_pi)

        return np.array(psi), np.array(eta), np.array(pi)

    def _estimation_maximization(self, principal_eig_v, hard_preds, pred_label, max_iters=100):
        q = 0
        y_ = [pred_label]

        converged = False

        k = len(pred_label)
        m = len(self.estimators_)
        # we consider the pred_label to be pred 0
        q = 1
        prev_y = pred_label
        pi = np.zeros((m,))
        while not converged and q < max_iters:
            psi, eta, pi = self._balenced_accuracy(hard_preds,
                                                   prev_y)
            y = np.zeros((k,), dtype=int)
            for i in range(0, k):
                sum = 0.0
                for j in range(0, m):
                    log_1 = np.log((psi[j] * eta[j]) / ((1 - psi[j]) * (1 - eta[j])))
                    log_2 = np.log((psi[j] * (1 - psi[j])) / (eta[j] * (1 - eta[j])))
                    log_sum = log_1 + log_2
                    res = hard_preds[j][i] * log_sum
                    sum += res
                y[i] = np.sign(sum)

            # Converged?
            q += 1

        new_v = 2*pi - 1

        return new_v

    def _predict(self, X):
        """Collect results from clf.predict calls. """

        if self.x_ is None:
            new_shape = X[0].shape
            new_shape = (self.sml_limit,) + new_shape
            self.x_ = np.zeros(new_shape)
            self.pred_label_ = np.zeros((self.sml_limit,), dtype=int)
            self.pred_labels_ = np.zeros((self.sml_limit,), dtype=int)
            self.tmp_scores_ = np.zeros((len(self.estimators_), self.sml_limit,))
            self.tmp_hard_preds_ = np.zeros((self.sml_limit, len(self.estimators_),), dtype=int)

        # Append the new x
        n, _, _ = X.shape
        if n > self.sml_limit:
            raise ValueError('Number of X must be less then or equal to '
                             '; got %d trials'
                             % n)
        else:
            self.x_in = self.x_in + n
            self.x_in = self.x_in if self.x_in < self.sml_limit else self.sml_limit
            if self.sml_limit <= n:
                X = X[-self.sml_limit:]
                n = len(X)
            else:
                self.x_[:-n] = self.x_[n:]
            self.x_[-n:] = X

        # loop through X and get probas
        tmp_score = self._collect_probas(self.x_[-n:])
        # tmp_score shape is (nb_estimators, self.x_in, 2) 2 because this is a binary classifier
        # get second column of first classifier tmp_score[0][:][:, 1]
        idx = []
        temp_tmp_score = np.zeros((n, len(self.estimators_)))
        for i in range(0, n):
            vals = tmp_score[i][:][:, 1]
            temp_tmp_score[i] = vals
            upper_dist = vals.max() - 0.5
            lower_dist = 0.5 - vals.min()
            if upper_dist > lower_dist:
                idx.append(np.argmin(vals))
            else:
                idx.append(np.argmax(vals))

        # idx is [nEpochs]

        self.tmp_scores_[:-n] = self.tmp_scores_[n:]
        self.tmp_scores_[:, -n:] = temp_tmp_score.T

        # get the largest score
        # In python land, we get index of abs(pred[1] - 0.5), where pred = clf.pred_proba() for each epoch
        # we want the index of the classifier with the largest distance from 0.5 for each epoch
        ensemble_score = []
        k = 0
        for index in idx:
            ensemble_score.append(tmp_score[k][index][1])

        # ensemble_score is [nEpochs, 1]
        # for each score, calculate score - 0.5
        ensemble_label = []
        for score in ensemble_score:
            sign = score - 0.5
            class_label = 0 if sign < 0 else 1
            ensemble_label.append(class_label)
        # ensemble_label is [nEpochs, 1]
        ensemble_label = np.array(ensemble_label)

        self.pred_labels_[:-n] = self.pred_labels_[n:]
        self.pred_labels_[-n:] = ensemble_label

        hard_preds = np.zeros((len(self.estimators_), n,), dtype=int)
        for i in range(0, len(self.estimators_)):
            for j in range(0, n):
                hard_preds[i][j] = 0 if tmp_score[j][i][0] > 0.5 else 1

        self.tmp_hard_preds_[:-n] = self.tmp_hard_preds_[n:]
        self.tmp_hard_preds_[-n:] = hard_preds.T

        if self.x_in > self.sml_threshold:
            # Apply sml
            sml_weight = self._apply_sml(self.tmp_hard_preds_[-self.x_in:].T)

            need_sum = sml_weight * temp_tmp_score
        else:
            weight = (1.0 / len(self.estimators_)) * np.ones((1, len(self.estimators_)))

            need_sum = weight * temp_tmp_score

        score = np.sum(need_sum, axis=1)

        score_label = []
        for scr in score:
            class_label = 0 if scr <= 0.5 else 1
            score_label.append(class_label)

        self.pred_label_[:-n] = self.pred_label_[n:]
        self.pred_label_[-n:] = np.array(score_label)

        if self.x_in >= self.sml_threshold and self.v_ is None:
            v = self._get_principal_eig(hard_preds)
            self.v_ = self._estimation_maximization(principal_eig_v=v,
                                                    hard_preds=self.tmp_hard_preds_[-self.x_in:].T,
                                                    pred_label=self.pred_label_[-self.x_in:])

        return score
