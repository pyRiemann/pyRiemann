"""
Ensemble Classificatiers

"""

# Authors: AJ Keller <aj@pushtheworld.us>
#
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
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
        self.sml_limit = sml_limit
        self.sml_threshold = len(estimators)
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
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_])

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

    def _predict(self, X):
        """Collect results from clf.predict calls. """

        if self.x_ is None:
            new_shape = X[0].shape
            new_shape = (self.sml_limit,) + new_shape
            self.x_ = np.zeros(new_shape)

        # Append the new x
        n, _, _ = X.shape
        if n > self.sml_limit:
            pass
        else:
            self.x_in = self.x_in + n
            self.x_in = self.x_in if self.x_in < self.sml_limit else self.sml_limit
            self.x_[:-n] = self.x_[n:]
            for i in range(1, n + 1):
                self.x_[-i] = X[i - 1]

        # loop through X and get probas
        tmpScore = self._collect_probas(self.x_[-self.x_in:])
        # tmpScore shape is (nb_estimators, self.x_in, 2) 2 because this is a binary classifier
        # get second column of first classifier tmpScore[0][:][:, 1]
        idx = []
        for i in range(0, self.x_in):
            vals = tmpScore[i][:][:, 1]
            upper_dist = vals.max() - 0.5
            lower_dist = 0.5 - vals.min()
            if upper_dist > lower_dist:
                idx.append(np.argmin(vals))
            else:
                idx.append(np.argmax(vals))


        # We want to do index abs(pred[1] - 0.5)
        # we want the index of the classifier with the largest distance from 0.5 for each epoch
        print 'hi'
        # return np.asarray([clf.predict(X) for clf in self.estimators]).T