"""Testing for the Ensemble"""

import numpy as np
from sklearn.utils.testing import assert_almost_equal, assert_array_equal
from sklearn.utils.testing import assert_equal, assert_true, assert_false
from sklearn.utils.testing import assert_raise_message
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_multilabel_classification
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

from pyriemann.ensemble import StigClassifier
from pyriemann.classification import (MDM, FgMDM, KNearestNeighbor,
                                      TSclassifier)


def generate_cov(Nt, Ne):
    """Generate a set of cavariances matrices for test purpose."""
    rs = np.random.RandomState(1234)
    diags = 2.0 + 0.1 * rs.randn(Nt, Ne)
    A = 2*rs.rand(Ne, Ne) - 1
    A /= np.atleast_2d(np.sqrt(np.sum(A**2, 1))).T
    covmats = np.empty((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.dot(np.dot(A, np.diag(diags[i])), A.T)
    return covmats


def generate_points(Nt=100):
    """Generate a set clusters."""
    rs = np.random.RandomState(1234)
    point_list = []
    label_list = []
    for i in range(0, Nt):
        point_x1 = rs.rand() * 12 - 3
        point_x2 = rs.rand() * 6 - 4
        point_y1 = rs.rand() * 12 - 2
        point_y2 = rs.rand() * 6 - 5
        point_1 = [point_x1, point_x2]
        point_2 = [point_y1, point_y2]
        point_list.append(point_1)
        point_list.append(point_2)
        label_list.append(0)
        label_list.append(1)

    return np.array(point_list), np.array(label_list)


# Load the iris dataset and randomly permute it
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
covset = generate_cov(100, 3)
labels = np.array([0, 1]).repeat(50)
mdm = MDM(metric='riemann')
mdm.fit(generate_cov(100, 3), labels)
estimators = []
for i in range(0, 10):
    _covset = covset + ((i + 1) * 0.01)
    estimators.append(('mdm%i' % i, MDM(metric='riemann').fit(_covset, labels)))


estimators_lr = []
for i in range(0, 10):
    num_points = i * 10 + 10
    point_list, label_list = generate_points(num_points)
    estimators_lr.append(('lr%i' % i, LogisticRegression().fit(point_list, label_list)))


def test_estimator_init():
    eclf = StigClassifier(estimators=[])
    assert_equal(eclf.x_, None)

    msg = ('Invalid `estimators` attribute, `estimators` should be'
           ' a list of (string, estimator) tuples')
    assert_raise_message(AttributeError, msg, eclf.fit, X, y)

    clf = LogisticRegression(random_state=1)

    eclf = StigClassifier(estimators=[('lr', clf)], weights=[1, 2])
    msg = ('Number of classifiers and weights must be equal'
           '; got 2 weights, 1 estimators')
    assert_raise_message(ValueError, msg, eclf.fit, X, y)


def test_balanced_accuracy():
    nTrials = 10
    nClfs = 1
    eclf = StigClassifier(estimators=[])

    # all TP
    pseudo_labels = np.ones((nClfs, nTrials), dtype=int)
    true_labels = np.ones((nTrials,), dtype=int)

    psi, eta, pi = eclf._balanced_accuracy(pseudo_labels, true_labels)

    assert_equal(psi[0], 1, "psy should be 1.")
    assert_equal(eta[0], 0, "eta should be 0.")
    assert_equal(pi[0], 0.5, "pi should be 0.")

    # all TN
    pseudo_labels = np.zeros((nClfs, nTrials), dtype=int)
    true_labels = np.zeros((nTrials,), dtype=int)

    psi, eta, pi = eclf._balanced_accuracy(pseudo_labels, true_labels)

    assert_equal(eta[0], 1.0, "eta should be 1.")
    assert_equal(psi[0], 0., "psi should be 1.")
    assert_equal(pi[0], 0.5, "pi should be 1.")

    # all FP
    pseudo_labels = np.ones((nClfs, nTrials), dtype=int)
    true_labels = np.ones((nTrials,), dtype=int)

    pseudo_labels[0] = 0

    psi, eta, pi = eclf._balanced_accuracy(pseudo_labels, true_labels)

    assert_equal(eta[0], 0.0, "eta should be 0.0 but was %f" % eta[0])
    assert_equal(psi[0], 0.0, "psi should be 0.0 but was %f" % eta[0])
    assert_equal(pi[0], 0.0, "pi should be 0.0 but was %f" % eta[0])

    # all FN
    pseudo_labels = np.ones((nClfs, nTrials), dtype=int)
    true_labels = np.zeros((nTrials,), dtype=int)

    psi, eta, pi = eclf._balanced_accuracy(pseudo_labels, true_labels)

    assert_equal(psi[0], 0.0, "psi should be 0.0 but was %f" % eta[0])
    assert_equal(eta[0], 0.0, "eta should be 0.0 but was %f" % eta[0])
    assert_equal(pi[0], 0.0, "pi should be 0.0 but was %f" % eta[0])

    # onr fn
    pseudo_labels = np.ones((nClfs, nTrials), dtype=int)
    true_labels = np.ones((nTrials,), dtype=int)

    pseudo_labels[0][0] = 0

    psi, eta, pi = eclf._balanced_accuracy(pseudo_labels, true_labels)

    assert_equal(psi[0], 1, "psy should be 1.")
    assert_equal(eta[0], 0.0, "eta should be 0.")
    assert_equal(pi[0], 0.5, "pi should be 0.5")

    # one fp
    pseudo_labels = np.ones((nClfs, nTrials), dtype=int)
    true_labels = np.ones((nTrials,), dtype=int)

    true_labels[0] = 0

    psi, eta, pi = eclf._balanced_accuracy(pseudo_labels, true_labels)

    assert_equal(psi[0], 0.9, "psy should be 1.")
    assert_equal(eta[0], 0.0, "eta should be 0.")
    assert_equal(pi[0], 0.45, "pi should be 1.")


def test_predict():
    expected_sml_limit = 50

    eclf = StigClassifier(estimators=estimators_lr,
                          sml_limit=expected_sml_limit)

    point_list, label_list = generate_points(num_points)

    expected_shape = (expected_sml_limit,) + point_list[0].shape

    # when num x is less then estimators or `sml_threshold`
    num_xs = len(estimators) - 2
    actual_preds = eclf.predict(point_list[:num_xs])
    assert_equal(actual_preds.shape[0], np.ones((num_xs,)).shape[0], "should make internal x_ a buffer of len sml_limit")

    # too good converges fast.
    num_xs_1 = len(estimators)
    actual_preds = eclf.predict(point_list[num_xs:num_xs + 40])
    assert_equal(actual_preds.shape[0], np.ones((num_xs_1,)).shape[0],
                 "should make internal x_ a buffer of len sml_limit")

    estimators_bad = []
    for i in range(0, 10):
        _covset = covset + (i * 0.02 * (1 if i % 2 == 0 else -1))
        estimators_bad.append(('mdm%i' % i, MDM(metric='riemann').fit(_covset, labels)))

    eclf_bad = StigClassifier(estimators=estimators_bad,
                              sml_limit=expected_sml_limit)

    # worse data not fast converge
    num_xs_1 = len(estimators)
    actual_preds = eclf_bad.predict(covset[10 + num_xs:10 + num_xs + 40])
    assert_equal(actual_preds.shape[0], np.ones((num_xs_1,)).shape[0],
                 "should make internal x_ a buffer of len sml_limit")

def test_majority_label_iris():
    """Check classification by majority label on dataset iris."""
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    clf3 = GaussianNB()
    eclf = StigClassifier(estimators=[
                ('lr', clf1), ('rf', clf2), ('gnb', clf3)])
    scores = cross_val_score(eclf, X, y, cv=5, scoring='accuracy')
    assert_almost_equal(scores.mean(), 0.95, decimal=2)


def test_tie_situation():
    """Check voting classifier selects smaller class label in tie situation."""
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    eclf = StigClassifier(estimators=[('lr', clf1), ('rf', clf2)])
    assert_equal(clf1.fit(X, y).predict(X)[73], 2)
    assert_equal(clf2.fit(X, y).predict(X)[73], 1)
    assert_equal(eclf.fit(X, y).predict(X)[73], 1)


def test_weights_iris():
    """Check classification by average probabilities on dataset iris."""
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    clf3 = GaussianNB()
    eclf = StigClassifier(estimators=[
                            ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                            weights=[1, 2, 10])
    scores = cross_val_score(eclf, X, y, cv=5, scoring='accuracy')
    assert_almost_equal(scores.mean(), 0.93, decimal=2)

def test_multilabel():
    """Check if error is raised for multilabel classification."""
    X, y = make_multilabel_classification(n_classes=2, n_labels=1,
                                          allow_unlabeled=False,
                                          random_state=123)
    clf = OneVsRestClassifier(SVC(kernel='linear'))

    eclf = StigClassifier(estimators=[('ovr', clf)])

    try:
        eclf.fit(X, y)
    except NotImplementedError:
        return


def test_parallel_predict():
    """Check parallel backend of StigClassifier on toy dataset."""
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    clf3 = GaussianNB()
    X = np.array([[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
    y = np.array([1, 1, 2, 2])

    eclf1 = StigClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
        n_jobs=1).fit(X, y)
    eclf2 = StigClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
        n_jobs=2).fit(X, y)

    assert_array_equal(eclf1.predict(X), eclf2.predict(X))
