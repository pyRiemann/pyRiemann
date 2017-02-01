"""Testing for the Ensemble"""

import numpy as np
from sklearn.utils.testing import assert_almost_equal, assert_array_equal
from sklearn.utils.testing import assert_equal
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


def generate_cov(Nt, Ne, rs=0):
    """Generate a set of cavariances matrices for test purpose."""
    rs = np.random.RandomState(1234 + rs)
    diags = 2.0 + 0.1 * rs.randn(Nt, Ne)
    A = 2*rs.rand(Ne, Ne) - 1
    A /= np.atleast_2d(np.sqrt(np.sum(A**2, 1))).T
    covmats = np.empty((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.dot(np.dot(A, np.diag(diags[i])), A.T)
    return covmats

# Load the iris dataset and randomly permute it
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
covset = generate_cov(100, 3)
labels = np.array([0, 1]).repeat(50)
mdm = MDM(metric='riemann')
mdm.fit(generate_cov(100, 3), labels)
estimators = []
for i in range(0, 10):
    estimators.append(('mdm%i' % i, MDM(metric='riemann').fit(generate_cov(100, 3, rs=i), labels)))


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


def test_predict():
    expected_sml_limit = 50

    eclf = StigClassifier(estimators=estimators,
                          sml_limit=expected_sml_limit)

    expected_shape = (expected_sml_limit,) + covset[0].shape

    # when num x is less then estimators or `sml_threshold`
    num_xs = len(estimators) - 2
    actual_preds = eclf.predict(covset[:num_xs])
    assert_equal(actual_preds.shape[0], np.ones((num_xs,)).shape[0], "should make internal x_ a buffer of len sml_limit")

    # when num x is less then estimators or `sml_threshold`
    num_xs_1 = len(estimators)
    actual_preds = eclf.predict(covset[num_xs:num_xs + num_xs_1])
    assert_equal(actual_preds.shape[0], np.ones((num_xs_1,)).shape[0], "should make internal x_ a buffer of len sml_limit")








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


def test_predict_on_toy_problem():
    """Manually check predicted class labels for toy dataset."""
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    clf3 = GaussianNB()

    X = np.array([[-1.1, -1.5],
                  [-1.2, -1.4],
                  [-3.4, -2.2],
                  [1.1, 1.2],
                  [2.1, 1.4],
                  [3.1, 2.3]])

    y = np.array([1, 1, 1, 2, 2, 2])

    assert_equal(all(clf1.fit(X, y).predict(X)), all([1, 1, 1, 2, 2, 2]))
    assert_equal(all(clf2.fit(X, y).predict(X)), all([1, 1, 1, 2, 2, 2]))
    assert_equal(all(clf3.fit(X, y).predict(X)), all([1, 1, 1, 2, 2, 2]))

    eclf = StigClassifier(estimators=[
                            ('lr', clf1), ('rf', clf2), ('gnb', clf3)])
    assert_equal(all(eclf.fit(X, y).predict(X)), all([1, 1, 1, 2, 2, 2]))

    eclf = StigClassifier(estimators=[
                            ('lr', clf1), ('rf', clf2), ('gnb', clf3)])
    assert_equal(all(eclf.fit(X, y).predict(X)), all([1, 1, 1, 2, 2, 2]))


def test_predict_proba_on_toy_problem():
    """Calculate predicted probabilities on toy dataset."""
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    clf3 = GaussianNB()
    X = np.array([[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
    y = np.array([1, 1, 2, 2])

    clf1_res = np.array([[0.59790391, 0.40209609],
                         [0.57622162, 0.42377838],
                         [0.50728456, 0.49271544],
                         [0.40241774, 0.59758226]])

    clf2_res = np.array([[0.8, 0.2],
                         [0.8, 0.2],
                         [0.2, 0.8],
                         [0.3, 0.7]])

    clf3_res = np.array([[0.9985082, 0.0014918],
                         [0.99845843, 0.00154157],
                         [0., 1.],
                         [0., 1.]])

    t00 = (2*clf1_res[0][0] + clf2_res[0][0] + clf3_res[0][0]) / 4
    t11 = (2*clf1_res[1][1] + clf2_res[1][1] + clf3_res[1][1]) / 4
    t21 = (2*clf1_res[2][1] + clf2_res[2][1] + clf3_res[2][1]) / 4
    t31 = (2*clf1_res[3][1] + clf2_res[3][1] + clf3_res[3][1]) / 4

    eclf = StigClassifier(estimators=[
                            ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                            weights=[2, 1, 1])
    eclf_res = eclf.fit(X, y).predict_proba(X)

    assert_almost_equal(t00, eclf_res[0][0], decimal=1)
    assert_almost_equal(t11, eclf_res[1][1], decimal=1)
    assert_almost_equal(t21, eclf_res[2][1], decimal=1)
    assert_almost_equal(t31, eclf_res[3][1], decimal=1)

    try:
        eclf = StigClassifier(estimators=[
                                ('lr', clf1), ('rf', clf2), ('gnb', clf3)])
        eclf.fit(X, y).predict_proba(X)

    except AttributeError:
        pass
    else:
        raise AssertionError('AttributeError for voting == "hard"'
                             ' and with predict_proba not raised')


def test_multilabel():
    """Check if error is raised for multilabel classification."""
    X, y = make_multilabel_classification(n_classes=2, n_labels=1,
                                          allow_unlabeled=False,
                                          random_state=123)
    clf = OneVsRestClassifier(SVC(kernel='linear'))

    eclf = StigClassifier(estimators=[('ovr', clf)], voting='hard')

    try:
        eclf.fit(X, y)
    except NotImplementedError:
        return


def test_gridsearch():
    """Check GridSearch support."""
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    eclf = StigClassifier(estimators=[
                ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                voting='soft')

    params = {'lr__C': [1.0, 100.0],
              'voting': ['soft', 'hard'],
              'weights': [[0.5, 0.5, 0.5], [1.0, 0.5, 0.5]]}

    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
    grid.fit(iris.data, iris.target)


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


def test_estimator_weights_format():
    # Test estimator weights inputs as list and array
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    eclf1 = StigClassifier(estimators=[
                ('lr', clf1), ('rf', clf2)],
                weights=[1, 2])
    eclf2 = StigClassifier(estimators=[
                ('lr', clf1), ('rf', clf2)],
                weights=np.array((1, 2)))
    eclf1.fit(X, y)
    eclf2.fit(X, y)
    assert_array_equal(eclf1.predict_proba(X), eclf2.predict_proba(X))