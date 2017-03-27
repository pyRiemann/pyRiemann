"""Testing for the Ensemble"""

import numpy as np
import mock
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


def generate_points(r=3, Nt=100):
    """Generate a set clusters."""
    rs = np.random.RandomState(1234)
    point_list = []
    label_list = []
    for i in range(0, Nt):
        point_x1 = rs.rand() * (4 * r) - r
        point_x2 = rs.rand() * (2 * r) - (r + 1)
        point_y1 = rs.rand() * (4 * r) - (r - 1)
        point_y2 = rs.rand() * (2 * r) - (r + 2)
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


def generate_random_covariances(N=10, K=20, snr=1, random_state=42, alpha=1e-6,
                                return_signal=False, effect_size=1, L=1):
    """

    :param N:
    :param K:
    :param snr:
    :param random_state:
    :param alpha:
    :param return_signal:
    :param effect_size: int
        As effect inincreases, clf acc goes up, discriminatory info in covs goes up
    :param L:
    :return:
    """
    """Generate random covariance matrices according to marcos white paper"""
    rs = np.random.RandomState(random_state)

    # scale sd in order to have signal + noise = 1
    # this keep the ratio signal / noise = snr
    noise_sd = (1. / (1 + snr))
    signal_sd = 1 - noise_sd
    # signal_sd = snr

    ########## signal
    # generate mixin matrix for the signal
    # check if it is inversible (it usually is)
    A = 2 * rs.rand(N, N) - 1
    while np.linalg.det(A) < 1e-9:
        A = 2 * rs.rand(N, N) - 1

    A /= np.atleast_2d(np.sqrt(np.sum(A ** 2, 1))).T
    # generate permutation
    perms = rs.permutation(N)

    labels = np.array([0] * K + [1] * K)
    # draw random diagonal elements
    # L = 10

    diags = rs.randn(N, 2 * K)
    diags = np.exp(diags)
    diags[L] += labels * effect_size
    diags *= np.atleast_2d([0.5 ** p for p in range(1, N + 1)]).T

    # diags = diags[perms]

    # Mix and permute cov mats
    Csignal = np.zeros((2 * K, N, N))
    for k in range(2 * K):
        Csignal[k] = np.dot(np.dot(A, np.diag(diags[:, k])), A.T)

    ############ noise

    Cnoise = np.zeros((2 * K, N, N))
    diags_noise = rs.randn(N, 2 * K)
    diags_noise = np.exp(diags_noise)
    diags_noise *= np.atleast_2d([0.5 ** p for p in range(1, N + 1)]).T

    for k in range(2 * K):
        # generate random mixin matrix
        An = 2 * rs.rand(N, N) - 1
        while np.linalg.det(An) < 1e-9:
            An = 2 * rs.rand(N, N) - 1
        # generate random permutation
        An /= np.atleast_2d(np.sqrt(np.sum(An ** 2, 1))).T
        Noise_k = np.dot(np.dot(An, np.diag(diags_noise[:, k])), An.T)
        # Noise_k = An + An.T
        # noise_sd = np.trace(Noise_k)/np.trace(Csignal[k])
        # alpha_k = alpha * np.trace(Csignal[k]) / (N)
        # mix
        Cnoise[k] = Noise_k

    trs = np.trace(np.sum(Csignal, 0))
    trn = np.trace(np.sum(Cnoise, 0))

    C = signal_sd * Csignal + noise_sd * Cnoise + alpha * np.eye(N)

    if return_signal:
        return C, diags, A, labels
    else:
        return C, labels


def test_estimator_init():

    clf = LogisticRegression(random_state=1).fit(X, y)
    msg = 'StigClassifier only works with 2 classes, input estimator has 3 classes'
    assert_raise_message(ValueError, msg, StigClassifier, [('lr', clf)])

    clf = LogisticRegression(random_state=1)
    msg = 'Estimator does not have property _classes. Fit classifier first.'
    assert_raise_message(AttributeError, msg, StigClassifier, [('lr', clf)])

    msg = 'Must use at least one estimator'
    assert_raise_message(ValueError, msg, StigClassifier, [])

    msg = 'Estimator does not have property _classes. Fit classifier first.'
    assert_raise_message(AttributeError, msg, StigClassifier, [('lr', 'taco')])


def test_balanced_accuracy():
    nTrials = 10
    nClfs = 1
    point_list, label_list = generate_points(num_points)
    clf = LogisticRegression(random_state=1).fit(point_list, label_list)
    eclf = StigClassifier(estimators=[('lr', clf)])

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

    # one fn
    pseudo_labels = np.ones((nClfs, nTrials), dtype=int)
    true_labels = np.ones((nTrials,), dtype=int)

    pseudo_labels[0][0] = 0

    psi, eta, pi = eclf._balanced_accuracy(pseudo_labels, true_labels)

    assert_almost_equal(psi[0], 0.9, err_msg="psy should be 0.9")
    assert_equal(eta[0], 0.0, "eta should be 0.")
    assert_equal(pi[0], 0.45, "pi should be 0.45")

    # one fp
    pseudo_labels = np.ones((nClfs, nTrials), dtype=int)
    true_labels = np.ones((nTrials,), dtype=int)

    true_labels[0] = 0

    psi, eta, pi = eclf._balanced_accuracy(pseudo_labels, true_labels)

    assert_equal(psi[0], 1.0, "psy should be 1.")
    assert_equal(eta[0], 0.0, "eta should be 0.")
    assert_equal(pi[0], 0.5, "pi should be 1.")

    # one tp
    pseudo_labels = np.zeros((nClfs, nTrials), dtype=int)
    true_labels = np.zeros((nTrials,), dtype=int)

    pseudo_labels[0][0] = 1
    true_labels[0] = 1

    psi, eta, pi = eclf._balanced_accuracy(pseudo_labels, true_labels)

    assert_equal(psi[0], 1.0, "psy should be 1.")
    assert_equal(eta[0], 1.0, "eta should be 0.")
    assert_equal(pi[0], 1.0, "pi should be 1.")


@mock.patch('pyriemann.ensemble.StigClassifier._get_principal_eig')
def test__apply_sml(mock__get_principal_eig):
    eclf = StigClassifier(estimators=estimators)

    expected_nClfs = len(estimators)
    expected_nTrials = 32

    rs = np.random.RandomState(1234)

    hard_preds = np.random.randint(0, 2, size=(expected_nClfs, expected_nTrials), dtype=int)

    expected_weight = np.zeros((1, expected_nClfs))
    expected_weight += 1./expected_nClfs

    mock__get_principal_eig.return_value = np.ones((expected_nClfs,))

    actual_weight = eclf._apply_sml(hard_preds)

    assert_almost_equal(actual_weight, expected_weight, err_msg="expected %s but got %s" % (expected_weight, actual_weight))

    # All the same but one
    mock__get_principal_eig.reset_mock()
    hard_preds = np.ones((expected_nClfs, expected_nTrials), dtype=int)

    hard_preds[0][1] = False

    actual_weight = eclf._apply_sml(hard_preds)

    expected_weight = np.zeros((1, expected_nClfs))
    expected_weight += 1. / expected_nClfs
    assert_almost_equal(actual_weight, expected_weight, err_msg="expected %s but got %s" % (expected_weight, actual_weight))

    # Can't do with numpy
    # mock__get_principal_eig.assert_called_with(hard_preds)
    assert_array_equal(mock__get_principal_eig.call_args[0][0], hard_preds)


def test__collect_probas():
    expected_sml_limit = 50

    eclf = StigClassifier(estimators=estimators,
                          sml_limit=expected_sml_limit)

    # when num x is less then estimators or `sml_threshold`
    num_xs = len(estimators) + 2
    num_classes = 2
    probas = eclf._collect_probas(covset[:num_xs])
    assert_equal(probas.shape, np.empty((len(estimators), num_xs)).shape,
                 "should make internal x_ a buffer of len sml_limit")
    assert_true(probas.dtype == float)


def test__collect_predicts():
    expected_sml_limit = 50

    eclf = StigClassifier(estimators=estimators,
                          sml_limit=expected_sml_limit)

    # when num x is less then estimators or `sml_threshold`
    num_xs = len(estimators) + 2
    num_classes = 2
    probas = eclf._collect_predicts(covset[:num_xs])
    assert_equal(probas.shape, np.empty((len(estimators), num_xs)).shape,
                 "should make internal x_ a buffer of len sml_limit")
    assert_true(probas.dtype == int)


def test__get_ensemble_scores_labels():
    eclf = StigClassifier(estimators=estimators)

    expected_nClfs = 2
    expected_nTrials = 3

    tmp_scores = np.zeros((expected_nClfs, expected_nTrials))

    rs = np.random.RandomState(1234)

    for i in range(expected_nClfs):
        for j in range(expected_nTrials):
            tmp_scores[i][j] = rs.rand()

    max_list = eclf._find_indexes_of_maxes(tmp_scores)

    expected_ensemble_scores = np.zeros((expected_nTrials,), dtype=float)
    expected_ensemble_labels = np.zeros((expected_nTrials,), dtype=float)

    for i in range(expected_nTrials):
        ind = max_list[i]
        expected_ensemble_scores[i] = tmp_scores[ind][i]
        class_label = 0 if tmp_scores[ind][i] < 0.5 else 1
        expected_ensemble_labels[i] = class_label

    actual_ensemble_scores, actual_ensemble_labels = eclf._get_ensemble_scores_labels(tmp_scores, max_list)

    assert_almost_equal(actual_ensemble_scores, expected_ensemble_scores)
    assert_array_equal(actual_ensemble_labels, expected_ensemble_labels)


def test__find_indexes_of_maxes():

    eclf = StigClassifier(estimators=estimators)

    expected_nClfs = 2
    expected_nTrials = 3

    tmp_scores = np.zeros((expected_nClfs, expected_nTrials))

    rs = np.random.RandomState(1234)

    for i in range(expected_nClfs):
        for j in range(expected_nTrials):
            tmp_scores[i][j] = rs.rand()

    expected_max_list = np.zeros((expected_nTrials,), dtype=int)

    for trial in range(expected_nTrials):
        max_ = 0.
        for clf in range(expected_nClfs):
            diff = tmp_scores[clf][trial] - 0.5
            if abs(diff) > max_:
                max_ = abs(diff)
                expected_max_list[trial] = clf

    actual_max_list = eclf._find_indexes_of_maxes(tmp_scores)

    assert_array_equal(actual_max_list, expected_max_list,
                       "expected max list of %s but got %s" % (expected_max_list, actual_max_list))


def test__get_principal_eig():

    eclf = StigClassifier(estimators=estimators_lr)

    none_return = eclf._get_principal_eig(np.array([]))

    assert_equal(none_return, None, "should be none")


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
    num_xs_1 = 40
    actual_preds = eclf.predict(point_list[num_xs:num_xs + num_xs_1])
    assert_equal(actual_preds.shape[0], np.ones((num_xs_1,)).shape[0],
                 "should make internal x_ a buffer of len sml_limit")

    # bad data
    num_xs = 40
    bad_point_list, _ = generate_points(r=6, Nt=num_points)
    actual_preds = eclf.predict(bad_point_list[:num_xs])
    assert_equal(actual_preds.shape[0], np.ones((num_xs,)).shape[0],
                 "should make preds of len number x's")


def test_predict_proba():
    expected_sml_limit = 50

    eclf = StigClassifier(estimators=estimators_lr,
                          sml_limit=expected_sml_limit)

    point_list, label_list = generate_points(num_points)

    expected_shape = (expected_sml_limit,) + point_list[0].shape

    # when num x is less then estimators or `sml_threshold`
    num_xs = len(estimators) - 2
    actual_soft_preds = eclf.predict_proba(point_list[:num_xs])
    assert_equal(actual_soft_preds.shape[0], np.ones((num_xs,)).shape[0], "should produce num_xs predictions")
    assert_equal(actual_soft_preds.shape[1], 2, "should return a probability for each class")

    # too good converges fast.
    num_xs_1 = 40
    actual_soft_preds = eclf.predict_proba(point_list[num_xs:num_xs + num_xs_1])
    assert_equal(actual_soft_preds.shape[0], np.ones((num_xs_1,)).shape[0], "should produce num_xs_1 predictions")
    assert_equal(actual_soft_preds.shape[1], 2, "should return a probability for each class")

    # bad data
    num_xs = 40
    bad_point_list, _ = generate_points(r=6, Nt=num_points)
    actual_preds_soft_bad = eclf.predict_proba(bad_point_list[:num_xs])
    assert_equal(actual_preds_soft_bad.shape[0], np.ones((num_xs,)).shape[0], "should produce num_xs_1 predictions")
    assert_equal(actual_preds_soft_bad.shape[1], 2, "should return a probability for each class")


def test_fit():
    expected_sml_limit = 50

    eclf = StigClassifier(estimators=estimators_lr,
                          sml_limit=expected_sml_limit)

    point_list, label_list = generate_points(num_points)

    expected_shape = (expected_sml_limit,) + point_list[0].shape

    # when num x is less then estimators or `sml_threshold`
    num_xs = len(estimators) - 2
    eclf.fit(point_list[:num_xs], label_list[:num_xs])
    actual_preds = eclf.predict(point_list[:num_xs])
    assert_equal(actual_preds.shape[0], np.ones((num_xs,)).shape[0], "should make internal x_ a buffer of len sml_limit")
    assert_equal(eclf.x_in, num_xs)

    # too good converges fast.
    num_xs_1 = 40
    eclf.fit(point_list[num_xs:num_xs + num_xs_1], label_list[num_xs:num_xs + num_xs_1])
    actual_preds = eclf.predict(point_list[num_xs:num_xs + num_xs_1])
    assert_equal(actual_preds.shape[0], np.ones((num_xs_1,)).shape[0],
                 "should make internal x_ a buffer of len sml_limit")
    assert_equal(eclf.x_in, num_xs_1)

    # bad data
    num_xs = 40
    bad_point_list, bad_label_list = generate_points(r=6, Nt=num_points)
    eclf.fit(bad_point_list[:num_xs], bad_label_list[:num_xs])
    actual_preds = eclf.predict(bad_point_list[:num_xs])
    assert_equal(actual_preds.shape[0], np.ones((num_xs,)).shape[0],
                 "should make preds of len number x's")
    assert_equal(eclf.x_in, num_xs)

    # calling fit without y
    num_xs = len(estimators) - 2
    eclf.fit(point_list[:num_xs])
    actual_preds = eclf.predict(point_list[:num_xs])
    assert_equal(actual_preds.shape[0], np.ones((num_xs,)).shape[0], "should make internal x_ a buffer of len sml_limit")
    assert_equal(eclf.x_in, num_xs)


def test_partial_fit():
    expected_sml_limit = 50

    eclf = StigClassifier(estimators=estimators_lr,
                          sml_limit=expected_sml_limit)

    point_list, label_list = generate_points(num_points)

    expected_shape = (expected_sml_limit,) + point_list[0].shape

    # when num x is less then estimators or `sml_threshold`
    num_xs = len(estimators) - 2
    eclf.partial_fit(point_list[:num_xs], label_list[:num_xs])
    actual_preds = eclf.predict(point_list[:num_xs])
    assert_equal(actual_preds.shape[0], np.ones((num_xs,)).shape[0], "should make internal x_ a buffer of len sml_limit")
    assert_equal(eclf.x_in, num_xs)

    # too good converges fast.
    num_xs_1 = 40
    eclf.partial_fit(point_list[num_xs:num_xs + num_xs_1], label_list[num_xs:num_xs + num_xs_1])
    actual_preds = eclf.predict(point_list[num_xs:num_xs + num_xs_1])
    assert_equal(actual_preds.shape[0], np.ones((num_xs_1,)).shape[0],
                 "should make internal x_ a buffer of len sml_limit")
    assert_equal(eclf.x_in, num_xs + num_xs_1)


    # bad data
    num_xs_2 = 40
    bad_point_list, bad_label_list = generate_points(r=6, Nt=num_points)
    eclf.partial_fit(bad_point_list[:num_xs_2], bad_label_list[:num_xs_2])
    actual_preds = eclf.predict(bad_point_list[:num_xs_2])
    assert_equal(actual_preds.shape[0], np.ones((num_xs_2,)).shape[0],
                 "should make preds of len number x's")
    assert_equal(eclf.x_in, expected_sml_limit)


def test__estimation_maximization():
    expected_nb_chan = 8
    expected_nb_matricies = 100
    expected_nb_clfs = 10

    estimators = []
    snrs = [0.01, 0.1, 1, 10]
    effect_sizes = [1, 2, 3]
    for snr in snrs:
        for effect_size in effect_sizes:
            X_, labels_ = generate_random_covariances(N=expected_nb_chan,
                                                      effect_size=effect_size,
                                                      K=expected_nb_matricies,
                                                      snr=snr,
                                                      return_signal=False,
                                                      random_state=42,
                                                      L=2)
            i = len(estimators)
            estimators.append(('mdm%i' % i, MDM(metric='riemann').fit(X_, labels_)))

    eclf = StigClassifier(estimators=estimators)

    X_test, labels_test = generate_random_covariances(N=expected_nb_chan,
                                                      effect_size=2.5,
                                                      K=expected_nb_matricies,
                                                      snr=5,
                                                      return_signal=False,
                                                      random_state=42,
                                                      L=2)
    for i in range(len(estimators)):
        _, clf = estimators[i]
        acc = clf.score(X_test, labels_test)

    eclf.fit(X_test)









