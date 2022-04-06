from conftest import get_distances, get_means, get_metrics
import numpy as np
from numpy.testing import assert_array_equal
from pyriemann.classification import (MDM, FgMDM, KNearestNeighbor,
                                      TSclassifier, SVC)

from pyriemann.estimation import Covariances
import pytest
from pytest import approx
from sklearn.dummy import DummyClassifier


rclf = [MDM, FgMDM, KNearestNeighbor, TSclassifier, SVC]


@pytest.mark.parametrize("classif", rclf)
class ClassifierTestCase:
    def test_two_classes(self, classif, get_covmats, get_labels):
        n_classes, n_trials, n_channels = 2, 6, 3
        covmats = get_covmats(n_trials, n_channels)
        labels = get_labels(n_trials, n_classes)
        self.clf_predict(classif, covmats, labels)
        self.clf_fit_independence(classif, covmats, labels)
        if classif is MDM:
            self.clf_fitpredict(classif, covmats, labels)
        if classif in (MDM, FgMDM):
            self.clf_transform(classif, covmats, labels)
        if classif in (MDM, FgMDM, KNearestNeighbor):
            self.clf_jobs(classif, covmats, labels)
        if classif in (MDM, FgMDM, TSclassifier):
            self.clf_predict_proba(classif, covmats, labels)
            self.clf_populate_classes(classif, covmats, labels)
        if classif is KNearestNeighbor:
            self.clf_predict_proba_trials(classif, covmats, labels)
        if classif is (FgMDM, TSclassifier):
            self.clf_tsupdate(classif, covmats, labels)

    def test_multi_classes(self, classif, get_covmats, get_labels):
        n_classes, n_trials, n_channels = 3, 9, 3
        covmats = get_covmats(n_trials, n_channels)
        labels = get_labels(n_trials, n_classes)
        self.clf_fit_independence(classif, covmats, labels)
        self.clf_predict(classif, covmats, labels)
        if classif is MDM:
            self.clf_fitpredict(classif, covmats, labels)
        if classif in (MDM, FgMDM):
            self.clf_transform(classif, covmats, labels)
        if classif in (MDM, FgMDM, KNearestNeighbor):
            self.clf_jobs(classif, covmats, labels)
        if classif in (MDM, FgMDM, TSclassifier):
            self.clf_predict_proba(classif, covmats, labels)
            self.clf_populate_classes(classif, covmats, labels)
        if classif is KNearestNeighbor:
            self.clf_predict_proba_trials(classif, covmats, labels)
        if classif is (FgMDM, TSclassifier):
            self.clf_tsupdate(classif, covmats, labels)


class TestClassifier(ClassifierTestCase):
    def clf_predict(self, classif, covmats, labels):
        n_trials = len(labels)
        clf = classif()
        clf.fit(covmats, labels)
        predicted = clf.predict(covmats)
        assert predicted.shape == (n_trials,)

    def clf_predict_proba(self, classif, covmats, labels):
        n_trials = len(labels)
        n_classes = len(np.unique(labels))
        clf = classif()
        clf.fit(covmats, labels)
        probabilities = clf.predict_proba(covmats)
        assert probabilities.shape == (n_trials, n_classes)
        assert probabilities.sum(axis=1) == approx(np.ones(n_trials))

    def clf_predict_proba_trials(self, classif, covmats, labels):
        n_trials = len(labels)
        # n_classes = len(np.unique(labels))
        clf = classif()
        clf.fit(covmats, labels)
        probabilities = clf.predict_proba(covmats)
        assert probabilities.shape == (n_trials, n_trials)
        assert probabilities.sum(axis=1) == approx(np.ones(n_trials))

    def clf_fitpredict(self, classif, covmats, labels):
        clf = classif()
        clf.fit_predict(covmats, labels)
        assert_array_equal(clf.classes_, np.unique(labels))

    def clf_transform(self, classif, covmats, labels):
        n_trials = len(labels)
        n_classes = len(np.unique(labels))
        clf = classif()
        clf.fit(covmats, labels)
        transformed = clf.transform(covmats)
        assert transformed.shape == (n_trials, n_classes)

    def clf_fit_independence(self, classif, covmats, labels):
        clf = classif()
        clf.fit(covmats, labels).predict(covmats)
        # retraining with different size should erase previous fit
        new_covmats = covmats[:, :-1, :-1]
        clf.fit(new_covmats, labels).predict(new_covmats)

    def clf_jobs(self, classif, covmats, labels):
        clf = classif(n_jobs=2)
        clf.fit(covmats, labels)
        clf.predict(covmats)

    def clf_populate_classes(self, classif, covmats, labels):
        clf = classif()
        clf.fit(covmats, labels)
        assert_array_equal(clf.classes_, np.unique(labels))

    def clf_classif_tsupdate(self, classif, covmats, labels):
        clf = classif(tsupdate=True)
        clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("classif", [MDM, FgMDM, TSclassifier])
@pytest.mark.parametrize("mean", ["faulty", 42])
@pytest.mark.parametrize("dist", ["not_real", 27])
def test_metric_dict_error(classif, mean, dist, get_covmats, get_labels):
    with pytest.raises((TypeError, KeyError)):
        n_trials, n_channels, n_classes = 6, 3, 2
        labels = get_labels(n_trials, n_classes)
        covmats = get_covmats(n_trials, n_channels)
        clf = classif(metric={"mean": mean, "distance": dist})
        clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("classif", [MDM, FgMDM])
@pytest.mark.parametrize("mean", get_means())
@pytest.mark.parametrize("dist", get_distances())
def test_metric_dist(classif, mean, dist, get_covmats, get_labels):
    n_trials, n_channels, n_classes = 4, 3, 2
    labels = get_labels(n_trials, n_classes)
    covmats = get_covmats(n_trials, n_channels)
    clf = classif(metric={"mean": mean, "distance": dist})
    clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("classif", rclf)
@pytest.mark.parametrize("metric", [42, "faulty", {"foo": "bar"}])
def test_metric_wrong_keys(classif, metric, get_covmats, get_labels):
    with pytest.raises((TypeError, KeyError, ValueError)):
        n_trials, n_channels, n_classes = 6, 3, 2
        labels = get_labels(n_trials, n_classes)
        covmats = get_covmats(n_trials, n_channels)
        clf = classif(metric=metric)
        clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("classif", rclf)
@pytest.mark.parametrize("metric", get_metrics())
def test_metric_str(classif, metric, get_covmats, get_labels):
    n_trials, n_channels, n_classes = 6, 3, 2
    labels = get_labels(n_trials, n_classes)
    covmats = get_covmats(n_trials, n_channels)
    clf = classif(metric=metric)

    if classif is SVC and metric not in ['riemann', 'euclid', 'logeuclid']:
        with pytest.raises(ValueError):
            clf.fit(covmats, labels).predict(covmats)

    else:
        clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("dist", ["not_real", 42])
def test_knn_dict_dist(dist, get_covmats, get_labels):
    with pytest.raises(KeyError):
        n_trials, n_channels, n_classes = 6, 3, 2
        labels = get_labels(n_trials, n_classes)
        covmats = get_covmats(n_trials, n_channels)
        clf = KNearestNeighbor(metric={"distance": dist})
        clf.fit(covmats, labels).predict(covmats)


def test_1NN(get_covmats, get_labels):
    """Test KNearestNeighbor with K=1"""
    n_trials, n_channels, n_classes = 9, 3, 3
    covmats = get_covmats(n_trials, n_channels)
    labels = get_labels(n_trials, n_classes)

    knn = KNearestNeighbor(1, metric="riemann")
    knn.fit(covmats, labels)
    preds = knn.predict(covmats)
    assert_array_equal(labels, preds)


def test_TSclassifier_classifier(get_covmats, get_labels):
    """Test TS Classifier"""
    n_trials, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_trials, n_channels)
    labels = get_labels(n_trials, n_classes)
    clf = TSclassifier(clf=DummyClassifier())
    clf.fit(covmats, labels).predict(covmats)


def test_TSclassifier_classifier_error():
    """Test TS if not Classifier"""
    with pytest.raises(TypeError):
        TSclassifier(clf=Covariances())


def test_supportvectormachine_svc_params():
    rsvc = SVC()
    assert rsvc.metric == 'riemann'

    rsvc.set_params(**{'metric': 'logeuclid'})
    assert rsvc.metric == 'logeuclid'

    rsvc.set_params(**{'max_iter': 501})
    assert rsvc.max_iter == 501


def test_supportvectormachine_svc_params_error(get_covmats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)

    with pytest.raises(TypeError):
        SVC(C='hello').fit(covmats, labels)

    with pytest.raises(TypeError):
        SVC(foo=5)
