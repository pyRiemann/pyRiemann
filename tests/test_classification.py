import pickle

from conftest import get_distances, get_means, get_metrics
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from pytest import approx
from sklearn.dummy import DummyClassifier

from pyriemann.estimation import Covariances
from pyriemann.utils.kernel import kernel
from pyriemann.utils.mean import mean_covariance
from pyriemann.classification import (
    MDM,
    FgMDM,
    KNearestNeighbor,
    TSclassifier,
    SVC,
    MeanField
)

rclf = [MDM, FgMDM, KNearestNeighbor, TSclassifier, SVC, MeanField]


@pytest.mark.parametrize("classif", rclf)
class ClassifierTestCase:
    def test_two_classes(self, classif, get_covmats, get_labels):
        n_classes, n_matrices, n_channels = 2, 6, 3
        covmats = get_covmats(n_matrices, n_channels)
        labels = get_labels(n_matrices, n_classes)
        self.clf_predict(classif, covmats, labels)
        self.clf_fit_independence(classif, covmats, labels)
        self.clf_predict_proba(classif, covmats, labels)
        self.clf_populate_classes(classif, covmats, labels)
        if classif in (MDM, KNearestNeighbor, MeanField):
            self.clf_fitpredict(classif, covmats, labels)
        if classif in (MDM, FgMDM):
            self.clf_transform(classif, covmats, labels)
        if hasattr(classif(), 'n_jobs'):
            self.clf_jobs(classif, covmats, labels)
        if hasattr(classif(), 'tsupdate'):
            self.clf_tsupdate(classif, covmats, labels)

    def test_multi_classes(self, classif, get_covmats, get_labels):
        n_classes, n_matrices, n_channels = 3, 9, 3
        covmats = get_covmats(n_matrices, n_channels)
        labels = get_labels(n_matrices, n_classes)
        self.clf_predict(classif, covmats, labels)
        self.clf_fit_independence(classif, covmats, labels)
        self.clf_predict_proba(classif, covmats, labels)
        self.clf_populate_classes(classif, covmats, labels)
        if classif in (MDM, KNearestNeighbor, MeanField):
            self.clf_fitpredict(classif, covmats, labels)
        if classif in (MDM, FgMDM):
            self.clf_transform(classif, covmats, labels)
        if hasattr(classif(), 'n_jobs'):
            self.clf_jobs(classif, covmats, labels)
        if hasattr(classif(), 'tsupdate'):
            self.clf_tsupdate(classif, covmats, labels)


class TestClassifier(ClassifierTestCase):
    def clf_predict(self, classif, covmats, labels):
        n_matrices = len(labels)
        clf = classif()
        clf.fit(covmats, labels)
        predicted = clf.predict(covmats)
        assert predicted.shape == (n_matrices,)

    def clf_predict_proba(self, classif, covmats, labels):
        n_matrices = len(labels)
        n_classes = len(np.unique(labels))
        clf = classif()
        if hasattr(clf, 'probability'):
            clf.set_params(**{'probability': True})
        clf.fit(covmats, labels)
        probabilities = clf.predict_proba(covmats)
        assert probabilities.shape == (n_matrices, n_classes)
        assert probabilities.sum(axis=1) == approx(np.ones(n_matrices))

    def clf_fitpredict(self, classif, covmats, labels):
        clf = classif()
        clf.fit_predict(covmats, labels)
        assert_array_equal(clf.classes_, np.unique(labels))

    def clf_transform(self, classif, covmats, labels):
        n_matrices = len(labels)
        n_classes = len(np.unique(labels))
        clf = classif()
        clf.fit(covmats, labels)
        transformed = clf.transform(covmats)
        assert transformed.shape == (n_matrices, n_classes)

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

    def clf_tsupdate(self, classif, covmats, labels):
        clf = classif(tsupdate=True)
        clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("classif", rclf)
@pytest.mark.parametrize("mean", ["faulty", 42])
@pytest.mark.parametrize("dist", ["not_real", 27])
def test_metric_dict_error(classif, mean, dist, get_covmats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    labels = get_labels(n_matrices, n_classes)
    covmats = get_covmats(n_matrices, n_channels)
    clf = classif(metric={"mean": mean, "distance": dist})
    with pytest.raises((TypeError, KeyError)):
        clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("classif", rclf)
@pytest.mark.parametrize("metric", [42, "faulty", {"foo": "bar"}])
def test_metric_errors(classif, metric, get_covmats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    labels = get_labels(n_matrices, n_classes)
    covmats = get_covmats(n_matrices, n_channels)
    clf = classif(metric=metric)
    with pytest.raises((TypeError, KeyError, ValueError)):
        clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("classif", rclf)
@pytest.mark.parametrize("metric", get_metrics())
def test_metric_str(classif, metric, get_covmats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    labels = get_labels(n_matrices, n_classes)
    covmats = get_covmats(n_matrices, n_channels)
    clf = classif(metric=metric)

    if classif in [SVC, FgMDM, TSclassifier] \
            and metric not in ['riemann', 'euclid', 'logeuclid']:
        with pytest.raises((KeyError, ValueError)):
            clf.fit(covmats, labels).predict(covmats)
    else:
        clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("metric_mean", get_means())
@pytest.mark.parametrize("metric_dist", get_distances())
def test_metric_mdm(metric_mean, metric_dist, get_covmats, get_labels):
    n_matrices, n_channels, n_classes = 4, 3, 2
    labels = get_labels(n_matrices, n_classes)
    covmats = get_covmats(n_matrices, n_channels)
    clf = MDM(metric={"mean": metric_mean, "distance": metric_dist})
    clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("metric_mean", get_means())
@pytest.mark.parametrize("metric_dist", get_distances())
@pytest.mark.parametrize("metric_map", ["euclid", "logeuclid", "riemann"])
def test_metric_fgmdm(metric_mean, metric_dist, metric_map, get_covmats,
                      get_labels):
    n_matrices, n_channels, n_classes = 4, 3, 2
    labels = get_labels(n_matrices, n_classes)
    covmats = get_covmats(n_matrices, n_channels)
    clf = FgMDM(metric={
        "mean": metric_mean,
        "distance": metric_dist,
        "map": metric_map
    })
    clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("metric_mean", get_means())
@pytest.mark.parametrize("metric_map", ["euclid", "logeuclid", "riemann"])
def test_metric_tsclassifier(metric_mean, metric_map, get_covmats, get_labels):
    n_matrices, n_channels, n_classes = 4, 3, 2
    labels = get_labels(n_matrices, n_classes)
    covmats = get_covmats(n_matrices, n_channels)
    clf = TSclassifier(metric={"mean": metric_mean, "map": metric_map})
    clf.fit(covmats, labels).predict(covmats)


def test_1nn(get_covmats, get_labels):
    """Test KNearestNeighbor with K=1"""
    n_matrices, n_channels, n_classes = 9, 3, 3
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)

    knn = KNearestNeighbor(1, metric="riemann")
    knn.fit(covmats, labels)
    preds = knn.predict(covmats)
    assert_array_equal(labels, preds)


def test_tsclassifier_fit(get_covmats, get_labels):
    """Test TS Classifier"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    clf = TSclassifier(clf=DummyClassifier())
    clf.fit(covmats, labels).predict(covmats)


def test_tsclassifier_clf_error(get_covmats, get_labels):
    """Test TS if not Classifier"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    with pytest.raises(TypeError):
        TSclassifier(clf=Covariances()).fit(covmats, labels)


def test_svc_params():
    rsvc = SVC()
    assert rsvc.metric == 'riemann'

    rsvc.set_params(**{'metric': 'logeuclid'})
    assert rsvc.metric == 'logeuclid'

    rsvc.set_params(**{'max_iter': 501})
    assert rsvc.max_iter == 501


def test_svc_params_error(get_covmats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)

    with pytest.raises(TypeError):
        SVC(C='hello').fit(covmats, labels)

    with pytest.raises(TypeError):
        SVC(foo=5)


@pytest.mark.parametrize("metric", ["riemann", "euclid", "logeuclid"])
def test_svc_cref_metric(get_covmats, get_labels, metric):
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    Cref = mean_covariance(covmats, metric=metric)

    rsvc = SVC(Cref=Cref).fit(covmats, labels)
    rsvc_1 = SVC(Cref=None, metric=metric).fit(covmats, labels)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)


@pytest.mark.parametrize("metric", ["riemann", "euclid", "logeuclid"])
def test_svc_cref_callable(get_covmats, get_labels, metric):
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    def Cref(X): return mean_covariance(X, metric=metric)

    rsvc = SVC(Cref=Cref).fit(covmats, labels)
    rsvc_1 = SVC(metric=metric).fit(covmats, labels)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)

    rsvc = SVC(Cref=Cref).fit(covmats, labels)
    rsvc.predict(covmats)
    rsvc_1 = SVC(metric=metric).fit(covmats, labels)
    rsvc_1.predict(covmats)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)


@pytest.mark.parametrize("metric", ["riemann", "euclid", "logeuclid"])
def test_svc_cref_error(get_covmats, get_labels, metric):
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    def Cref(X, met): mean_covariance(X, metric=met)

    with pytest.raises(TypeError):
        SVC(Cref=Cref).fit(covmats, labels)

    Cref = metric

    with pytest.raises(TypeError):
        SVC(Cref=Cref).fit(covmats, labels)


@pytest.mark.parametrize("metric", ["riemann", "euclid", "logeuclid"])
def test_svc_kernel_callable(get_covmats, get_labels, metric):
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)

    rsvc = SVC(kernel_fct=kernel,
               metric=metric).fit(covmats, labels)
    rsvc_1 = SVC(metric=metric).fit(covmats, labels)
    p1 = rsvc.predict(covmats[:-1])
    p2 = rsvc_1.predict(covmats[:-1])
    assert np.array_equal(p1, p2)

    def custom_kernel(X, Y, Cref, metric):
        return np.ones((len(X), len(Y)))
    SVC(kernel_fct=custom_kernel,
        metric=metric).fit(covmats, labels).predict(covmats[:-1])

    def custom_kernel(X, Y, Cref):
        return np.ones((len(X), len(Y)))
    with pytest.raises(TypeError):
        SVC(kernel_fct=custom_kernel, metric=metric).fit(covmats, labels)

    custom_kernel = np.array([1, 2])
    with pytest.raises(TypeError):
        SVC(kernel_fct=custom_kernel, metric=metric).fit(covmats, labels)

    # check if pickleable
    pickle.dumps(rsvc)
    pickle.dumps(rsvc_1)


@pytest.mark.parametrize("method_label", ["sum_means", "inf_means"])
def test_meanfield(get_covmats, get_labels, method_label):
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    mf = MeanField(method_label=method_label).fit(covmats, labels)
    pred = mf.predict(covmats)
    assert pred.shape == (n_matrices,)
    proba = mf.predict_proba(covmats)
    assert proba.shape == (n_matrices, n_classes)
    transf = mf.transform(covmats)
    assert transf.shape == (n_matrices, n_classes)
