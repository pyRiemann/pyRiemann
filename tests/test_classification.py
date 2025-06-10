import pickle

import numpy as np
from numpy.testing import assert_array_equal
import pytest
from pytest import approx
from scipy.spatial.distance import euclidean
from scipy.stats import mode
from sklearn.dummy import DummyClassifier

from conftest import get_distances, get_means, get_metrics
from pyriemann.estimation import Covariances
from pyriemann.utils.kernel import kernel
from pyriemann.utils.mean import mean_covariance
from pyriemann.classification import (
    _mode_2d,
    MDM,
    FgMDM,
    KNearestNeighbor,
    TSClassifier,
    SVC,
    MeanField,
    class_distinctiveness,
)

classifs = [MDM, FgMDM, KNearestNeighbor, TSClassifier, SVC, MeanField]


@pytest.mark.parametrize(
    "X, axis, expected",
    [
        (
            np.array([[0, 5], [1, 7], [0, 6], [2, 7]]),
            0,
            np.array([0, 7]),
        ),
        (
            np.array([[0, 1, 2, 1, 1, 0], [7, 5, 7, 6, 7, 6]]),
            1,
            np.array([1, 7]),
        ),
        (
            np.array([["a", "b", "a", "c", "a"], ["d", "e", "f", "e", "e"]]),
            1,
            np.array(["a", "e"]),
        ),
    ],
)
def test_mode(X, axis, expected):
    actual = _mode_2d(X, axis=axis)
    assert_array_equal(actual, expected)

    if np.issubdtype(X.dtype, np.number):
        sp, _ = mode(X, axis=axis)
        assert_array_equal(actual, sp.ravel())


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("classif", classifs)
def test_classifier(kind, n_classes, classif,
                    get_mats, get_labels, get_weights):
    if kind == "hpd" and classif in [FgMDM, TSClassifier, SVC]:
        pytest.skip()
    if n_classes == 2:
        n_matrices, n_channels = 6, 3
    else:
        assert n_classes == 3
        n_matrices, n_channels = 9, 3
    mats = get_mats(n_matrices, n_channels, kind)
    labels = get_labels(n_matrices, n_classes)
    weights = get_weights(n_matrices)

    clf_fit(classif, mats, labels, weights)
    clf_predict(classif, mats, labels)
    clf_fit_independence(classif, mats, labels)
    clf_predict_proba(classif, mats, labels)
    clf_score(classif, mats, labels)
    clf_populate_classes(classif, mats, labels)
    if classif in (MDM, FgMDM, MeanField):
        clf_transform(classif, mats, labels)
        clf_fittransform(classif, mats, labels)
    if hasattr(classif(), "n_jobs"):
        clf_jobs(classif, mats, labels)
    if hasattr(classif(), "tsupdate"):
        clf_tsupdate(classif, mats, labels)


def clf_fit(classif, mats, labels, weights):
    n_classes = len(np.unique(labels))
    clf = classif().fit(mats, labels)
    assert clf.classes_.shape == (n_classes,)
    assert_array_equal(clf.classes_, np.unique(labels))

    clf.fit(mats, labels, sample_weight=weights)


def clf_predict(classif, mats, labels):
    n_matrices = len(labels)
    clf = classif()
    pred = clf.fit(mats, labels).predict(mats)
    assert pred.shape == (n_matrices,)


def clf_predict_proba(classif, mats, labels):
    n_matrices, n_classes = len(labels), len(np.unique(labels))
    clf = classif()
    if hasattr(clf, "probability"):
        clf.set_params(**{"probability": True})
    proba = clf.fit(mats, labels).predict_proba(mats)
    assert proba.shape == (n_matrices, n_classes)
    assert proba.sum(axis=1) == approx(np.ones(n_matrices))


def clf_score(classif, mats, labels):
    clf = classif()
    clf.fit(mats, labels).score(mats, labels)


def clf_transform(classif, mats, labels):
    n_matrices, n_classes = len(labels), len(np.unique(labels))
    clf = classif()
    transf = clf.fit(mats, labels).transform(mats)
    assert transf.shape == (n_matrices, n_classes)


def clf_fittransform(classif, mats, labels):
    clf = classif()
    transf = clf.fit_transform(mats, labels)
    transf2 = clf.fit(mats, labels).transform(mats)
    assert_array_equal(transf, transf2)


def clf_fit_independence(classif, mats, labels):
    clf = classif()
    clf.fit(mats, labels).predict(mats)
    # retraining with different size should erase previous fit
    new_mats = mats[:, :-1, :-1]
    clf.fit(new_mats, labels).predict(new_mats)


def clf_jobs(classif, mats, labels):
    clf = classif(n_jobs=2)
    clf.fit(mats, labels).predict(mats)


def clf_populate_classes(classif, mats, labels):
    clf = classif()
    clf.fit(mats, labels)
    assert_array_equal(clf.classes_, np.unique(labels))


def clf_tsupdate(classif, mats, labels):
    clf = classif(tsupdate=True)
    clf.fit(mats, labels).predict(mats)


@pytest.mark.parametrize("classif", classifs)
@pytest.mark.parametrize("mean", ["faulty", 42])
@pytest.mark.parametrize("dist", ["not_real", 27])
def test_metric_dict_error(classif, mean, dist, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    labels = get_labels(n_matrices, n_classes)
    mats = get_mats(n_matrices, n_channels, "spd")
    clf = classif(metric={"mean": mean, "distance": dist})
    with pytest.raises((TypeError, KeyError, ValueError)):
        clf.fit(mats, labels).predict(mats)


@pytest.mark.parametrize("classif", classifs)
@pytest.mark.parametrize("metric", [42, "faulty", {"foo": "bar"}])
def test_metric_errors(classif, metric, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    labels = get_labels(n_matrices, n_classes)
    mats = get_mats(n_matrices, n_channels, "spd")
    clf = classif(metric=metric)
    with pytest.raises((TypeError, KeyError, ValueError)):
        clf.fit(mats, labels).predict(mats)


@pytest.mark.parametrize("classif", classifs)
@pytest.mark.parametrize("metric", get_metrics())
def test_metric_str(classif, metric, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    labels = get_labels(n_matrices, n_classes)
    mats = get_mats(n_matrices, n_channels, "spd")
    clf = classif(metric=metric)

    if classif in [SVC, FgMDM, TSClassifier] \
            and metric not in ["euclid", "logchol", "logeuclid", "riemann"]:
        with pytest.raises((KeyError, ValueError)):
            clf.fit(mats, labels).predict(mats)
    else:
        clf.fit(mats, labels).predict(mats)


def call_mean(X, sample_weight=None):
    return np.average(X, axis=0, weights=sample_weight)


def call_dist(A, B, squared=False):
    return euclidean(A.flatten(), B.flatten())


@pytest.mark.parametrize("metric_mean", list(get_means()) + [call_mean])
@pytest.mark.parametrize("metric_dist", list(get_distances()) + [call_dist])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
def test_mdm(metric_mean, metric_dist, n_classes, get_mats, get_labels):
    n_matrices, n_channels = 4 * n_classes, 3
    labels = get_labels(n_matrices, n_classes)
    mats = get_mats(n_matrices, n_channels, "spd")

    clf = MDM(metric={"mean": metric_mean, "distance": metric_dist})
    clf.fit(mats, labels).predict(mats)
    assert clf.covmeans_.shape == (n_classes, n_channels, n_channels)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", [
    "euclid", "logchol", "logeuclid", "riemann"
])
def test_mdm_hpd(kind, metric, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 4, 2
    labels = get_labels(n_matrices, n_classes)
    mats = get_mats(n_matrices, n_channels, kind)

    clf = MDM(metric="riemann")
    clf.fit(mats, labels).predict_proba(mats)


@pytest.mark.parametrize("metric_mean", get_means())
@pytest.mark.parametrize("metric_dist", get_distances())
@pytest.mark.parametrize("metric_map", [
    "euclid", "logchol", "logeuclid", "riemann", "wasserstein"
])
def test_fgmdm(metric_mean, metric_dist, metric_map, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 4, 3, 2
    labels = get_labels(n_matrices, n_classes)
    mats = get_mats(n_matrices, n_channels, "spd")

    clf = FgMDM(metric={
        "mean": metric_mean,
        "distance": metric_dist,
        "map": metric_map
    })
    clf.fit(mats, labels).predict(mats)


@pytest.mark.parametrize("k", [1, 3, 4])
def test_knn(k, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 9, 3, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)

    knn = KNearestNeighbor(k, metric="riemann")
    knn.fit(mats, labels)
    assert knn.covmeans_.shape == (n_matrices, n_channels, n_channels)
    assert knn.classmeans_.shape == (n_matrices,)

    preds = knn.predict(mats)
    if k == 1:
        assert_array_equal(labels, preds)


@pytest.mark.parametrize("metric_mean", get_means())
@pytest.mark.parametrize("metric_map", [
    "euclid", "logchol", "logeuclid", "riemann", "wasserstein"
])
def test_tsclassifier(metric_mean, metric_map, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 4, 3, 2
    labels = get_labels(n_matrices, n_classes)
    mats = get_mats(n_matrices, n_channels, "spd")

    clf = TSClassifier(metric={"mean": metric_mean, "map": metric_map})
    clf.fit(mats, labels).predict(mats)


def test_tsclassifier_fit(get_mats, get_labels):
    """Test TS Classifier"""
    n_matrices, n_channels, n_classes = 6, 3, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)

    clf = TSClassifier(clf=DummyClassifier())
    clf.fit(mats, labels).predict(mats)


def test_tsclassifier_clf_error(get_mats, get_labels):
    """Test TS if not Classifier"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)
    with pytest.raises(TypeError):
        TSClassifier(clf=Covariances()).fit(mats, labels)


def test_svc_params():
    rsvc = SVC()
    assert rsvc.metric == "riemann"

    rsvc.set_params(**{"metric": "logeuclid"})
    assert rsvc.metric == "logeuclid"

    rsvc.set_params(**{"max_iter": 501})
    assert rsvc.max_iter == 501


def test_svc_params_error(get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)

    with pytest.raises(TypeError):
        SVC(C="hello").fit(mats, labels)

    with pytest.raises(TypeError):
        SVC(foo=5)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_cref_metric(get_mats, get_labels, metric):
    n_matrices, n_channels, n_classes = 6, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)
    Cref = mean_covariance(mats, metric=metric)

    rsvc = SVC(Cref=Cref).fit(mats, labels)
    rsvc_1 = SVC(Cref=None, metric=metric).fit(mats, labels)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_cref_callable(get_mats, get_labels, metric):
    n_matrices, n_channels, n_classes = 6, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)
    def Cref(X): return mean_covariance(X, metric=metric)

    rsvc = SVC(Cref=Cref).fit(mats, labels)
    rsvc_1 = SVC(metric=metric).fit(mats, labels)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)

    rsvc = SVC(Cref=Cref).fit(mats, labels)
    rsvc.predict(mats)
    rsvc_1 = SVC(metric=metric).fit(mats, labels)
    rsvc_1.predict(mats)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_cref_error(get_mats, get_labels, metric):
    n_matrices, n_channels, n_classes = 6, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)
    def Cref(X, met): mean_covariance(X, metric=met)

    with pytest.raises(TypeError):
        SVC(Cref=Cref).fit(mats, labels)

    Cref = metric

    with pytest.raises(TypeError):
        SVC(Cref=Cref).fit(mats, labels)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_kernel_callable(get_mats, get_labels, metric):
    n_matrices, n_channels, n_classes = 6, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)

    rsvc = SVC(kernel_fct=kernel, metric=metric).fit(mats, labels)
    rsvc_1 = SVC(metric=metric).fit(mats, labels)
    p1 = rsvc.predict(mats[:-1])
    p2 = rsvc_1.predict(mats[:-1])
    assert np.array_equal(p1, p2)

    def custom_kernel(X, Y, Cref, metric):
        return np.ones((len(X), len(Y)))
    SVC(kernel_fct=custom_kernel,
        metric=metric).fit(mats, labels).predict(mats[:-1])

    # check if pickleable
    pickle.dumps(rsvc)
    pickle.dumps(rsvc_1)


@pytest.mark.parametrize("kernel_fct", [None, "precomputed"])
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_kernel_precomputed(get_mats, get_labels, kernel_fct, metric):
    n_matrices, n_channels, n_classes = 6, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)

    SVC(kernel_fct=kernel_fct, metric=metric).fit(mats, labels)


def test_svc_kernel_error(get_mats, get_labels):
    n_matrices, n_channels, n_classes = 4, 2, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)

    def custom_kernel(X, Y, Cref):
        return np.ones((len(X), len(Y)))
    with pytest.raises(TypeError):
        SVC(kernel_fct=custom_kernel, metric="euclid").fit(mats, labels)

    custom_kernel = np.array([1, 2])
    with pytest.raises(TypeError):
        SVC(kernel_fct=custom_kernel, metric="riemann").fit(mats, labels)


@pytest.mark.parametrize("power_list", [[-1, 0, 1], [0, 0.1]])
@pytest.mark.parametrize("method_label", ["sum_means", "inf_means"])
@pytest.mark.parametrize("metric", get_distances())
def test_meanfield(get_mats, get_labels, power_list, method_label, metric):
    n_powers = len(power_list)
    n_matrices, n_channels, n_classes = 8, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)

    mf = MeanField(
        power_list=power_list,
        method_label=method_label,
        metric=metric,
    ).fit(mats, labels)
    covmeans = mf.covmeans_
    assert len(covmeans) == n_powers
    assert len(covmeans[power_list[0]]) == n_classes
    assert covmeans[power_list[0]][labels[0]].shape == (n_channels, n_channels)

    proba = mf.predict_proba(mats)
    assert proba.shape == (n_matrices, n_classes)
    transf = mf.transform(mats)
    assert transf.shape == (n_matrices, n_classes)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("n_classes", [1, 2, 3])
@pytest.mark.parametrize("metric_mean", get_means())
@pytest.mark.parametrize("metric_dist", get_distances())
@pytest.mark.parametrize("exponent", [1, 2])
def test_class_distinctiveness(kind, n_classes, metric_mean, metric_dist,
                               exponent, get_mats, get_labels):
    """Test function for class distinctiveness measure for two class problem"""
    n_matrices, n_channels = 6, 3
    mats = get_mats(n_matrices, n_channels, kind)
    labels = get_labels(n_matrices, n_classes)

    if n_classes == 1:
        with pytest.raises(ValueError):
            class_distinctiveness(mats, labels)
        return

    class_dis, num, denom = class_distinctiveness(
        mats,
        labels,
        exponent,
        metric={"mean": metric_mean, "distance": metric_dist},
        return_num_denom=True
    )
    assert class_dis >= 0  # negative class_dis value
    assert num >= 0  # negative numerator value
    assert denom >= 0  # negative denominator value
    assert isinstance(class_dis, float), "Unexpected object of class_dis"
    assert isinstance(num, float), "Unexpected object of num"
    assert isinstance(denom, float), "Unexpected object of denum"
