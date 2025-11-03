import pickle

import numpy as np
from numpy.testing import assert_array_equal
import pytest
from pytest import approx
from scipy.spatial.distance import euclidean
from scipy.stats import mode
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline

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
    X = get_mats(n_matrices, n_channels, kind)
    y = get_labels(n_matrices, n_classes)
    weights = get_weights(n_matrices)

    clf_fit(classif, X, y, weights)
    clf_predict(classif, X, y)
    clf_fit_independence(classif, X, y)
    clf_predict_proba(classif, X, y)
    clf_score(classif, X, y)
    clf_populate_classes(classif, X, y)
    if classif in (MDM, FgMDM, MeanField):
        clf_transform(classif, X, y)
        clf_fittransform(classif, X, y)
    if hasattr(classif(), "n_jobs"):
        clf_jobs(classif, X, y)
    if hasattr(classif(), "tsupdate"):
        clf_tsupdate(classif, X, y)
    clf_pipeline(classif, y, get_mats)


def clf_fit(classif, X, y, weights):
    n_classes = len(np.unique(y))
    clf = classif().fit(X, y)
    assert clf.classes_.shape == (n_classes,)
    assert_array_equal(clf.classes_, np.unique(y))

    clf.fit(X, y, sample_weight=weights)


def clf_predict(classif, X, y):
    n_matrices = len(y)
    clf = classif()
    pred = clf.fit(X, y).predict(X)
    assert pred.shape == (n_matrices,)


def clf_predict_proba(classif, X, y):
    n_matrices, n_classes = len(y), len(np.unique(y))
    clf = classif()
    if hasattr(clf, "probability"):
        clf.set_params(**{"probability": True})
    proba = clf.fit(X, y).predict_proba(X)
    assert proba.shape == (n_matrices, n_classes)
    assert proba.sum(axis=1) == approx(np.ones(n_matrices))


def clf_score(classif, X, y):
    clf = classif()
    clf.fit(X, y).score(X, y)


def clf_transform(classif, X, y):
    n_matrices, n_classes = len(y), len(np.unique(y))
    clf = classif()
    transf = clf.fit(X, y).transform(X)
    assert transf.shape == (n_matrices, n_classes)


def clf_fittransform(classif, X, y):
    clf = classif()
    transf = clf.fit(X, y).transform(X)
    transf2 = clf.fit_transform(X, y)
    assert_array_equal(transf, transf2)


def clf_fit_independence(classif, X, y):
    clf = classif()
    clf.fit(X, y).predict(X)
    # retraining with different size should erase previous fit
    Xnew = X[:, :-1, :-1]
    clf.fit(Xnew, y).predict(Xnew)


def clf_jobs(classif, X, y):
    clf = classif(n_jobs=2)
    clf.fit(X, y).predict(X)


def clf_populate_classes(classif, X, y):
    clf = classif()
    clf.fit(X, y)
    assert_array_equal(clf.classes_, np.unique(y))


def clf_tsupdate(classif, X, y):
    clf = classif(tsupdate=True)
    clf.fit(X, y).predict(X)


def clf_pipeline(classif, labels, get_mats):
    n_matrices, n_channels, n_times = len(labels), 3, 100
    X = get_mats(n_matrices, [n_channels, n_times], "real")

    clf = classif()
    if hasattr(clf, "probability"):
        clf.set_params(**{"probability": True})

    pip = make_pipeline(Covariances(estimator="scm"), clf)
    pip.fit(X, labels)
    pip.predict(X)
    pip.predict_proba(X)
    cross_val_score(
        pip, X, labels,
        cv=KFold(n_splits=3), scoring="accuracy", n_jobs=1
    )


@pytest.mark.parametrize("classif", classifs)
@pytest.mark.parametrize("mean", ["faulty", 42])
@pytest.mark.parametrize("dist", ["not_real", 27])
def test_metric_dict_error(classif, mean, dist, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)
    clf = classif(metric={"mean": mean, "distance": dist})
    with pytest.raises((TypeError, KeyError, ValueError)):
        clf.fit(X, y).predict(X)


@pytest.mark.parametrize("classif", classifs)
@pytest.mark.parametrize("metric", [42, "faulty", {"foo": "bar"}])
def test_metric_errors(classif, metric, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)
    clf = classif(metric=metric)
    with pytest.raises((TypeError, KeyError, ValueError)):
        clf.fit(X, y).predict(X)


@pytest.mark.parametrize("classif", classifs)
@pytest.mark.parametrize("metric", get_metrics())
def test_metric_str(classif, metric, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)

    clf = classif(metric=metric)
    if classif in [SVC, FgMDM, TSClassifier] \
            and metric not in ["euclid", "logchol", "logeuclid", "riemann"]:
        with pytest.raises((KeyError, ValueError)):
            clf.fit(X, y).predict(X)
    else:
        clf.fit(X, y).predict(X)


def call_mean(X, sample_weight=None):
    return np.average(X, axis=0, weights=sample_weight)


def call_dist(A, B, squared=False):
    return euclidean(A.flatten(), B.flatten())


@pytest.mark.parametrize("metric_mean", list(get_means()) + [call_mean])
@pytest.mark.parametrize("metric_dist", list(get_distances()) + [call_dist])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
def test_mdm(metric_mean, metric_dist, n_classes, get_mats, get_labels):
    n_matrices, n_channels = 4 * n_classes, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)

    clf = MDM(metric={"mean": metric_mean, "distance": metric_dist})
    clf.fit(X, y).predict(X)
    assert clf.covmeans_.shape == (n_classes, n_channels, n_channels)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", [
    "euclid", "logchol", "logeuclid", "riemann"
])
def test_mdm_hpd(kind, metric, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 4, 2
    X = get_mats(n_matrices, n_channels, kind)
    y = get_labels(n_matrices, n_classes)

    clf = MDM(metric="riemann")
    clf.fit(X, y).predict_proba(X)


@pytest.mark.parametrize("metric_mean", get_means())
@pytest.mark.parametrize("metric_dist", get_distances())
@pytest.mark.parametrize("metric_map", [
    "euclid", "logchol", "logeuclid", "riemann", "wasserstein"
])
def test_fgmdm(metric_mean, metric_dist, metric_map, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 4, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)

    clf = FgMDM(metric={
        "mean": metric_mean,
        "distance": metric_dist,
        "map": metric_map
    })
    clf.fit(X, y).predict(X)


@pytest.mark.parametrize("k", [1, 3, 4])
def test_knn(k, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 9, 3, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)

    knn = KNearestNeighbor(k, metric="riemann")
    knn.fit(X, y)
    assert knn.covmeans_.shape == (n_matrices, n_channels, n_channels)
    assert knn.classmeans_.shape == (n_matrices,)

    preds = knn.predict(X)
    if k == 1:
        assert_array_equal(y, preds)


@pytest.mark.parametrize("metric_mean", get_means())
@pytest.mark.parametrize("metric_map", [
    "euclid", "logchol", "logeuclid", "riemann", "wasserstein"
])
def test_tsclassifier_metrics(metric_mean, metric_map, get_mats, get_labels):
    n_matrices, n_channels, n_classes = 4, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)

    clf = TSClassifier(metric={"mean": metric_mean, "map": metric_map})
    clf.fit(X, y).predict(X)


def test_tsclassifier_fit(get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)

    clf = TSClassifier(clf=DummyClassifier())
    clf.fit(X, y).predict(X)


def test_tsclassifier_clf_error(get_mats, get_labels):
    """Test TS if not Classifier"""
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)
    with pytest.raises(TypeError):
        TSClassifier(clf=Covariances()).fit(X, y)


def test_svc_params():
    rsvc = SVC()
    assert rsvc.metric == "riemann"

    rsvc.set_params(**{"metric": "logeuclid"})
    assert rsvc.metric == "logeuclid"

    rsvc.set_params(**{"max_iter": 501})
    assert rsvc.max_iter == 501


def test_svc_params_error(get_mats, get_labels):
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)

    with pytest.raises(TypeError):
        SVC(C="hello").fit(X, y)

    with pytest.raises(TypeError):
        SVC(foo=5)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_cref_metric(get_mats, get_labels, metric):
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)
    Cref = mean_covariance(X, metric=metric)

    rsvc = SVC(Cref=Cref).fit(X, y)
    rsvc_1 = SVC(Cref=None, metric=metric).fit(X, y)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_cref_callable(get_mats, get_labels, metric):
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)
    def Cref(X): return mean_covariance(X, metric=metric)

    rsvc = SVC(Cref=Cref).fit(X, y)
    rsvc_1 = SVC(metric=metric).fit(X, y)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)

    rsvc = SVC(Cref=Cref).fit(X, y)
    rsvc.predict(X)
    rsvc_1 = SVC(metric=metric).fit(X, y)
    rsvc_1.predict(X)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_cref_error(get_mats, get_labels, metric):
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)

    def Cref(X, met): mean_covariance(X, metric=met)
    with pytest.raises(TypeError):
        SVC(Cref=Cref).fit(X, y)

    with pytest.raises(TypeError):
        SVC(Cref=metric).fit(X, y)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_kernel_callable(get_mats, get_labels, metric):
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)

    rsvc = SVC(kernel_fct=kernel, metric=metric).fit(X, y)
    rsvc_1 = SVC(metric=metric).fit(X, y)
    p1 = rsvc.predict(X[:-1])
    p2 = rsvc_1.predict(X[:-1])
    assert np.array_equal(p1, p2)

    def custom_kernel(X, Y, Cref, metric):
        return np.ones((len(X), len(Y)))
    SVC(kernel_fct=custom_kernel,
        metric=metric).fit(X, y).predict(X[:-1])

    # check if pickleable
    pickle.dumps(rsvc)
    pickle.dumps(rsvc_1)


@pytest.mark.parametrize("kernel_fct", [None, "precomputed"])
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_kernel_precomputed(get_mats, get_labels, kernel_fct, metric):
    n_matrices, n_channels, n_classes = 6, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)

    SVC(kernel_fct=kernel_fct, metric=metric).fit(X, y)


def test_svc_kernel_error(get_mats, get_labels):
    n_matrices, n_channels, n_classes = 4, 2, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)

    def custom_kernel(X, Y, Cref):
        return np.ones((len(X), len(Y)))
    with pytest.raises(TypeError):
        SVC(kernel_fct=custom_kernel, metric="euclid").fit(X, y)

    custom_kernel = np.array([1, 2])
    with pytest.raises(TypeError):
        SVC(kernel_fct=custom_kernel, metric="riemann").fit(X, y)


@pytest.mark.parametrize("power_list", [[-1, 0, 1], [0, 0.1]])
@pytest.mark.parametrize("method_combination", [
    "sum_means", "inf_means", None
])
@pytest.mark.parametrize("metric", get_distances())
def test_meanfield(get_mats, get_labels,
                   power_list, method_combination, metric):
    n_powers = len(power_list)
    n_matrices, n_channels, n_classes = 8, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)

    mf = MeanField(
        power_list=power_list,
        method_combination=method_combination,
        metric=metric,
    ).fit(X, y)
    assert mf.covmeans_.shape == (n_classes, n_powers, n_channels, n_channels)

    transf = mf.transform(X)
    if method_combination is None:
        assert transf.shape == (n_matrices, n_classes * n_powers)
    else:
        assert transf.shape == (n_matrices, n_classes)
        pred = mf.predict(X)
        assert pred.shape == (n_matrices,)
        proba = mf.predict_proba(X)
        assert proba.shape == (n_matrices, n_classes)
        mf.score(X, y)


@pytest.mark.parametrize("power_list", [[-1, 0, 1], [0, 0.1]])
def test_meanfield_transformer(get_mats, get_labels, power_list):
    n_matrices, n_channels, n_classes = 8, 3, 2
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)

    pip = make_pipeline(
        MeanField(power_list=power_list, method_combination=None),
        LDA(),
    )
    pip.fit(X, y)
    pip.predict(X)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("n_classes", [1, 2, 3])
@pytest.mark.parametrize("metric_mean", get_means())
@pytest.mark.parametrize("metric_dist", get_distances())
@pytest.mark.parametrize("exponent", [1, 2])
def test_class_distinctiveness(kind, n_classes, metric_mean, metric_dist,
                               exponent, get_mats, get_labels):
    """Test function for class distinctiveness measure for two class problem"""
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, kind)
    y = get_labels(n_matrices, n_classes)

    if n_classes == 1:
        with pytest.raises(ValueError):
            class_distinctiveness(X, y)
        return

    class_dis, num, denom = class_distinctiveness(
        X,
        y,
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
