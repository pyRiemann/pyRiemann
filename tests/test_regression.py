import pickle

import numpy as np
from numpy.testing import assert_array_equal
import pytest
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline

from conftest import get_distances, get_means, get_metrics
from pyriemann.estimation import Covariances
from pyriemann.regression import (SVR, KNearestNeighborRegressor)
from pyriemann.utils.kernel import kernel
from pyriemann.utils.mean import mean_covariance

regress = [SVR, KNearestNeighborRegressor]


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("regres", regress)
def test_regression(kind, regres, get_mats, get_targets, get_weights, rndstate):
    if kind == "hpd" and regres is SVR:
        pytest.skip()
    n_matrices, n_channels = 6, 3
    mats = get_mats(n_matrices, n_channels, kind)
    targets = get_targets(n_matrices)
    weights = get_weights(n_matrices)

    reg_fit(regres, mats, targets, weights)
    reg_predict(regres, mats, targets)
    reg_fit_independence(regres, mats, targets)
    reg_score(regres, mats, targets)
    reg_pipeline(regres, targets, rndstate)


def reg_fit(regres, mats, targets, weights):
    n_matrices, n_channels, _ = mats.shape
    reg = regres().fit(mats, targets)

    if reg is SVR:
        assert reg.data_.shape == (n_matrices, n_channels, n_channels)
    elif reg is KNearestNeighborRegressor:
        assert reg.covmeans_.shape == (n_matrices, n_channels, n_channels)
        assert reg.values_.shape == (n_matrices,)

    reg.fit(mats, targets, sample_weight=weights)


def reg_predict(regres, mats, targets):
    reg = regres()
    predicted = reg.fit(mats, targets).predict(mats)
    assert predicted.shape == (len(targets),)


def reg_fit_independence(regres, mats, targets):
    reg = regres()
    reg.fit(mats, targets).predict(mats)
    # retraining with different size should erase previous fit
    new_mats = mats[:, :-1, :-1]
    reg.fit(new_mats, targets).predict(new_mats)


def reg_score(regres, mats, targets):
    reg = regres()
    reg.fit(mats, targets).score(mats, targets)


def reg_pipeline(regres, targets, rndstate):
    n_matrices, n_channels, n_times = len(targets), 3, 100
    epochs = rndstate.randn(n_matrices, n_channels, n_times)

    pip = make_pipeline(Covariances(estimator="scm"), regres())
    pip.fit(epochs, targets)
    pip.predict(epochs)
    cross_val_score(
        pip, epochs, targets,
        cv=KFold(n_splits=2), scoring="r2", n_jobs=1
    )


@pytest.mark.parametrize("regres", [KNearestNeighborRegressor])
@pytest.mark.parametrize("mean", ["faulty", 42])
@pytest.mark.parametrize("dist", ["not_real", 27])
def test_metric_dict_error(regres, mean, dist, get_mats, get_targets):
    n_matrices, n_channels = 6, 3
    targets = get_targets(n_matrices)
    mats = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):
        clf = regres(metric={"mean": mean, "distance": dist})
        clf.fit(mats, targets).predict(mats)


@pytest.mark.parametrize("regres", [KNearestNeighborRegressor])
@pytest.mark.parametrize("mean", get_means())
@pytest.mark.parametrize("dist", get_distances())
def test_metric_dist(regres, mean, dist, get_mats, get_targets):
    n_matrices, n_channels = 4, 3
    targets = get_targets(n_matrices)
    mats = get_mats(n_matrices, n_channels, "spd")
    clf = regres(metric={"mean": mean, "distance": dist})
    clf.fit(mats, targets).predict(mats)


@pytest.mark.parametrize("regres", regress)
@pytest.mark.parametrize("metric", [42, "faulty", {"foo": "bar"}])
def test_metric_wrong_keys(regres, metric, get_mats, get_targets):
    n_matrices, n_channels = 6, 3
    targets = get_targets(n_matrices)
    mats = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises((TypeError, KeyError, ValueError)):
        clf = regres(metric=metric)
        clf.fit(mats, targets).predict(mats)


@pytest.mark.parametrize("regres", regress)
@pytest.mark.parametrize("metric", get_metrics())
def test_metric_str(regres, metric, get_mats, get_targets):
    n_matrices, n_channels = 6, 3
    targets = get_targets(n_matrices)
    mats = get_mats(n_matrices, n_channels, "spd")
    clf = regres(metric=metric)

    if regres is SVR and metric not in ["euclid", "logchol",
                                        "logeuclid", "riemann"]:
        with pytest.raises(ValueError):
            clf.fit(mats, targets).predict(mats)

    else:
        clf.fit(mats, targets).predict(mats)


@pytest.mark.parametrize("dist", ["not_real", 42])
def test_knn_dict_dist(dist, get_mats, get_targets):
    n_matrices, n_channels = 6, 3
    targets = get_targets(n_matrices)
    mats = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(KeyError):
        clf = KNearestNeighborRegressor(metric={"distance": dist})
        clf.fit(mats, targets).predict(mats)


def test_1NN(get_mats, get_targets):
    """Test KNearestNeighborRegressor with K=1"""
    n_matrices, n_channels = 9, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    targets = get_targets(n_matrices)

    knn = KNearestNeighborRegressor(1, metric="riemann")
    knn.fit(mats, targets)
    preds = knn.predict(mats[:-1])
    assert_array_equal(targets[:-1], preds)


def test_svr_params():
    rsvr = SVR()
    assert rsvr.metric == "riemann"

    rsvr.set_params(**{"metric": "logeuclid"})
    assert rsvr.metric == "logeuclid"

    rsvr.set_params(**{"max_iter": 501})
    assert rsvr.max_iter == 501


def test_svr_params_error(get_mats, get_targets):
    n_matrices, n_channels = 6, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    targets = get_targets(n_matrices)

    with pytest.raises(TypeError):
        SVR(C='hello').fit(mats, targets)

    with pytest.raises(TypeError):
        SVR(foo=5)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svr_cref_metric(get_mats, get_targets, metric):
    n_matrices, n_channels = 6, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    targets = get_targets(n_matrices)
    Cref = mean_covariance(mats, metric=metric)

    rsvc = SVR(Cref=Cref).fit(mats, targets)
    rsvc_1 = SVR(Cref=None, metric=metric).fit(mats, targets)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_cref_callable(get_mats, get_targets, metric):
    n_matrices, n_channels = 6, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    targets = get_targets(n_matrices)
    def Cref(X): return mean_covariance(X, metric=metric)

    rsvc = SVR(Cref=Cref).fit(mats, targets)
    rsvc_1 = SVR(metric=metric).fit(mats, targets)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)

    rsvc = SVR(Cref=Cref).fit(mats, targets)
    rsvc.predict(mats)
    rsvc_1 = SVR(metric=metric).fit(mats, targets)
    rsvc_1.predict(mats)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_cref_error(get_mats, get_targets, metric):
    n_matrices, n_channels = 6, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    targets = get_targets(n_matrices)
    def Cref(X, met): mean_covariance(X, metric=met)

    with pytest.raises(TypeError):
        SVR(Cref=Cref).fit(mats, targets)

    Cref = metric

    with pytest.raises(TypeError):
        SVR(Cref=Cref).fit(mats, targets)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_kernel_callable(get_mats, get_targets, metric):
    n_matrices, n_channels = 6, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    targets = get_targets(n_matrices)

    rsvc = SVR(kernel_fct=kernel,
               metric=metric).fit(mats, targets)
    rsvc_1 = SVR(metric=metric).fit(mats, targets)
    p1 = rsvc.predict(mats[:-1])
    p2 = rsvc_1.predict(mats[:-1])
    assert np.array_equal(p1, p2)

    def custom_kernel(X, Y, Cref, metric):
        return np.ones((len(X), len(Y)))
    SVR(kernel_fct=custom_kernel,
        metric=metric).fit(mats, targets).predict(mats[:-1])

    def custom_kernel(X, Y, Cref):
        return np.ones((len(X), len(Y)))
    with pytest.raises(TypeError):
        SVR(kernel_fct=custom_kernel, metric=metric).fit(mats, targets)

    custom_kernel = np.array([1, 2])
    with pytest.raises(TypeError):
        SVR(kernel_fct=custom_kernel, metric=metric).fit(mats, targets)

    # check if pickleable
    pickle.dumps(rsvc)
    pickle.dumps(rsvc_1)
