import pickle

import numpy as np
from numpy.testing import assert_array_equal
import pytest
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline

from pyriemann.estimation import Covariances
from pyriemann.regression import SVR, KNearestNeighborRegressor
from pyriemann.utils.kernel import kernel
from pyriemann.utils.mean import mean_covariance

regress = [SVR, KNearestNeighborRegressor]


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("regres", regress)
def test_regression(kind, regres, get_mats, get_targets, get_weights):
    if kind == "hpd" and regres is SVR:
        pytest.skip()
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, kind)
    y = get_targets(n_matrices)
    weights = get_weights(n_matrices)

    reg_fit(regres, X, y, weights)
    reg_predict(regres, X, y)
    reg_fit_independence(regres, X, y)
    reg_score(regres, X, y)
    reg_pipeline(regres, y, get_mats, kind)


def reg_fit(regres, X, y, weights):
    n_matrices, n_channels, _ = X.shape
    reg = regres().fit(X, y)

    if reg is SVR:
        assert reg.data_.shape == (n_matrices, n_channels, n_channels)
    elif reg is KNearestNeighborRegressor:
        assert reg.covmeans_.shape == (n_matrices, n_channels, n_channels)
        assert reg.values_.shape == (n_matrices,)

    reg.fit(X, y, sample_weight=weights)


def reg_predict(regres, X, y):
    reg = regres()
    pred = reg.fit(X, y).predict(X)
    assert pred.shape == (len(y),)


def reg_fit_independence(regres, X, y):
    reg = regres()
    reg.fit(X, y).predict(X)
    # retraining with different size should erase previous fit
    Xnew = X[:, :-1, :-1]
    reg.fit(Xnew, y).predict(Xnew)


def reg_score(regres, X, y):
    reg = regres()
    score = reg.fit(X, y).score(X, y)
    assert isinstance(score, float)


def reg_pipeline(regres, y, get_mats, kind):
    n_matrices, n_channels, n_times = len(y), 3, 100
    kind_e = {"spd": "real", "hpd": "comp"}
    epochs = get_mats(n_matrices, [n_channels, n_times], kind_e.get(kind))

    pip = make_pipeline(Covariances(estimator="scm"), regres())
    pip.fit(epochs, y)
    pip.predict(epochs)
    cross_val_score(
        pip, epochs, y,
        cv=KFold(n_splits=2), scoring="r2", n_jobs=1
    )


@pytest.mark.parametrize("regres", [KNearestNeighborRegressor])
@pytest.mark.parametrize("mean", ["faulty", 42])
@pytest.mark.parametrize("dist", ["not_real", 27])
def test_metric_dict_error(regres, mean, dist, get_mats, get_targets):
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_targets(n_matrices)
    with pytest.raises(ValueError):
        reg = regres(metric={"mean": mean, "distance": dist})
        reg.fit(X, y).predict(X)


@pytest.mark.parametrize("regres", [KNearestNeighborRegressor])
@pytest.mark.parametrize("mean", ["euclid", "logeuclid", "riemann"])
@pytest.mark.parametrize("dist", ["euclid", "logeuclid", "riemann"])
def test_metric_dist(regres, mean, dist, get_mats, get_targets):
    n_matrices, n_channels = 4, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_targets(n_matrices)
    reg = regres(metric={"mean": mean, "distance": dist})
    reg.fit(X, y).predict(X)


@pytest.mark.parametrize("regres", regress)
@pytest.mark.parametrize("metric", [42, "faulty", {"foo": "bar"}])
def test_metric_wrong_keys(regres, metric, get_mats, get_targets):
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_targets(n_matrices)
    with pytest.raises((TypeError, KeyError, ValueError)):
        reg = regres(metric=metric)
        reg.fit(X, y).predict(X)


@pytest.mark.parametrize("regres", regress)
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_metric_str(regres, metric, get_mats, get_targets):
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_targets(n_matrices)
    reg = regres(metric=metric)
    reg.fit(X, y).predict(X)


@pytest.mark.parametrize("dist", ["not_real", 42])
def test_knn_dict_dist(dist, get_mats, get_targets):
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_targets(n_matrices)
    with pytest.raises(KeyError):
        knn = KNearestNeighborRegressor(metric={"distance": dist})
        knn.fit(X, y).predict(X)


def test_1nn(get_mats, get_targets):
    """Test KNearestNeighborRegressor with K=1"""
    n_matrices, n_channels = 9, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_targets(n_matrices)

    knn = KNearestNeighborRegressor(1, metric="riemann")
    knn.fit(X, y)
    preds = knn.predict(X[:-1])
    assert_array_equal(y[:-1], preds)


def test_svr_params():
    rsvr = SVR()
    assert rsvr.metric == "riemann"

    rsvr.set_params(**{"metric": "logeuclid"})
    assert rsvr.metric == "logeuclid"

    rsvr.set_params(**{"max_iter": 501})
    assert rsvr.max_iter == 501


def test_svr_params_error(get_mats, get_targets):
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_targets(n_matrices)

    with pytest.raises(TypeError):
        SVR(C='hello').fit(X, y)

    with pytest.raises(TypeError):
        SVR(foo=5)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svr_cref_metric(get_mats, get_targets, metric):
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_targets(n_matrices)
    Cref = mean_covariance(X, metric=metric)

    rsvc = SVR(Cref=Cref).fit(X, y)
    rsvc_1 = SVR(Cref=None, metric=metric).fit(X, y)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_cref_callable(get_mats, get_targets, metric):
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_targets(n_matrices)
    def Cref(X): return mean_covariance(X, metric=metric)

    rsvc = SVR(Cref=Cref).fit(X, y)
    rsvc_1 = SVR(metric=metric).fit(X, y)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)

    rsvc = SVR(Cref=Cref).fit(X, y)
    rsvc.predict(X)
    rsvc_1 = SVR(metric=metric).fit(X, y)
    rsvc_1.predict(X)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_cref_error(get_mats, get_targets, metric):
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_targets(n_matrices)
    def Cref(X, met): mean_covariance(X, metric=met)

    with pytest.raises(TypeError):
        SVR(Cref=Cref).fit(X, y)

    Cref = metric
    with pytest.raises(TypeError):
        SVR(Cref=Cref).fit(X, y)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_svc_kernel_callable(get_mats, get_targets, metric):
    n_matrices, n_channels = 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_targets(n_matrices)

    rsvc = SVR(kernel_fct=kernel, metric=metric).fit(X, y)
    rsvc_1 = SVR(metric=metric).fit(X, y)
    p1 = rsvc.predict(X[:-1])
    p2 = rsvc_1.predict(X[:-1])
    assert np.array_equal(p1, p2)

    def custom_kernel(X, Y, Cref, metric):
        return np.ones((len(X), len(Y)))
    SVR(kernel_fct=custom_kernel, metric=metric).fit(X, y).predict(X[:-1])

    def custom_kernel(X, Y, Cref):
        return np.ones((len(X), len(Y)))
    with pytest.raises(TypeError):
        SVR(kernel_fct=custom_kernel, metric=metric).fit(X, y)

    custom_kernel = np.array([1, 2])
    with pytest.raises(TypeError):
        SVR(kernel_fct=custom_kernel, metric=metric).fit(X, y)

    # check if pickleable
    pickle.dumps(rsvc)
    pickle.dumps(rsvc_1)
