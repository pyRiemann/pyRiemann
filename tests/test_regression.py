import pickle

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from conftest import get_distances, get_means, get_metrics
from pyriemann.regression import (SVR, KNearestNeighborRegressor)
from pyriemann.utils.kernel import kernel
from pyriemann.utils.mean import mean_covariance

regs = [SVR, KNearestNeighborRegressor]


@pytest.mark.parametrize("regres", regs)
class RegressorTestCase:
    def test_regression(self, regres, get_covmats, get_targets):
        n_matrices, n_channels = 6, 3
        covmats = get_covmats(n_matrices, n_channels)
        targets = get_targets(n_matrices)

        self.reg_predict(regres, covmats, targets)
        self.reg_fit_independence(regres, covmats, targets)
        self.reg_score(regres, covmats, targets)


class TestRegressor(RegressorTestCase):
    def reg_predict(self, regres, covmats, targets):
        clf = regres()
        clf.fit(covmats, targets)
        predicted = clf.predict(covmats)
        assert predicted.shape == (len(targets),)

    def reg_fit_independence(self, regres, covmats, targets):
        clf = regres()
        clf.fit(covmats, targets).predict(covmats)
        # retraining with different size should erase previous fit
        new_covmats = covmats[:, :-1, :-1]
        clf.fit(new_covmats, targets).predict(new_covmats)

    def reg_score(self, regres, covmats, targets):
        clf = regres()
        clf.fit(covmats, targets)
        clf.score(covmats, targets)


@pytest.mark.parametrize("regres", [KNearestNeighborRegressor])
@pytest.mark.parametrize("mean", ["faulty", 42])
@pytest.mark.parametrize("dist", ["not_real", 27])
def test_metric_dict_error(regres, mean, dist, get_covmats, get_targets):
    n_matrices, n_channels = 6, 3
    targets = get_targets(n_matrices)
    covmats = get_covmats(n_matrices, n_channels)
    with pytest.raises(ValueError):
        clf = regres(metric={"mean": mean, "distance": dist})
        clf.fit(covmats, targets).predict(covmats)


@pytest.mark.parametrize("regres", [KNearestNeighborRegressor])
@pytest.mark.parametrize("mean", get_means())
@pytest.mark.parametrize("dist", get_distances())
def test_metric_dist(regres, mean, dist, get_covmats, get_targets):
    n_matrices, n_channels = 4, 3
    targets = get_targets(n_matrices)
    covmats = get_covmats(n_matrices, n_channels)
    clf = regres(metric={"mean": mean, "distance": dist})
    clf.fit(covmats, targets).predict(covmats)


@pytest.mark.parametrize("regres", regs)
@pytest.mark.parametrize("metric", [42, "faulty", {"foo": "bar"}])
def test_metric_wrong_keys(regres, metric, get_covmats, get_targets):
    n_matrices, n_channels = 6, 3
    targets = get_targets(n_matrices)
    covmats = get_covmats(n_matrices, n_channels)
    with pytest.raises((TypeError, KeyError, ValueError)):
        clf = regres(metric=metric)
        clf.fit(covmats, targets).predict(covmats)


@pytest.mark.parametrize("regres", regs)
@pytest.mark.parametrize("metric", get_metrics())
def test_metric_str(regres, metric, get_covmats, get_targets):
    n_matrices, n_channels = 6, 3
    targets = get_targets(n_matrices)
    covmats = get_covmats(n_matrices, n_channels)
    clf = regres(metric=metric)

    if regres is SVR and metric not in ['riemann', 'euclid', 'logeuclid']:
        with pytest.raises(ValueError):
            clf.fit(covmats, targets).predict(covmats)

    else:
        clf.fit(covmats, targets).predict(covmats)


@pytest.mark.parametrize("dist", ["not_real", 42])
def test_knn_dict_dist(dist, get_covmats, get_targets):
    n_matrices, n_channels = 6, 3
    targets = get_targets(n_matrices)
    covmats = get_covmats(n_matrices, n_channels)
    with pytest.raises(KeyError):
        clf = KNearestNeighborRegressor(metric={"distance": dist})
        clf.fit(covmats, targets).predict(covmats)


def test_1NN(get_covmats, get_targets):
    """Test KNearestNeighborRegressor with K=1"""
    n_matrices, n_channels = 9, 3
    covmats = get_covmats(n_matrices, n_channels)
    targets = get_targets(n_matrices)

    knn = KNearestNeighborRegressor(1, metric="riemann")
    knn.fit(covmats, targets)
    preds = knn.predict(covmats[:-1])
    assert_array_equal(targets[:-1], preds)


def test_svr_params():
    rsvr = SVR()
    assert rsvr.metric == 'riemann'

    rsvr.set_params(**{'metric': 'logeuclid'})
    assert rsvr.metric == 'logeuclid'

    rsvr.set_params(**{'max_iter': 501})
    assert rsvr.max_iter == 501


def test_svr_params_error(get_covmats, get_targets):
    n_matrices, n_channels = 6, 3
    covmats = get_covmats(n_matrices, n_channels)
    targets = get_targets(n_matrices)

    with pytest.raises(TypeError):
        SVR(C='hello').fit(covmats, targets)

    with pytest.raises(TypeError):
        SVR(foo=5)


@pytest.mark.parametrize("metric", ["riemann", "euclid", "logeuclid"])
def test_svr_cref_metric(get_covmats, get_targets, metric):
    n_matrices, n_channels = 6, 3
    covmats = get_covmats(n_matrices, n_channels)
    targets = get_targets(n_matrices)
    Cref = mean_covariance(covmats, metric=metric)

    rsvc = SVR(Cref=Cref).fit(covmats, targets)
    rsvc_1 = SVR(Cref=None, metric=metric).fit(covmats, targets)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)


@pytest.mark.parametrize("metric", ["riemann", "euclid", "logeuclid"])
def test_svc_cref_callable(get_covmats, get_targets, metric):
    n_matrices, n_channels = 6, 3
    covmats = get_covmats(n_matrices, n_channels)
    targets = get_targets(n_matrices)
    def Cref(X): return mean_covariance(X, metric=metric)

    rsvc = SVR(Cref=Cref).fit(covmats, targets)
    rsvc_1 = SVR(metric=metric).fit(covmats, targets)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)

    rsvc = SVR(Cref=Cref).fit(covmats, targets)
    rsvc.predict(covmats)
    rsvc_1 = SVR(metric=metric).fit(covmats, targets)
    rsvc_1.predict(covmats)
    assert np.array_equal(rsvc.Cref_, rsvc_1.Cref_)


@pytest.mark.parametrize("metric", ["riemann", "euclid", "logeuclid"])
def test_svc_cref_error(get_covmats, get_targets, metric):
    n_matrices, n_channels = 6, 3
    covmats = get_covmats(n_matrices, n_channels)
    targets = get_targets(n_matrices)
    def Cref(X, met): mean_covariance(X, metric=met)

    with pytest.raises(TypeError):
        SVR(Cref=Cref).fit(covmats, targets)

    Cref = metric

    with pytest.raises(TypeError):
        SVR(Cref=Cref).fit(covmats, targets)


@pytest.mark.parametrize("metric", ["riemann", "euclid", "logeuclid"])
def test_svc_kernel_callable(get_covmats, get_targets, metric):
    n_matrices, n_channels = 6, 3
    covmats = get_covmats(n_matrices, n_channels)
    targets = get_targets(n_matrices)

    rsvc = SVR(kernel_fct=kernel,
               metric=metric).fit(covmats, targets)
    rsvc_1 = SVR(metric=metric).fit(covmats, targets)
    p1 = rsvc.predict(covmats[:-1])
    p2 = rsvc_1.predict(covmats[:-1])
    assert np.array_equal(p1, p2)

    def custom_kernel(X, Y, Cref, metric):
        return np.ones((len(X), len(Y)))
    SVR(kernel_fct=custom_kernel,
        metric=metric).fit(covmats, targets).predict(covmats[:-1])

    def custom_kernel(X, Y, Cref):
        return np.ones((len(X), len(Y)))
    with pytest.raises(TypeError):
        SVR(kernel_fct=custom_kernel, metric=metric).fit(covmats, targets)

    custom_kernel = np.array([1, 2])
    with pytest.raises(TypeError):
        SVR(kernel_fct=custom_kernel, metric=metric).fit(covmats, targets)

    # check if pickleable
    pickle.dumps(rsvc)
    pickle.dumps(rsvc_1)
