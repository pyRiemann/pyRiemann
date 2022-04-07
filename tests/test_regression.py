from conftest import get_distances, get_means, get_metrics
from numpy.testing import assert_array_equal
from pyriemann.regression import (SVR, KNearestNeighborRegressor)

import pytest


regs = [SVR, KNearestNeighborRegressor]


@pytest.mark.parametrize("regres", regs)
class RegressorTestCase:
    def test_regression(self, regres, get_covmats, get_targets):
        n_matrices, n_channels = 6, 3
        covmats = get_covmats(n_matrices, n_channels)
        targets = get_targets(n_matrices)
        self.reg_predict(regres, covmats, targets)
        self.reg_fit_independence(regres, covmats, targets)


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


@pytest.mark.parametrize("regres", [KNearestNeighborRegressor])
@pytest.mark.parametrize("mean", ["faulty", 42])
@pytest.mark.parametrize("dist", ["not_real", 27])
def test_metric_dict_error(regres, mean, dist, get_covmats, get_targets):
    with pytest.raises((TypeError, KeyError)):
        n_matrices, n_channels = 6, 3
        targets = get_targets(n_matrices)
        covmats = get_covmats(n_matrices, n_channels)
        clf = regres(metric={"mean": mean, "distance": dist})
        clf.fit(covmats, targets).predict(covmats)


@pytest.mark.parametrize("regres", [KNearestNeighborRegressor])
@pytest.mark.parametrize("mean", get_means())
@pytest.mark.parametrize("dist", get_distances())
def test_metric_dist(regres, mean, dist, get_covmats, get_targets):
    n_matrices, n_channels, n_classes = 4, 3, 2
    targets = get_targets(n_matrices)
    covmats = get_covmats(n_matrices, n_channels)
    clf = regres(metric={"mean": mean, "distance": dist})
    clf.fit(covmats, targets).predict(covmats)


@pytest.mark.parametrize("regres", regs)
@pytest.mark.parametrize("metric", [42, "faulty", {"foo": "bar"}])
def test_metric_wrong_keys(regres, metric, get_covmats, get_targets):
    with pytest.raises((TypeError, KeyError, ValueError)):
        n_matrices, n_channels, n_classes = 6, 3, 2
        targets = get_targets(n_matrices)
        covmats = get_covmats(n_matrices, n_channels)
        clf = regres(metric=metric)
        clf.fit(covmats, targets).predict(covmats)


@pytest.mark.parametrize("regres", regs)
@pytest.mark.parametrize("metric", get_metrics())
def test_metric_str(regres, metric, get_covmats, get_targets):
    n_matrices, n_channels, n_classes = 6, 3, 2
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
    with pytest.raises(KeyError):
        n_matrices, n_channels, n_classes = 6, 3, 2
        targets = get_targets(n_matrices)
        covmats = get_covmats(n_matrices, n_channels)
        clf = KNearestNeighborRegressor(metric={"distance": dist})
        clf.fit(covmats, targets).predict(covmats)


def test_1NN(get_covmats, get_targets):
    """Test KNearestNeighborRegressor with K=1"""
    n_matrices, n_channels, n_classes = 9, 3, 3
    covmats = get_covmats(n_matrices, n_channels)
    targets = get_targets(n_matrices)

    knn = KNearestNeighborRegressor(1, metric="riemann")
    knn.fit(covmats, targets)
    preds = knn.predict(covmats)
    assert_array_equal(targets, preds)


def test_supportvectormachine_svr_params():
    rsvr = SVR()
    assert rsvr.metric == 'riemann'

    rsvr.set_params(**{'metric': 'logeuclid'})
    assert rsvr.metric == 'logeuclid'

    rsvr.set_params(**{'max_iter': 501})
    assert rsvr.max_iter == 501


def test_supportvectormachine_svc_params_error(get_covmats, get_targets):
    n_matrices, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    targets = get_targets(n_matrices)

    with pytest.raises(TypeError):
        SVR(C='hello').fit(covmats, targets)

    with pytest.raises(TypeError):
        SVR(foo=5)
