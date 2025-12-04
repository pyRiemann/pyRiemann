import numpy as np
from numpy.testing import assert_array_equal
import pytest
from pytest import approx

from pyriemann.tangentspace import TangentSpace, FGDA

metrics = ["euclid", "logchol", "logeuclid", "riemann", "wasserstein"]


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("tspace", [TangentSpace, FGDA])
def test_tangentspace(kind, tspace, get_mats, get_labels, get_weights):
    if kind == "hpd" and tspace in [FGDA]:
        pytest.skip()
    n_classes, n_matrices, n_channels = 2, 6, 3
    X = get_mats(n_matrices, n_channels, kind)
    y = get_labels(n_matrices, n_classes)
    weights = get_weights(n_matrices)

    ts_fit(tspace, X, y, weights)
    ts_transform(tspace, X, y)
    ts_fit_transform(tspace, X, y)
    ts_fit_independence(tspace, X, y)
    if tspace is TangentSpace:
        ts_transform_wo_fit(tspace, X)
        ts_inversetransform(tspace, X)


def ts_fit(tspace, X, y, weights):
    ts = tspace().fit(X, y)

    if tspace is TangentSpace:
        n_channels = X.shape[-1]
        assert ts.reference_.shape == (n_channels, n_channels)
    elif tspace is FGDA:
        n_classes = len(np.unique(y))
        assert ts.classes_.shape == (n_classes,)

    ts.fit(X, y, sample_weight=weights)


def ts_transform(tspace, X, y):
    n_matrices, n_channels, n_channels = X.shape
    Xtr = tspace().fit(X, y).transform(X)
    if tspace is TangentSpace:
        n_ts = (n_channels * (n_channels + 1)) // 2
        assert Xtr.shape == (n_matrices, n_ts)
    elif tspace is FGDA:
        assert Xtr.shape == (n_matrices, n_channels, n_channels)


def ts_fit_transform(tspace, X, y):
    n_matrices, n_channels, n_channels = X.shape
    ts = tspace()
    Xtr = ts.fit(X, y).transform(X)
    Xtr2 = ts.fit_transform(X, y)
    assert_array_equal(Xtr, Xtr2)


def ts_fit_independence(tspace, X, y):
    n_matrices, n_channels, n_channels = X.shape
    ts = tspace()
    ts.fit(X, y)
    # retraining with different size should erase previous fit
    Xnew = X[:, :-1, :-1]
    ts.fit(Xnew, y)
    # fit_transform should work as well
    ts.fit_transform(Xnew, y)


def ts_transform_wo_fit(tspace, X):
    n_matrices, n_channels, n_channels = X.shape
    n_ts = (n_channels * (n_channels + 1)) // 2
    ts = tspace()
    Xtr = ts.transform(X)
    assert Xtr.shape == (n_matrices, n_ts)


def ts_inversetransform(tspace, X):
    ts = tspace().fit(X)
    assert ts.inverse_transform(ts.transform(X)) == approx(X)


@pytest.mark.parametrize("tspace", [TangentSpace, FGDA])
@pytest.mark.parametrize("metric", [42, "faulty", {"foo": "bar"}])
def test_tangentspaces_metric_wrong_keys(tspace, metric, get_mats, get_labels):
    with pytest.raises((TypeError, KeyError, ValueError)):
        n_classes, n_matrices, n_channels = 2, 6, 3
        X = get_mats(n_matrices, n_channels, "spd")
        y = get_labels(n_matrices, n_classes)
        ts = tspace(metric=metric)
        ts.fit(X, y).transform(X)


@pytest.mark.parametrize("fit", [True, False])
@pytest.mark.parametrize("tsupdate", [True, False])
@pytest.mark.parametrize("metric_mean", ["euclid", "logeuclid", "riemann"])
@pytest.mark.parametrize("metric_map", metrics)
def test_tangentspace_init(fit, tsupdate, metric_mean, metric_map, get_mats):
    n_matrices, n_channels = 4, 3
    n_ts = (n_channels * (n_channels + 1)) // 2
    X = get_mats(n_matrices, n_channels, "spd")

    ts = TangentSpace(
        metric={"mean": metric_mean, "map": metric_map},
        tsupdate=tsupdate,
    )
    if fit:
        ts.fit(X)
    Xtr = ts.transform(X)
    assert Xtr.shape == (n_matrices, n_ts)


def test_tangentspace_vecdim_error():
    n_matrices, n_ts = 4, 6
    ts = TangentSpace()
    with pytest.raises(ValueError):
        tsvectors_wrong = np.empty((n_matrices, n_ts + 1))
        ts.transform(tsvectors_wrong)


def test_tangentspace_matdim_error():
    n_matrices, n_channels = 4, 3
    ts = TangentSpace()
    with pytest.raises(ValueError):
        not_square_mat = np.empty((n_matrices, n_channels, n_channels + 1))
        ts.transform(not_square_mat)
    with pytest.raises(ValueError):
        too_many_dim = np.empty((1, 2, 3, 4))
        ts.transform(too_many_dim)


@pytest.mark.parametrize("tsupdate", [True, False])
@pytest.mark.parametrize("metric_mean", ["euclid", "logeuclid", "riemann"])
@pytest.mark.parametrize("metric_map", metrics)
def test_fgda_init(tsupdate, metric_mean, metric_map, get_mats, get_labels):
    n_classes, n_matrices, n_channels = 2, 6, 3
    X = get_mats(n_matrices, n_channels, "spd")
    y = get_labels(n_matrices, n_classes)

    ts = FGDA(
        metric={"mean": metric_mean, "map": metric_map},
        tsupdate=tsupdate,
    )
    ts.fit(X, y)
    Xtr = ts.transform(X)
    assert Xtr.shape == (n_matrices, n_channels, n_channels)
