import numpy as np
import pytest
from pytest import approx

from conftest import get_metrics
from pyriemann.tangentspace import TangentSpace, FGDA


@pytest.mark.parametrize("tspace", [TangentSpace, FGDA])
def test_tangentspace(tspace, get_mats, get_labels, get_weights):
    n_classes, n_matrices, n_channels = 2, 6, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)
    weights = get_weights(n_matrices)

    clf_fit(tspace, mats, labels, weights)
    clf_transform(tspace, mats, labels)
    clf_fit_transform(tspace, mats, labels)
    clf_fit_transform_independence(tspace, mats, labels)
    if tspace is TangentSpace:
        clf_transform_wo_fit(tspace, mats)
        clf_inversetransform(tspace, mats)


def clf_fit(tspace, mats, labels, weights):
    clf = tspace().fit(mats, labels)

    if tspace is TangentSpace:
        n_channels = mats.shape[-1]
        assert clf.reference_.shape == (n_channels, n_channels)
    elif tspace is FGDA:
        n_classes = len(np.unique(labels))
        assert clf.classes_.shape == (n_classes,)

    clf.fit(mats, labels, sample_weight=weights)


def clf_transform(tspace, mats, labels):
    n_matrices, n_channels, n_channels = mats.shape
    ts = tspace().fit(mats, labels)
    Xtr = ts.transform(mats)
    if tspace is TangentSpace:
        n_ts = (n_channels * (n_channels + 1)) // 2
        assert Xtr.shape == (n_matrices, n_ts)
    else:
        assert Xtr.shape == (n_matrices, n_channels, n_channels)


def clf_fit_transform(tspace, mats, labels):
    n_matrices, n_channels, n_channels = mats.shape
    ts = tspace()
    Xtr = ts.fit_transform(mats, labels)
    if tspace is TangentSpace:
        n_ts = (n_channels * (n_channels + 1)) // 2
        assert Xtr.shape == (n_matrices, n_ts)
    else:
        assert Xtr.shape == (n_matrices, n_channels, n_channels)


def clf_fit_transform_independence(tspace, mats, labels):
    n_matrices, n_channels, n_channels = mats.shape
    ts = tspace()
    ts.fit(mats, labels)
    # retraining with different size should erase previous fit
    new_mats = mats[:, :-1, :-1]
    ts.fit(new_mats, labels)
    # fit_transform should work as well
    ts.fit_transform(mats, labels)


def clf_transform_wo_fit(tspace, mats):
    n_matrices, n_channels, n_channels = mats.shape
    n_ts = (n_channels * (n_channels + 1)) // 2
    ts = tspace()
    Xtr = ts.transform(mats)
    assert Xtr.shape == (n_matrices, n_ts)


def clf_inversetransform(tspace, mats):
    n_matrices, n_channels, n_channels = mats.shape
    ts = tspace().fit(mats)
    Xtr = ts.transform(mats)
    invmats = ts.inverse_transform(Xtr)
    assert invmats == approx(mats)


@pytest.mark.parametrize("fit", [True, False])
@pytest.mark.parametrize("tsupdate", [True, False])
@pytest.mark.parametrize("metric_mean", get_metrics())
@pytest.mark.parametrize("metric_map", [
    "euclid", "logchol", "logeuclid", "riemann", "wasserstein"
])
def test_TangentSpace_init(fit, tsupdate, metric_mean, metric_map, get_mats):
    n_matrices, n_channels = 4, 3
    n_ts = (n_channels * (n_channels + 1)) // 2
    mats = get_mats(n_matrices, n_channels, "spd")

    ts = TangentSpace(
        metric={"mean": metric_mean, "map": metric_map},
        tsupdate=tsupdate,
    )
    if fit:
        ts.fit(mats)
    Xtr = ts.transform(mats)
    assert Xtr.shape == (n_matrices, n_ts)


@pytest.mark.parametrize("tsupdate", [True, False])
@pytest.mark.parametrize("metric_mean", get_metrics())
@pytest.mark.parametrize("metric_map", [
    "euclid", "logchol", "logeuclid", "riemann", "wasserstein"
])
def test_FGDA_init(tsupdate, metric_mean, metric_map, get_mats, get_labels):
    n_classes, n_matrices, n_channels = 2, 6, 3
    labels = get_labels(n_matrices, n_classes)
    mats = get_mats(n_matrices, n_channels, "spd")

    ts = FGDA(
        metric={"mean": metric_mean, "map": metric_map},
        tsupdate=tsupdate,
    )
    ts.fit(mats, labels)
    Xtr = ts.transform(mats)
    assert Xtr.shape == (n_matrices, n_channels, n_channels)


@pytest.mark.parametrize("tspace", [TangentSpace, FGDA])
@pytest.mark.parametrize("metric", [42, "faulty", {"foo": "bar"}])
def test_metric_wrong_keys(tspace, metric, get_mats, get_labels):
    with pytest.raises((TypeError, KeyError, ValueError)):
        n_classes, n_matrices, n_channels = 2, 6, 3
        labels = get_labels(n_matrices, n_classes)
        mats = get_mats(n_matrices, n_channels, "spd")
        clf = tspace(metric=metric)
        clf.fit(mats, labels).transform(mats)


def test_TS_vecdim_error():
    n_matrices, n_ts = 4, 6
    ts = TangentSpace()
    with pytest.raises(ValueError):
        tsvectors_wrong = np.empty((n_matrices, n_ts + 1))
        ts.transform(tsvectors_wrong)


def test_TS_matdim_error():
    n_matrices, n_channels = 4, 3
    ts = TangentSpace()
    with pytest.raises(ValueError):
        not_square_mat = np.empty((n_matrices, n_channels, n_channels + 1))
        ts.transform(not_square_mat)
    with pytest.raises(ValueError):
        too_many_dim = np.empty((1, 2, 3, 4))
        ts.transform(too_many_dim)
