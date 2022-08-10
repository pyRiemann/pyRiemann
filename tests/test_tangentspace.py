from conftest import get_metrics
import numpy as np
from pyriemann.tangentspace import TangentSpace, FGDA
import pytest
from pytest import approx


@pytest.mark.parametrize("tspace", [TangentSpace, FGDA])
class TangentSpaceTestCase:
    def test_tangentspace(self, tspace, get_covmats, get_labels):
        n_classes, n_matrices, n_channels = 2, 6, 3
        covmats = get_covmats(n_matrices, n_channels)
        labels = get_labels(n_matrices, n_classes)
        self.clf_transform(tspace, covmats, labels)
        self.clf_fit_transform(tspace, covmats, labels)
        self.clf_fit_transform_independence(tspace, covmats, labels)
        if tspace is TangentSpace:
            self.clf_transform_wo_fit(tspace, covmats)
            self.clf_inversetransform(tspace, covmats)


class TestTangentSpace(TangentSpaceTestCase):
    def clf_transform(self, tspace, covmats, labels):
        n_matrices, n_channels, n_channels = covmats.shape
        ts = tspace().fit(covmats, labels)
        Xtr = ts.transform(covmats)
        if tspace is TangentSpace:
            n_ts = (n_channels * (n_channels + 1)) // 2
            assert Xtr.shape == (n_matrices, n_ts)
        else:
            assert Xtr.shape == (n_matrices, n_channels, n_channels)

    def clf_fit_transform(self, tspace, covmats, labels):
        n_matrices, n_channels, n_channels = covmats.shape
        ts = tspace()
        Xtr = ts.fit_transform(covmats, labels)
        if tspace is TangentSpace:
            n_ts = (n_channels * (n_channels + 1)) // 2
            assert Xtr.shape == (n_matrices, n_ts)
        else:
            assert Xtr.shape == (n_matrices, n_channels, n_channels)

    def clf_fit_transform_independence(self, tspace, covmats, labels):
        n_matrices, n_channels, n_channels = covmats.shape
        ts = tspace()
        ts.fit(covmats, labels)
        # retraining with different size should erase previous fit
        new_covmats = covmats[:, :-1, :-1]
        ts.fit(new_covmats, labels)
        # fit_transform should work as well
        ts.fit_transform(covmats, labels)

    def clf_transform_wo_fit(self, tspace, covmats):
        n_matrices, n_channels, n_channels = covmats.shape
        n_ts = (n_channels * (n_channels + 1)) // 2
        ts = tspace()
        Xtr = ts.transform(covmats)
        assert Xtr.shape == (n_matrices, n_ts)

    def clf_inversetransform(self, tspace, covmats):
        n_matrices, n_channels, n_channels = covmats.shape
        ts = tspace().fit(covmats)
        Xtr = ts.transform(covmats)
        covinv = ts.inverse_transform(Xtr)
        assert covinv == approx(covmats)


@pytest.mark.parametrize("fit", [True, False])
@pytest.mark.parametrize("tsupdate", [True, False])
@pytest.mark.parametrize("metric_mean", get_metrics())
@pytest.mark.parametrize("metric_map", ["euclid", "logeuclid", "riemann"])
def test_TangentSpace_init(fit, tsupdate, metric_mean, metric_map,
                           get_covmats):
    n_matrices, n_channels = 4, 3
    n_ts = (n_channels * (n_channels + 1)) // 2
    covmats = get_covmats(n_matrices, n_channels)
    ts = TangentSpace(
        metric={"mean": metric_mean, "map": metric_map},
        tsupdate=tsupdate
    )
    if fit:
        ts.fit(covmats)
    Xtr = ts.transform(covmats)
    assert Xtr.shape == (n_matrices, n_ts)


@pytest.mark.parametrize("tsupdate", [True, False])
@pytest.mark.parametrize("metric_mean", get_metrics())
@pytest.mark.parametrize("metric_map", ["euclid", "logeuclid", "riemann"])
def test_FGDA_init(tsupdate, metric_mean, metric_map, get_covmats, get_labels):
    n_classes, n_matrices, n_channels = 2, 6, 3
    labels = get_labels(n_matrices, n_classes)
    covmats = get_covmats(n_matrices, n_channels)
    ts = FGDA(
        metric={"mean": metric_mean, "map": metric_map},
        tsupdate=tsupdate
    )
    ts.fit(covmats, labels)
    Xtr = ts.transform(covmats)
    assert Xtr.shape == (n_matrices, n_channels, n_channels)


@pytest.mark.parametrize("tspace", [TangentSpace, FGDA])
@pytest.mark.parametrize("metric", [42, "faulty", {"foo": "bar"}])
def test_metric_wrong_keys(tspace, metric, get_covmats, get_labels):
    with pytest.raises((TypeError, KeyError, ValueError)):
        n_classes, n_matrices, n_channels = 2, 6, 3
        labels = get_labels(n_matrices, n_classes)
        covmats = get_covmats(n_matrices, n_channels)
        clf = tspace(metric=metric)
        clf.fit(covmats, labels).transform(covmats)


def test_TS_vecdim_error(get_covmats, rndstate):
    n_matrices, n_ts = 4, 6
    ts = TangentSpace()
    with pytest.raises(ValueError):
        tsvectors_wrong = np.empty((n_matrices, n_ts + 1))
        ts.transform(tsvectors_wrong)


def test_TS_matdim_error(get_covmats):
    n_matrices, n_channels = 4, 3
    ts = TangentSpace()
    with pytest.raises(ValueError):
        not_square_mat = np.empty((n_matrices, n_channels, n_channels + 1))
        ts.transform(not_square_mat)
    with pytest.raises(ValueError):
        too_many_dim = np.empty((1, 2, 3, 4))
        ts.transform(too_many_dim)
