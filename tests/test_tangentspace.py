from conftest import get_covmats, get_metrics
import numpy as np
from numpy.testing import assert_array_almost_equal
from pyriemann.tangentspace import TangentSpace, FGDA
import pytest
from pytest import approx


@pytest.mark.parametrize("tspace", [TangentSpace, FGDA])
class TangentSpaceTestCase:
    def test_tangentspace(self, tspace, get_covmats):
        n_classes, n_trials, n_channels = 2, 6, 3
        covmats = get_covmats(n_trials, n_channels)
        labels = np.array([0, 1]).repeat(n_trials // n_classes)
        self.clf_transform(tspace, covmats, labels)
        self.clf_fit_transform(tspace, covmats, labels)
        self.clf_fit_transform_independence(tspace, covmats, labels)
        if tspace is TangentSpace:
            self.clf_transform_wo_fit(tspace, covmats)
            self.clf_inversetransform(tspace, covmats)


class TestTangentSpace(TangentSpaceTestCase):
    def clf_transform(self, tspace, covmats, labels):
        n_trials, n_channels, n_channels = covmats.shape
        ts = tspace().fit(covmats, labels)
        Xtr = ts.transform(covmats)
        if tspace is TangentSpace:
            n_ts = (n_channels * (n_channels + 1)) // 2
            assert Xtr.shape == (n_trials, n_ts)
        else:
            assert Xtr.shape == (n_trials, n_channels, n_channels)

    def clf_fit_transform(self, tspace, covmats, labels):
        n_trials, n_channels, n_channels = covmats.shape
        ts = tspace()
        Xtr = ts.fit_transform(covmats, labels)
        if tspace is TangentSpace:
            n_ts = (n_channels * (n_channels + 1)) // 2
            assert Xtr.shape == (n_trials, n_ts)
        else:
            assert Xtr.shape == (n_trials, n_channels, n_channels)

    def clf_fit_transform_independence(self, tspace, covmats, labels):
        n_trials, n_channels, n_channels = covmats.shape
        ts = tspace()
        ts.fit(covmats, labels)
        # retraining with different size should erase previous fit
        new_covmats = covmats[:, :-1, :-1]
        ts.fit(new_covmats, labels)
        # fit_transform should work as well
        ts.fit_transform(covmats, labels)

    def clf_transform_wo_fit(self, tspace, covmats):
        n_trials, n_channels, n_channels = covmats.shape
        n_ts = (n_channels * (n_channels + 1)) // 2
        ts = tspace()
        Xtr = ts.transform(covmats)
        assert Xtr.shape == (n_trials, n_ts)

    def clf_inversetransform(self, tspace, covmats):
        n_trials, n_channels, n_channels = covmats.shape
        ts = tspace().fit(covmats)
        Xtr = ts.transform(covmats)
        covinv = ts.inverse_transform(Xtr)
        assert covinv == approx(covmats)


@pytest.mark.parametrize("fit", [True, False])
@pytest.mark.parametrize("tsupdate", [True, False])
@pytest.mark.parametrize("metric", get_metrics())
def test_TangentSpace_init(fit, tsupdate, metric, get_covmats):
    n_trials, n_channels = 4, 3
    n_ts = (n_channels * (n_channels + 1)) // 2
    covmats = get_covmats(n_trials, n_channels)
    ts = TangentSpace(metric=metric, tsupdate=tsupdate)
    if fit:
        ts.fit(covmats)
    Xtr = ts.transform(covmats)
    assert Xtr.shape == (n_trials, n_ts)


@pytest.mark.parametrize("tsupdate", [True, False])
@pytest.mark.parametrize("metric", get_metrics())
def test_FGDA_init(tsupdate, metric, get_covmats):
    n_classes, n_trials, n_channels = 2, 6, 3
    labels = np.array([0, 1]).repeat(n_trials // n_classes)
    covmats = get_covmats(n_trials, n_channels)
    ts = FGDA(metric=metric, tsupdate=tsupdate)
    ts.fit(covmats, labels)
    Xtr = ts.transform(covmats)
    assert Xtr.shape == (n_trials, n_channels, n_channels)
