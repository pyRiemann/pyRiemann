from conftest import get_metrics
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from pyriemann.spatialfilters import Xdawn, CSP, SPoC, BilinearFilter, AJDC


@pytest.mark.parametrize("spfilt", [Xdawn, CSP, SPoC, BilinearFilter, AJDC])
class SpatialFiltersTestCase:
    def test_two_classes(self, spfilt, get_covmats, rndstate, get_labels):
        n_classes = 2
        n_matrices, n_channels, n_times = 8, 3, 512
        labels = get_labels(n_matrices, n_classes)
        if spfilt is Xdawn:
            X = rndstate.randn(n_matrices, n_channels, n_times)
        elif spfilt in (CSP, SPoC, BilinearFilter):
            X = get_covmats(n_matrices, n_channels)
        elif spfilt is AJDC:
            n_subjects, n_conditions = 2, 2
            X = rndstate.randn(n_subjects, n_conditions, n_channels, n_times)

        self.clf_fit(spfilt, X, labels, n_channels, n_times)
        self.clf_fit_independence(spfilt, X, labels, n_channels, n_times)
        if spfilt is CSP:
            self.clf_fit_error(spfilt, X, labels)
        self.clf_transform(spfilt, X, labels, n_matrices, n_channels, n_times)
        if spfilt in (CSP, SPoC, BilinearFilter):
            self.clf_transform_error(spfilt, X, labels, n_channels)

    def test_three_classes(self, spfilt, get_covmats, rndstate, get_labels):
        n_classes = 3
        n_matrices, n_channels, n_times = 6, 3, 512
        labels = get_labels(n_matrices, n_classes)
        if spfilt is Xdawn:
            X = rndstate.randn(n_matrices, n_channels, n_times)
        elif spfilt in (CSP, SPoC, BilinearFilter):
            X = get_covmats(n_matrices, n_channels)
        elif spfilt is AJDC:
            n_subjects, n_conditions = 2, 2
            X = rndstate.randn(n_subjects, n_conditions, n_channels, n_times)

        self.clf_fit(spfilt, X, labels, n_channels, n_times)
        self.clf_fit_independence(spfilt, X, labels, n_channels, n_times)
        if spfilt is CSP:
            self.clf_fit_error(spfilt, X, labels)
        self.clf_transform(spfilt, X, labels, n_matrices, n_channels, n_times)
        if spfilt in (CSP, SPoC, BilinearFilter):
            self.clf_transform_error(spfilt, X, labels, n_channels)


class TestSpatialFilters(SpatialFiltersTestCase):
    def clf_fit(self, spfilt, X, labels, n_channels, n_times):
        n_classes = len(np.unique(labels))
        if spfilt is BilinearFilter:
            filters = np.eye(n_channels)
            sf = spfilt(filters)
        else:
            sf = spfilt()
        sf.fit(X, labels)
        if spfilt is Xdawn:
            assert len(sf.classes_) == n_classes
            assert sf.filters_.shape == (n_classes * n_channels, n_channels)
            for sfilt in sf.filters_:
                assert sfilt.shape == (n_channels,)
        elif spfilt in [CSP, SPoC]:
            assert sf.filters_.shape == (n_channels, n_channels)
        elif spfilt is AJDC:
            assert sf.forward_filters_.shape == (sf.n_sources_, n_channels)

    def clf_fit_error(self, spfilt, X, labels):
        sf = spfilt()
        with pytest.raises(ValueError):
            sf.fit(X, labels * 0.0)  # 1 class
        with pytest.raises(ValueError):
            sf.fit(X, labels[:1])  # unequal # of samples
        with pytest.raises(TypeError):
            sf.fit(X, "foo")  # y must be an array
        with pytest.raises(TypeError):
            sf.fit("foo", labels)  # X must be an array
        with pytest.raises(ValueError):
            sf.fit(X[:, 0], labels)
        with pytest.raises(ValueError):
            sf.fit(X, X)

    def clf_transform(self, spfilt, X, labels,
                      n_matrices, n_channels, n_times):
        n_classes = len(np.unique(labels))
        if spfilt is BilinearFilter:
            filters = np.eye(n_channels)
            sf = spfilt(filters)
        else:
            sf = spfilt()
        if spfilt is AJDC:
            sf.fit(X, labels)
            X_new = np.squeeze(X[0])
            n_matrices = X_new.shape[0]
            Xtr = sf.transform(X_new)
        else:
            Xtr = sf.fit(X, labels).transform(X)
        if spfilt is Xdawn:
            n_comp = n_classes * n_channels
            assert Xtr.shape == (n_matrices, n_comp, n_times)
        elif spfilt is BilinearFilter:
            assert Xtr.shape == (n_matrices, n_channels, n_channels)
        elif spfilt is AJDC:
            assert Xtr.shape == (n_matrices, n_channels, n_times)
        else:
            assert Xtr.shape == (n_matrices, n_channels)

    def clf_transform_error(self, spfilt, X, labels, n_channels):
        if spfilt is BilinearFilter:
            filters = np.eye(n_channels)
            sf = spfilt(filters)
        else:
            sf = spfilt()
        with pytest.raises(ValueError):
            sf.fit(X, labels).transform(X[:, :-1, :-1])

    def clf_fit_independence(self, spfilt, X, labels, n_channels, n_times):
        if spfilt is BilinearFilter:
            filters = np.eye(n_channels)
            sf = spfilt(filters)
        else:
            sf = spfilt()
        sf.fit(X, labels)
        if spfilt is Xdawn:
            X_new = X[:, :-1, :]
        elif spfilt in (CSP, SPoC, BilinearFilter):
            X_new = X[:, :-1, :-1]
        elif spfilt is AJDC:
            X_new = X[:, :, :-1, :]
        # retraining with different size should erase previous fit
        sf.fit(X_new, labels)


def test_xdawn_baselinecov(rndstate, get_labels):
    """Test cov precomputation"""
    n_matrices, n_channels, n_times = 6, 5, 100
    n_classes, default_nfilter = 2, 4
    x = rndstate.randn(n_matrices, n_channels, n_times)
    labels = get_labels(n_matrices, n_classes)
    baseline_cov = np.identity(n_channels)
    xd = Xdawn(baseline_cov=baseline_cov)
    xd.fit(x, labels).transform(x)
    assert len(xd.filters_) == n_classes * default_nfilter
    for sfilt in xd.filters_:
        assert sfilt.shape == (n_channels,)


@pytest.mark.parametrize("nfilter", [3, 4])
@pytest.mark.parametrize("metric", get_metrics())
@pytest.mark.parametrize("log", [True, False])
def test_csp_init(nfilter, metric, log, get_covmats, get_labels):
    n_classes, n_matrices, n_channels = 2, 6, 3
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    csp = CSP(nfilter=nfilter, metric=metric, log=log)
    csp.fit(covmats, labels)
    Xtr = csp.transform(covmats)
    if log:
        assert Xtr.shape == (n_matrices, n_channels)
    else:
        assert Xtr.shape == (n_matrices, n_channels, n_channels)
    assert csp.filters_.shape == (n_channels, n_channels)
    assert csp.patterns_.shape == (n_channels, n_channels)


def test_bilinearfilter_filter_error(get_covmats, get_labels):
    n_classes, n_matrices, n_channels = 2, 6, 3
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    with pytest.raises(TypeError):
        BilinearFilter("foo").fit(covmats, labels)


def test_bilinearfilter_log_error(get_covmats, get_labels):
    n_classes, n_matrices, n_channels = 2, 6, 3
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    with pytest.raises(TypeError):
        BilinearFilter(np.eye(3), log="foo").fit(covmats, labels)


@pytest.mark.parametrize("log", [True, False])
def test_bilinearfilter_log(log, get_covmats, get_labels):
    n_classes, n_matrices, n_channels = 2, 6, 3
    covmats = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    bf = BilinearFilter(np.eye(n_channels), log=log)
    Xtr = bf.fit(covmats, labels).transform(covmats)
    if log:
        assert Xtr.shape == (n_matrices, n_channels)
    else:
        assert Xtr.shape == (n_matrices, n_channels, n_channels)


def test_ajdc_init():
    ajdc = AJDC(fmin=1, fmax=32, fs=64)
    assert ajdc.window == 128
    assert ajdc.overlap == 0.5
    assert ajdc.dim_red is None
    assert ajdc.verbose


def test_ajdc_fit(rndstate):
    n_subjects, n_conditions, n_channels, n_times = 5, 3, 8, 512
    X = rndstate.randn(n_subjects, n_conditions, n_channels, n_times)
    ajdc = AJDC().fit(X)
    assert ajdc.forward_filters_.shape == (ajdc.n_sources_, n_channels)
    assert ajdc.backward_filters_.shape == (n_channels, ajdc.n_sources_)


def test_ajdc_fit_error(rndstate):
    n_conditions, n_channels, n_times = 3, 8, 512
    ajdc = AJDC()
    with pytest.raises(ValueError):  # unequal # of conditions
        ajdc.fit(
            [
                rndstate.randn(n_conditions, n_channels, n_times),
                rndstate.randn(n_conditions + 1, n_channels, n_times),
            ]
        )
    with pytest.raises(ValueError):  # unequal # of channels
        ajdc.fit(
            [
                rndstate.randn(n_conditions, n_channels, n_times),
                rndstate.randn(n_conditions, n_channels + 1, n_times),
            ]
        )


def test_ajdc_transform_error(rndstate):
    n_subjects, n_conditions, n_channels, n_times = 2, 2, 3, 256
    X = rndstate.randn(n_subjects, n_conditions, n_channels, n_times)
    ajdc = AJDC().fit(X)
    n_matrices = 4
    X_new = rndstate.randn(n_matrices, n_channels, n_times)
    with pytest.raises(ValueError):  # not 3 dims
        ajdc.transform(X_new[0])
    with pytest.raises(ValueError):  # unequal # of chans
        ajdc.transform(rndstate.randn(n_matrices, n_channels + 1, 1))


def test_ajdc_fit_variable_input(rndstate):
    n_subjects, n_cond, n_chan, n_times = 2, 2, 3, 256
    X = rndstate.randn(n_subjects, n_cond, n_chan, n_times)
    ajdc = AJDC()
    # 3 subjects, same # conditions and channels, different # of times
    X = [
        rndstate.randn(n_cond, n_chan, n_times + rndstate.randint(500))
        for _ in range(3)
    ]
    ajdc.fit(X)

    # 2 subjects, 2 conditions, same # channels, different # of times
    X = [
        [rndstate.randn(n_chan, n_times + rndstate.randint(500))
         for _ in range(2)],
        [rndstate.randn(n_chan, n_times + rndstate.randint(500))
         for _ in range(2)],
    ]
    ajdc.fit(X)


def test_ajdc_inverse_transform(rndstate):
    n_subjects, n_conditions, n_channels, n_times = 2, 2, 3, 256
    X = rndstate.randn(n_subjects, n_conditions, n_channels, n_times)
    ajdc = AJDC().fit(X)
    n_matrices = 4
    X_new = rndstate.randn(n_matrices, n_channels, n_times)
    Xt = ajdc.transform(X_new)
    Xit = ajdc.inverse_transform(Xt)
    assert_array_equal(Xit.shape, [n_matrices, n_channels, n_times])
    with pytest.raises(ValueError):  # not 3 dims
        ajdc.inverse_transform(Xt[0])
    with pytest.raises(ValueError):  # unequal # of sources
        shape = (n_matrices, ajdc.n_sources_ + 1, 1)
        ajdc.inverse_transform(rndstate.randn(*shape))

    Xit = ajdc.inverse_transform(Xt, supp=[ajdc.n_sources_ - 1])
    assert Xit.shape == (n_matrices, n_channels, n_times)
    with pytest.raises(ValueError):  # not a list
        ajdc.inverse_transform(Xt, supp=1)


def test_ajdc_expl_var(rndstate):
    # Test get_src_expl_var
    n_subjects, n_conditions, n_channels, n_times = 2, 2, 3, 256
    X = rndstate.randn(n_subjects, n_conditions, n_channels, n_times)
    ajdc = AJDC().fit(X)
    n_matrices = 4
    X_new = rndstate.randn(n_matrices, n_channels, n_times)
    v = ajdc.get_src_expl_var(X_new)
    assert v.shape == (n_matrices, ajdc.n_sources_)
    with pytest.raises(ValueError):  # not 3 dims
        ajdc.get_src_expl_var(X_new[0])
    with pytest.raises(ValueError):  # unequal # of chans
        ajdc.get_src_expl_var(rndstate.randn(n_matrices, n_channels + 1, 1))
