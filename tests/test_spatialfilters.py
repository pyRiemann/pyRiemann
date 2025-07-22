import numpy as np
from numpy.testing import assert_array_equal
import pytest

from conftest import get_metrics
from pyriemann.spatialfilters import Xdawn, CSP, SPoC, BilinearFilter, AJDC


@pytest.mark.parametrize("spfilt", [Xdawn, CSP, SPoC, BilinearFilter, AJDC])
@pytest.mark.parametrize("n_channels", [3, 5, 7])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_spatial_filters(spfilt, n_channels, n_classes,
                         get_mats, rndstate, get_labels):
    if n_classes == 2:
        n_matrices, n_times = 10, 256
    else:
        n_matrices, n_times = 9, 256
    labels = get_labels(n_matrices, n_classes)
    if spfilt is Xdawn:
        X = rndstate.randn(n_matrices, n_channels, n_times)
    elif spfilt in (CSP, SPoC, BilinearFilter):
        X = get_mats(n_matrices, n_channels, "spd")
    elif spfilt is AJDC:
        n_subjects, n_conditions = 2, 2
        X = rndstate.randn(n_subjects, n_conditions, n_channels, n_times)

    clf_fit(spfilt, X, labels, n_channels, n_times)
    clf_fit_independence(spfilt, X, labels, n_channels)
    if spfilt is CSP:
        clf_fit_error(spfilt, X, labels)
    clf_transform(spfilt, X, labels, n_matrices, n_channels, n_times)
    if spfilt in (CSP, SPoC, BilinearFilter):
        clf_transform_error(spfilt, X, labels, n_channels)


def clf_fit(spfilt, X, labels, n_channels, n_times):
    n_classes = len(np.unique(labels))
    if spfilt is BilinearFilter:
        n_filters = 4
        sf = spfilt(filters=np.eye(n_filters, n_channels))
    elif spfilt is AJDC:
        sf = spfilt(dim_red={"n_components": n_channels - 1})
    else:
        sf = spfilt()

    sf.fit(X, labels)

    if spfilt is AJDC:
        assert sf.forward_filters_.shape == (sf.n_sources_, n_channels)
    elif spfilt is BilinearFilter:
        assert sf.filters_.shape == (n_filters, n_channels)
    elif spfilt is Xdawn:
        n_components = min(n_channels, sf.nfilter)
        assert len(sf.classes_) == n_classes
        assert sf.filters_.shape == (n_classes * n_components, n_channels)
        assert sf.patterns_.shape == (n_classes * n_components, n_channels)
        assert sf.evokeds_.shape == (n_classes * n_components, n_times)
    elif spfilt in [CSP, SPoC]:
        n_components = min(n_channels, sf.nfilter)
        assert sf.filters_.shape == (n_components, n_channels)
        assert sf.patterns_.shape == (n_components, n_channels)


def clf_fit_error(spfilt, X, labels):
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


def clf_transform(spfilt, X, labels, n_matrices, n_channels, n_times):
    n_classes = len(np.unique(labels))
    if spfilt is BilinearFilter:
        n_filters = 4
        sf = spfilt(filters=np.eye(n_filters, n_channels))
    elif spfilt is AJDC:
        sf = spfilt(dim_red={"expl_var": 0.9})
    else:
        sf = spfilt()
    if spfilt is AJDC:
        sf.fit(X, labels)
        X_new = np.squeeze(X[0])
        n_matrices = X_new.shape[0]
        Xtr = sf.transform(X_new)
    else:
        Xtr = sf.fit(X, labels).transform(X)

        Xtr1 = sf.fit_transform(X, labels)

        assert_array_equal(Xtr, Xtr1)

    if spfilt is AJDC:
        assert Xtr.shape == (n_matrices, n_channels, n_times)
    elif spfilt is BilinearFilter:
        assert Xtr.shape == (n_matrices, n_filters, n_filters)
    elif spfilt is Xdawn:
        n_components = min(n_channels, sf.nfilter)
        assert Xtr.shape == (n_matrices, n_classes * n_components, n_times)
    else:
        n_components = min(n_channels, sf.nfilter)
        assert Xtr.shape == (n_matrices, n_components)


def clf_transform_error(spfilt, X, labels, n_channels):
    if spfilt is BilinearFilter:
        sf = spfilt(np.eye(n_channels))
    elif spfilt is AJDC:
        sf = spfilt(dim_red={"max_cond": 10})
    else:
        sf = spfilt()
    with pytest.raises(ValueError):
        sf.fit(X, labels).transform(X[:, :-1, :-1])


def clf_fit_independence(spfilt, X, labels, n_channels):
    if spfilt is BilinearFilter:
        sf = spfilt(np.eye(n_channels))
    elif spfilt is AJDC:
        sf = spfilt(dim_red={"max_cond": 10})
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


@pytest.mark.parametrize("n_channels", [3, 4, 5])
@pytest.mark.parametrize("use_baseline_cov", [True, False])
def test_xdawn_baselinecov(n_channels, use_baseline_cov, rndstate, get_labels):
    n_classes, n_matrices, n_times = 2, 6, 100
    x = rndstate.randn(n_matrices, n_channels, n_times)
    labels = get_labels(n_matrices, n_classes)
    if use_baseline_cov:
        baseline_cov = np.identity(n_channels)
    else:
        baseline_cov = None
    xd = Xdawn(baseline_cov=baseline_cov)
    xd.fit(x, labels).transform(x)
    n_components = min(n_channels, xd.nfilter)
    assert xd.filters_.shape == (n_classes * n_components, n_channels)


@pytest.mark.parametrize("n_filters", [3, 4, 5])
@pytest.mark.parametrize("metric", get_metrics())
@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("ajd_method", ["ajd_pham", "rjd", "uwedge"])
def test_csp(n_filters, metric, log, ajd_method, get_mats, get_labels):
    n_classes, n_matrices, n_channels = 2, 6, 4
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)

    n_components = min(n_channels, n_filters)

    csp = CSP(nfilter=n_filters, metric=metric, log=log, ajd_method=ajd_method)
    csp.fit(mats, labels)
    assert csp.filters_.shape == (n_components, n_channels)
    assert csp.patterns_.shape == (n_components, n_channels)
    Xtr = csp.transform(mats)
    if log:
        assert Xtr.shape == (n_matrices, n_components)
    else:
        assert Xtr.shape == (n_matrices, n_components, n_components)


def test_bilinearfilter_errors(get_mats, get_labels):
    n_classes, n_matrices, n_channels = 2, 6, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)

    with pytest.raises(TypeError):
        BilinearFilter("foo").fit(mats, labels)
    with pytest.raises(TypeError):
        BilinearFilter(np.eye(3), log="foo").fit(mats, labels)


@pytest.mark.parametrize("n_filters", [3, 4])
@pytest.mark.parametrize("log", [True, False])
def test_bilinearfilter(n_filters, log, get_mats, get_labels):
    n_classes, n_matrices, n_channels = 2, 6, 4
    mats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)

    bf = BilinearFilter(np.eye(n_filters, n_channels), log=log)
    bf.fit(mats, labels)
    assert bf.filters_.shape == (n_filters, n_channels)
    Xtr = bf.transform(mats)
    if log:
        assert Xtr.shape == (n_matrices, n_filters)
    else:
        assert Xtr.shape == (n_matrices, n_filters, n_filters)


def test_ajdc_init():
    ajdc = AJDC(fmin=1, fmax=32, fs=64)
    assert ajdc.window == 128
    assert ajdc.overlap == 0.5
    assert ajdc.dim_red is None
    assert ajdc.verbose


def test_ajdc_fit(rndstate):
    n_subjects, n_conditions, n_channels, n_times = 5, 3, 8, 512
    X = rndstate.randn(n_subjects, n_conditions, n_channels, n_times)
    ajdc = AJDC(dim_red={"n_components": n_channels - 1}).fit(X)
    assert ajdc.forward_filters_.shape == (ajdc.n_sources_, n_channels)
    assert ajdc.backward_filters_.shape == (n_channels, ajdc.n_sources_)
    with pytest.warns(UserWarning):  # dim_red is None
        AJDC().fit(X)


def test_ajdc_fit_error(rndstate):
    n_subjects, n_conditions, n_channels, n_times = 2, 3, 8, 512
    ajdc = AJDC(dim_red={"expl_var": 0.9})
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
    X = rndstate.randn(n_subjects, n_conditions, n_channels, n_times)
    V = rndstate.randn(n_channels, n_channels - 1)
    ajdc = AJDC(dim_red={'warm_restart': V})
    with pytest.raises(ValueError):  # initial diag not square
        ajdc.fit(X)


def test_ajdc_transform_error(rndstate):
    n_subjects, n_conditions, n_channels, n_times = 2, 2, 4, 256
    X = rndstate.randn(n_subjects, n_conditions, n_channels, n_times)
    ajdc = AJDC(dim_red={"warm_restart": np.eye(n_channels - 1)}).fit(X)
    n_matrices = 4
    X_new = rndstate.randn(n_matrices, n_channels, n_times)
    with pytest.raises(ValueError):  # not 3 dims
        ajdc.transform(X_new[0])
    with pytest.raises(ValueError):  # unequal # of chans
        ajdc.transform(rndstate.randn(n_matrices, n_channels + 1, 1))


def test_ajdc_fit_variable_input(rndstate):
    n_subjects, n_cond, n_chan, n_times = 2, 2, 3, 256
    X = rndstate.randn(n_subjects, n_cond, n_chan, n_times)
    ajdc = AJDC(dim_red={"expl_var": 0.9})
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
    n_subjects, n_conditions, n_channels, n_times = 2, 2, 5, 256
    X = rndstate.randn(n_subjects, n_conditions, n_channels, n_times)
    ajdc = AJDC(dim_red={"warm_restart": np.eye(n_channels - 1)}).fit(X)
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


def test_ajdc_get_src_expl_var(rndstate):
    n_subjects, n_conditions, n_channels, n_times = 2, 2, 3, 256
    X = rndstate.randn(n_subjects, n_conditions, n_channels, n_times)
    ajdc = AJDC(dim_red={"expl_var": 0.9}).fit(X)
    n_matrices = 4
    X_new = rndstate.randn(n_matrices, n_channels, n_times)
    v = ajdc.get_src_expl_var(X_new)
    assert v.shape == (n_matrices, ajdc.n_sources_)
    with pytest.raises(ValueError):  # not 3 dims
        ajdc.get_src_expl_var(X_new[0])
    with pytest.raises(ValueError):  # unequal # of chans
        ajdc.get_src_expl_var(rndstate.randn(n_matrices, n_channels + 1, 1))
