import numpy as np
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_raises,
)
from sklearn.covariance import shrunk_covariance
import pytest

from pyriemann.estimation import (
    Covariances,
    ERPCovariances,
    XdawnCovariances,
    CrossSpectra,
    CoSpectra,
    TimeDelayCovariances,
    Coherences,
    Shrinkage,
    BlockCovariances,
    Kernels,
)
from pyriemann.utils.test import (
    is_sym_pos_def as is_spd,
    is_sym_pos_semi_def as is_spsd,
    is_herm_pos_def as is_hpd,
    is_herm_pos_semi_def as is_hpsd,
    is_hankel,
)

estim = ["corr", "cov", "lwf", "mcd", "oas", "sch", "scm"]
m_estim = ["hub", "stu", "tyl"]
coh = ["ordinary", "instantaneous", "lagged", "imaginary"]


@pytest.mark.parametrize("estimator", estim + m_estim)
def test_covariances(estimator, get_mats):
    """Test Covariances"""
    n_matrices, n_channels, n_times = 2, 3, 100
    X = get_mats(n_matrices, [n_channels, n_times], "real")

    covest = Covariances(estimator=estimator).fit(X)
    Xt = covest.transform(X)
    assert covest.get_params() == dict(estimator=estimator)
    assert Xt.shape == (n_matrices, n_channels, n_channels)
    assert is_spd(Xt)

    if estimator == "mcd":
        pytest.skip()
    Xt2 = covest.fit_transform(X)
    assert_array_almost_equal(Xt, Xt2)


@pytest.mark.parametrize(
    "estimator, kwds",
    [
        ("cov", {"bias": True}),
        ("hub", {"tol": 10e-2, "q": 0.95}),
        ("lwf", {"assume_centered": True}),
        ("mcd", {"support_fraction": 0.78}),
        ("oas", {"assume_centered": True}),
        ("scm", {"assume_centered": True}),
        ("stu", {"nu": 2}),
        ("tyl", {"n_iter_max": 20, "norm": "determinant"}),
    ],
)
def test_covariances_kwds(estimator, kwds, get_mats):
    n_matrices, n_channels, n_times = 3, 3, 100
    X = get_mats(n_matrices, [n_channels, n_times], "real") + 1

    Xt_none = Covariances(estimator=estimator).fit_transform(X)
    Xt_kwds = Covariances(estimator=estimator, **kwds).fit_transform(X)
    assert_raises(AssertionError, assert_array_equal, Xt_none, Xt_kwds)


def test_time_delay_covariances_xtd():
    X = np.array(
        [
            [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]],
            [[-1, -2, -3, -4, -5], [-11, -12, -13, -14, -15]],
        ]
    )

    covest = TimeDelayCovariances(delays=[1])
    covest.fit_transform(X)
    Xtd = np.array(
        [
            [
                [1, 2, 3, 4, 5],
                [11, 12, 13, 14, 15],
                [5, 1, 2, 3, 4],
                [15, 11, 12, 13, 14],
            ],
            [
                [-1, -2, -3, -4, -5],
                [-11, -12, -13, -14, -15],
                [-5, -1, -2, -3, -4],
                [-15, -11, -12, -13, -14],
            ],
        ]
    )
    assert_array_equal(covest.Xtd_, Xtd)


@pytest.mark.parametrize("delays", [4, [1, 5]])
def test_time_delay_covariances(delays, get_mats):
    n_matrices, n_channels, n_times = 2, 3, 100
    X = get_mats(n_matrices, [n_channels, n_times], "real")

    covest = TimeDelayCovariances(delays=delays)
    Xt = covest.fit(X).transform(X)
    assert covest.get_params() == dict(estimator="scm", delays=delays)
    if isinstance(delays, int):
        n_delays = delays
    elif isinstance(delays, list):
        n_delays = 1 + len(delays)
    assert Xt.shape == (n_matrices, n_delays * n_channels, n_delays * n_channels)
    assert covest.Xtd_.shape == (n_matrices, n_delays * n_channels, n_times)
    assert is_spd(Xt)
    assert not is_hankel(Xt[0])

    Xt2 = covest.fit_transform(X)
    assert_array_almost_equal(Xt, Xt2)


@pytest.mark.parametrize("estimator", estim)
@pytest.mark.parametrize("svd", [None, 2, 3, 4])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
def test_erp_covariances(estimator, svd, n_classes, get_mats, get_labels):
    """Test fit ERPCovariances"""
    n_matrices, n_channels, n_times = 3 * n_classes, 3, 100
    X = get_mats(n_matrices, [n_channels, n_times], "real")
    y = get_labels(n_matrices, n_classes)

    covest = ERPCovariances(estimator=estimator, svd=svd)
    covmats = covest.fit(X, y).transform(X)
    assert covest.get_params() == dict(classes=None, estimator=estimator, svd=svd)
    if svd is None:
        protosize = n_classes * n_channels
        covsize = (1 + n_classes) * n_channels
    else:
        protosize = n_classes * min(svd, n_channels)
        covsize = n_channels + n_classes * min(svd, n_channels)
    assert covest.P_.shape == (protosize, n_times)
    assert covmats.shape == (n_matrices, covsize, covsize)
    assert is_spsd(covmats)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_erp_covariances_classes(n_classes, get_mats, get_labels):
    n_matrices, n_channels, n_times = 3 * n_classes, 3, 100
    X = get_mats(n_matrices, [n_channels, n_times], "real")
    y = get_labels(n_matrices, n_classes)

    covest = ERPCovariances(classes=[0])
    covmats = covest.fit_transform(X, y)
    assert covmats.shape == (n_matrices, 2 * n_channels, 2 * n_channels)
    assert is_spsd(covmats)


def test_erp_covariances_svd_error(get_mats, get_labels):
    """Assert error on param svd"""
    n_classes, n_matrices, n_channels, n_times = 2, 4, 3, 10
    X = get_mats(n_matrices, [n_channels, n_times], "real")
    y = get_labels(n_matrices, n_classes)
    with pytest.raises(TypeError):
        ERPCovariances(svd="42").fit(X, y)


@pytest.mark.parametrize("est", estim)
@pytest.mark.parametrize("xdawn_est", estim)
def test_xdawn_covariances_est(est, xdawn_est, get_mats, get_labels):
    n_classes, nfilter = 2, 3
    n_matrices, n_channels, n_times = 4, 6, 100
    X = get_mats(n_matrices, [n_channels, n_times], "real")
    y = get_labels(n_matrices, n_classes)

    covest = XdawnCovariances(nfilter, estimator=est, xdawn_estimator=xdawn_est)
    covmats = covest.fit(X, y).transform(X)
    assert covest.get_params() == dict(
        nfilter=nfilter,
        applyfilters=True,
        classes=None,
        estimator=est,
        xdawn_estimator=xdawn_est,
        baseline_cov=None,
    )
    protosize = n_classes * min(n_channels, nfilter)
    assert covest.P_.shape == (protosize, n_times)
    covsize = n_channels + n_classes * min(n_channels, nfilter)
    assert covmats.shape == (n_matrices, covsize, covsize)
    assert is_spsd(covmats)


@pytest.mark.parametrize("n_classes", [1, 2, 3])
@pytest.mark.parametrize("nfilter", [2, 4, 6])
@pytest.mark.parametrize("applyfilters", [True, False])
@pytest.mark.parametrize("baseline", [True, False])
def test_xdawn_covariances_filters(
    n_classes, nfilter, applyfilters, baseline, get_mats, get_labels
):
    n_matrices, n_channels, n_times = 3 * n_classes, 4, 128
    X = get_mats(n_matrices, [n_channels, n_times], "real")
    y = get_labels(n_matrices, n_classes)
    if baseline:
        baseline_cov = np.identity(n_channels)
    else:
        baseline_cov = None

    covest = XdawnCovariances(
        nfilter=nfilter,
        applyfilters=applyfilters,
        baseline_cov=baseline_cov,
    )
    covmats = covest.fit_transform(X, y)
    assert covest.get_params() == dict(
        nfilter=nfilter,
        applyfilters=applyfilters,
        classes=None,
        estimator="scm",
        xdawn_estimator="scm",
        baseline_cov=baseline_cov,
    )
    protosize = n_classes * min(n_channels, nfilter)
    assert covest.P_.shape == (protosize, n_times)
    if applyfilters:
        covsize = 2 * n_classes * min(n_channels, nfilter)
    else:
        covsize = n_channels + n_classes * min(n_channels, nfilter)
    assert covmats.shape == (n_matrices, covsize, covsize)


@pytest.mark.parametrize("block_size", [1, 6, [4, 8], np.array([5, 7])])
@pytest.mark.parametrize("estim", estim)
def test_block_covariances(block_size, estim, get_mats):
    """Test BlockCovariances"""
    n_matrices, n_channels, n_times = 2, 12, 100
    X = get_mats(n_matrices, [n_channels, n_times], "real")

    covest = BlockCovariances(block_size=block_size, estimator=estim)
    covest.fit(X)
    covmats = covest.fit_transform(X)
    assert covest.get_params() == dict(block_size=block_size, estimator=estim)
    assert covmats.shape == (n_matrices, n_channels, n_channels)
    assert is_spd(covmats)


def test_block_covariances_errors(get_mats):
    """Test BlockCovariances errors"""
    n_matrices, n_channels, n_times = 2, 12, 100
    X = get_mats(n_matrices, [n_channels, n_times], "real")

    with pytest.raises(ValueError):
        BlockCovariances(block_size=5).fit_transform(X)
    with pytest.raises(ValueError):
        BlockCovariances(block_size=[4, 4, 5]).fit_transform(X)
    with pytest.raises(ValueError):
        BlockCovariances(block_size="[4, 4, 5]").fit_transform(X)


@pytest.mark.parametrize("estim", [CrossSpectra, CoSpectra])
def test_xspectra(estim, get_mats):
    """Test CrossSpectra and CoSpectra"""
    n_matrices, n_channels, n_times = 2, 3, 1000
    X = get_mats(n_matrices, [n_channels, n_times], "real")

    spest = estim().fit(X)
    spmats = spest.transform(X)
    assert spest.get_params() == dict(
        window=128, overlap=0.75, fmin=None, fmax=None, fs=None
    )
    n_freqs = 65
    assert spmats.shape == (n_matrices, n_channels, n_channels, n_freqs)
    if estim is CoSpectra:
        assert is_spsd(spmats.transpose(0, 3, 1, 2))
    else:
        assert is_hpsd(spmats.transpose(0, 3, 1, 2))

    spmats2 = spest.fit_transform(X)
    assert_array_almost_equal(spmats, spmats2)


@pytest.mark.parametrize("coh", coh)
def test_coherences(coh, get_mats):
    """Test fit Coherences"""
    n_matrices, n_channels, n_times = 2, 5, 200
    X = get_mats(n_matrices, [n_channels, n_times], "real")

    cohest = Coherences(coh=coh)
    assert cohest.get_params() == dict(
        window=128, overlap=0.75, fs=None, fmin=None, fmax=None, coh=coh
    )

    if coh == "lagged":
        cohest.set_params(**{"fs": 128, "fmin": 1, "fmax": 63})
        n_freqs = 63
    else:
        n_freqs = 65

    cohest.fit(X)
    cohmats = cohest.transform(X)
    assert cohmats.shape == (n_matrices, n_channels, n_channels, n_freqs)
    if coh in ["ordinary", "instantaneous"]:
        assert is_spsd(cohmats.transpose(0, 3, 1, 2))

    cohmats2 = cohest.fit_transform(X)
    assert_array_almost_equal(cohmats, cohmats2)


@pytest.mark.parametrize(
    "metric",
    [
        "linear",
        "poly",
        "polynomial",
        "rbf",
        "laplacian",
        "cosine",
    ],
)
def test_kernels(metric, get_mats):
    """Test Kernels"""
    n_matrices, n_channels, n_times = 2, 5, 10
    X = get_mats(n_matrices, [n_channels, n_times], "real")

    est = Kernels(metric=metric)
    kernels = est.fit(X).transform(X)
    assert kernels.shape == (n_matrices, n_channels, n_channels)
    assert is_spd(kernels)

    kernels2 = est.fit_transform(X)
    assert_array_almost_equal(kernels, kernels2)


def test_kernels_linear(get_mats):
    """Test that linear kernels are related to covariances"""
    n_matrices, n_channels, n_times = 3, 4, 50
    X = get_mats(n_matrices, [n_channels, n_times], "real")

    covmats = Covariances(estimator="cov").fit_transform(X)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.sqrt(n_times - 1)
    kernels = Kernels(metric="linear").fit_transform(X)
    assert_array_almost_equal(covmats, kernels, 6)


@pytest.mark.parametrize(
    "metric, kwds",
    [
        ("polynomial", {"degree": 2, "gamma": 0.5, "coef0": 0.8}),
        ("rbf", {"gamma": 0.5}),
        ("laplacian", {"gamma": 0.5}),
    ],
)
def test_kernels_kwds(metric, kwds, get_mats):
    n_matrices, n_channels, n_times = 3, 6, 10
    X = get_mats(n_matrices, [n_channels, n_times], "real")

    kernels_none = Kernels(metric=metric).fit_transform(X)
    kernels_kwds = Kernels(metric=metric, **kwds).fit_transform(X)
    assert_raises(AssertionError, assert_array_equal, kernels_none, kernels_kwds)


@pytest.mark.parametrize("shrinkage", [0.1, 0.9])
@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_shrinkage(shrinkage, kind, get_mats):
    """Test Shrinkage"""
    n_matrices, n_channels = 4, 3
    X = get_mats(n_matrices, n_channels, kind)

    sh = Shrinkage(shrinkage=shrinkage)
    assert sh.get_params() == dict(shrinkage=shrinkage)

    Xt = sh.fit(X).transform(X)
    assert Xt.shape == X.shape
    assert is_hpd(Xt)

    assert_array_equal(shrunk_covariance(X[0].real, shrinkage), Xt[0].real)
    assert_raises(AssertionError, assert_array_equal, X.real, Xt.real)
    assert_array_equal(X.imag, Xt.imag)

    Xt2 = sh.fit_transform(X)
    assert_array_almost_equal(Xt, Xt2)
