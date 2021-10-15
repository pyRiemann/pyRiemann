from pyriemann.estimation import (
    Covariances,
    ERPCovariances,
    XdawnCovariances,
    CospCovariances,
    HankelCovariances,
    Coherences,
    Shrinkage,
)
import pytest

from pyriemann.utils.test import (is_sym_pos_def as is_spd,
                                  is_sym_pos_semi_def as is_spsd)

estim = ["cov", "scm", "lwf", "oas", "mcd", "corr", "sch"]
coh = ["ordinary", "instantaneous", "lagged", "imaginary"]


@pytest.mark.parametrize("estimator", estim)
def test_covariances(estimator, rndstate):
    """Test Covariances"""
    n_trials, n_channels, n_times = 2, 3, 100
    x = rndstate.randn(n_trials, n_channels, n_times)
    cov = Covariances(estimator=estimator)
    cov.fit(x)
    covmats = cov.fit_transform(x)
    assert cov.get_params() == dict(estimator=estimator)
    assert covmats.shape == (n_trials, n_channels, n_channels)
    assert is_spd(covmats)


def test_hankel_covariances(rndstate):
    """Test Hankel Covariances"""
    n_trials, n_channels, n_times = 2, 3, 100
    x = rndstate.randn(n_trials, n_channels, n_times)
    cov = HankelCovariances()
    cov.fit(x)
    covmats = cov.fit_transform(x)
    assert cov.get_params() == dict(estimator="scm", delays=4)
    assert covmats.shape == (n_trials, 4 * n_channels, 4 * n_channels)
    assert is_spd(covmats)


def test_hankel_covariances_delays(rndstate):
    n_trials, n_channels, n_times = 2, 3, 100
    x = rndstate.randn(n_trials, n_channels, n_times)
    cov = HankelCovariances(delays=[1, 2])
    cov.fit(x)
    covmats = cov.fit_transform(x)
    assert covmats.shape == (n_trials, 3 * n_channels, 3 * n_channels)
    assert is_spd(covmats)


@pytest.mark.parametrize("estimator", estim)
@pytest.mark.parametrize("svd", [None, 2])
def test_erp_covariances(estimator, svd, rndstate, get_labels):
    """Test fit ERPCovariances"""
    n_classes, n_trials, n_channels, n_times = 2, 4, 3, 100
    x = rndstate.randn(n_trials, n_channels, n_times)
    labels = get_labels(n_trials, n_classes)
    cov = ERPCovariances(estimator=estimator, svd=svd)
    covmats = cov.fit_transform(x, labels)
    if svd is None:
        covsize = (n_classes + 1) * n_channels
    else:
        covsize = n_classes * svd + n_channels
    assert cov.get_params() == dict(classes=None, estimator=estimator, svd=svd)
    assert covmats.shape == (n_trials, covsize, covsize)
    assert is_spsd(covmats)


def test_erp_covariances_classes(rndstate, get_labels):
    n_trials, n_channels, n_times, n_classes = 4, 3, 100, 2
    x = rndstate.randn(n_trials, n_channels, n_times)
    labels = get_labels(n_trials, n_classes)
    cov = ERPCovariances(classes=[0])
    covmats = cov.fit_transform(x, labels)
    assert covmats.shape == (n_trials, 2 * n_channels, 2 * n_channels)
    assert is_spsd(covmats)


def test_erp_covariances_svd_error(rndstate):
    # assert raise svd
    with pytest.raises(TypeError):
        ERPCovariances(svd="42")


@pytest.mark.parametrize("est", estim)
def test_xdawn_est(est, rndstate, get_labels):
    """Test fit XdawnCovariances"""
    n_classes, nfilter = 2, 2
    n_trials, n_channels, n_times = 4, 6, 100
    x = rndstate.randn(n_trials, n_channels, n_times)
    labels = get_labels(n_trials, n_classes)
    cov = XdawnCovariances(nfilter, estimator=est)
    covmats = cov.fit_transform(x, labels)
    assert cov.get_params() == dict(
        nfilter=nfilter,
        applyfilters=True,
        classes=None,
        estimator=est,
        xdawn_estimator="scm",
        baseline_cov=None,
    )
    covsize = 2 * (n_classes * nfilter)
    assert covmats.shape == (n_trials, covsize, covsize)
    assert is_spsd(covmats)


@pytest.mark.parametrize("xdawn_est", estim)
def test_xdawn_covariances_est(xdawn_est, rndstate, get_labels):
    """Test fit XdawnCovariances"""
    n_classes, nfilter = 2, 2
    n_trials, n_channels, n_times = 4, 6, 100
    x = rndstate.randn(n_trials, n_channels, n_times)
    labels = get_labels(n_trials, n_classes)
    cov = XdawnCovariances(nfilter, xdawn_estimator=xdawn_est)
    covmats = cov.fit_transform(x, labels)
    assert cov.get_params() == dict(
        nfilter=nfilter,
        applyfilters=True,
        classes=None,
        estimator="scm",
        xdawn_estimator=xdawn_est,
        baseline_cov=None,
    )
    covsize = 2 * (n_classes * nfilter)
    assert covmats.shape == (n_trials, covsize, covsize)
    assert is_spsd(covmats)


@pytest.mark.parametrize("nfilter", [2, 4])
def test_xdawn_covariances_nfilter(nfilter, rndstate, get_labels):
    """Test fit XdawnCovariances"""
    n_classes, n_trials, n_channels, n_times = 2, 4, 8, 100
    x = rndstate.randn(n_trials, n_channels, n_times)
    labels = get_labels(n_trials, n_classes)
    cov = XdawnCovariances(nfilter=nfilter)
    covmats = cov.fit_transform(x, labels)
    assert cov.get_params() == dict(
        nfilter=nfilter,
        applyfilters=True,
        classes=None,
        estimator="scm",
        xdawn_estimator="scm",
        baseline_cov=None,
    )
    covsize = 2 * (n_classes * nfilter)
    assert covmats.shape == (n_trials, covsize, covsize)
    assert is_spsd(covmats)


def test_xdawn_covariances_applyfilters(rndstate, get_labels):
    n_classes, nfilter = 2, 2
    n_trials, n_channels, n_times = 4, 6, 100
    x = rndstate.randn(n_trials, n_channels, n_times)
    labels = get_labels(n_trials, n_classes)
    cov = XdawnCovariances(nfilter=nfilter, applyfilters=False)
    covmats = cov.fit_transform(x, labels)
    covsize = n_classes * nfilter + n_channels
    assert covmats.shape == (n_trials, covsize, covsize)
    assert is_spsd(covmats)


def test_cosp_covariances(rndstate):
    """Test fit CospCovariances"""
    n_trials, n_channels, n_times = 2, 3, 1000
    x = rndstate.randn(n_trials, n_channels, n_times)
    cov = CospCovariances()
    cov.fit(x)
    covmats = cov.transform(x)
    assert cov.get_params() == dict(
        window=128, overlap=0.75, fmin=None, fmax=None, fs=None
    )
    n_freqs = 65
    assert covmats.shape == (n_trials, n_channels, n_channels, n_freqs)
    assert is_spsd(covmats.transpose(0, 3, 1, 2))


@pytest.mark.parametrize("coh", coh)
def test_coherences(coh, rndstate):
    """Test fit Coherences"""
    n_trials, n_channels, n_times = 2, 5, 200
    x = rndstate.randn(n_trials, n_channels, n_times)

    cov = Coherences(coh=coh)
    cov.fit(x)
    covmats = cov.fit_transform(x)
    assert cov.get_params() == dict(
        window=128, overlap=0.75, fmin=None, fmax=None, fs=None, coh=coh
    )
    n_freqs = 65
    assert covmats.shape == (n_trials, n_channels, n_channels, n_freqs)
    if coh in ["ordinary", "instantaneous"]:
        assert is_spsd(covmats.transpose(0, 3, 1, 2))


@pytest.mark.parametrize("shrinkage", [0.1, 0.9])
def test_shrinkage(shrinkage, rndstate):
    """Test Shrinkage"""
    n_trials, n_channels, n_times = 2, 3, 100
    x = rndstate.randn(n_trials, n_channels, n_times)
    covmats = Covariances().fit_transform(x)
    sh = Shrinkage(shrinkage=shrinkage)
    covmats = sh.fit(covmats).transform(covmats)
    assert sh.get_params() == dict(shrinkage=shrinkage)
    assert covmats.shape == (n_trials, n_channels, n_channels)
    assert is_spd(covmats)
