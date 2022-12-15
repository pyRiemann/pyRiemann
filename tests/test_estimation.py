import numpy as np
import pytest

from pyriemann.estimation import (
    Covariances,
    ERPCovariances,
    XdawnCovariances,
    CospCovariances,
    HankelCovariances,
    Coherences,
    Shrinkage,
    BlockCovariances
)
from pyriemann.utils.test import (is_sym_pos_def as is_spd,
                                  is_sym_pos_semi_def as is_spsd)

estim = ["cov", "scm", "lwf", "oas", "mcd", "corr", "sch"]
coh = ["ordinary", "instantaneous", "lagged", "imaginary"]


@pytest.mark.parametrize("estimator", estim)
def test_covariances(estimator, rndstate):
    """Test Covariances"""
    n_matrices, n_channels, n_times = 2, 3, 100
    x = rndstate.randn(n_matrices, n_channels, n_times)
    cov = Covariances(estimator=estimator).fit(x)
    covmats = cov.fit_transform(x)
    assert cov.get_params() == dict(estimator=estimator)
    assert covmats.shape == (n_matrices, n_channels, n_channels)
    assert is_spd(covmats)


@pytest.mark.parametrize("delays", [4, [1, 2]])
def test_hankel_covariances_delays(delays, rndstate):
    n_matrices, n_channels, n_times = 2, 3, 100
    x = rndstate.randn(n_matrices, n_channels, n_times)
    cov = HankelCovariances(delays=delays).fit(x)
    covmats = cov.fit_transform(x)
    assert cov.get_params() == dict(estimator="scm", delays=delays)
    if isinstance(delays, list):
        n_delays = 1 + len(delays)
    elif isinstance(delays, int):
        n_delays = delays
    assert covmats.shape == (n_matrices, n_delays * n_channels,
                             n_delays * n_channels)
    assert is_spd(covmats)


@pytest.mark.parametrize("estimator", estim)
@pytest.mark.parametrize("svd", [None, 2, 3, 4])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
def test_erp_covariances(estimator, svd, n_classes, rndstate, get_labels):
    """Test fit ERPCovariances"""
    n_matrices, n_channels, n_times = 3 * n_classes, 3, 100
    x = rndstate.randn(n_matrices, n_channels, n_times)
    labels = get_labels(n_matrices, n_classes)
    cov = ERPCovariances(estimator=estimator, svd=svd)
    covmats = cov.fit_transform(x, labels)
    assert cov.get_params() == dict(classes=None, estimator=estimator, svd=svd)
    if svd is None:
        protosize = n_classes * n_channels
        covsize = (1 + n_classes) * n_channels
    else:
        protosize = n_classes * min(svd, n_channels)
        covsize = n_channels + n_classes * min(svd, n_channels)
    assert cov.P_.shape == (protosize, n_times)
    assert covmats.shape == (n_matrices, covsize, covsize)
    assert is_spsd(covmats)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_erp_covariances_classes(n_classes, rndstate, get_labels):
    n_matrices, n_channels, n_times = 3 * n_classes, 3, 100
    x = rndstate.randn(n_matrices, n_channels, n_times)
    labels = get_labels(n_matrices, n_classes)
    cov = ERPCovariances(classes=[0])
    covmats = cov.fit_transform(x, labels)
    assert covmats.shape == (n_matrices, 2 * n_channels, 2 * n_channels)
    assert is_spsd(covmats)


def test_erp_covariances_svd_error(rndstate, get_labels):
    """ assert raise svd """
    n_classes, n_matrices, n_channels, n_times = 2, 4, 3, 10
    x = rndstate.randn(n_matrices, n_channels, n_times)
    labels = get_labels(n_matrices, n_classes)
    with pytest.raises(TypeError):
        ERPCovariances(svd="42").fit(x, labels)


@pytest.mark.parametrize("est", estim)
@pytest.mark.parametrize("xdawn_est", estim)
def test_xdawn_covariances_est(est, xdawn_est, rndstate, get_labels):
    n_classes, nfilter = 2, 3
    n_matrices, n_channels, n_times = 4, 6, 100
    x = rndstate.randn(n_matrices, n_channels, n_times)
    labels = get_labels(n_matrices, n_classes)
    cov = XdawnCovariances(nfilter, estimator=est, xdawn_estimator=xdawn_est)
    covmats = cov.fit_transform(x, labels)
    assert cov.get_params() == dict(
        nfilter=nfilter,
        applyfilters=True,
        classes=None,
        estimator=est,
        xdawn_estimator=xdawn_est,
        baseline_cov=None,
    )
    protosize = n_classes * min(n_channels, nfilter)
    assert cov.P_.shape == (protosize, n_times)
    covsize = n_channels + n_classes * min(n_channels, nfilter)
    assert covmats.shape == (n_matrices, covsize, covsize)
    assert is_spsd(covmats)


@pytest.mark.parametrize("n_classes", [1, 2, 3])
@pytest.mark.parametrize("nfilter", [2, 4, 6])
@pytest.mark.parametrize("applyfilters", [True, False])
@pytest.mark.parametrize("baseline", [True, False])
def test_xdawn_covariances_filters(n_classes, nfilter, applyfilters, baseline,
                                   rndstate, get_labels):
    n_matrices, n_channels, n_times = 3 * n_classes, 4, 128
    x = rndstate.randn(n_matrices, n_channels, n_times)
    labels = get_labels(n_matrices, n_classes)
    if baseline:
        baseline_cov = np.identity(n_channels)
    else:
        baseline_cov = None
    cov = XdawnCovariances(
        nfilter=nfilter,
        applyfilters=applyfilters,
        baseline_cov=baseline_cov,
    )
    covmats = cov.fit_transform(x, labels)
    assert cov.get_params() == dict(
        nfilter=nfilter,
        applyfilters=applyfilters,
        classes=None,
        estimator="scm",
        xdawn_estimator="scm",
        baseline_cov=baseline_cov,
    )
    protosize = n_classes * min(n_channels, nfilter)
    assert cov.P_.shape == (protosize, n_times)
    if applyfilters:
        covsize = 2 * n_classes * min(n_channels, nfilter)
    else:
        covsize = n_channels + n_classes * min(n_channels, nfilter)
    assert covmats.shape == (n_matrices, covsize, covsize)


@pytest.mark.parametrize("estimator", estim)
def test_block_covariances_est(estimator, rndstate):
    """Test BlockCovariances estimators"""
    n_matrices, n_channels, n_times = 2, 12, 100
    x = rndstate.randn(n_matrices, n_channels, n_times)
    cov = BlockCovariances(block_size=6, estimator=estimator)
    cov.fit(x)
    covmats = cov.fit_transform(x)
    assert cov.get_params() == dict(block_size=6, estimator=estimator)
    assert covmats.shape == (n_matrices, n_channels, n_channels)
    assert is_spd(covmats)


@pytest.mark.parametrize("block_size", [1, 6, [4, 8]])
def test_block_covariances_blocks(block_size, rndstate):
    """Test BlockCovariances fit"""
    n_matrices, n_channels, n_times = 2, 12, 100
    x = rndstate.randn(n_matrices, n_channels, n_times)
    cov = BlockCovariances(block_size=block_size)
    cov.fit(x)
    covmats = cov.fit_transform(x)
    assert cov.get_params() == dict(block_size=block_size, estimator='scm')
    assert covmats.shape == (n_matrices, n_channels, n_channels)
    assert is_spd(covmats)


def test_block_covariances_int_value_error(rndstate):
    """Test BlockCovariances error"""
    n_matrices, n_channels, n_times = 2, 12, 100
    x = rndstate.randn(n_matrices, n_channels, n_times)
    cov = BlockCovariances(block_size=5)
    with pytest.raises(ValueError):
        cov.fit_transform(x)


def test_block_covariances_array_value_error(rndstate):
    """Test BlockCovariances error"""
    n_matrices, n_channels, n_times = 2, 12, 100
    x = rndstate.randn(n_matrices, n_channels, n_times)
    cov = BlockCovariances(block_size=[4, 4, 5])
    with pytest.raises(ValueError):
        cov.fit_transform(x)


def test_block_covariances_block_size_type_error(rndstate):
    """Test BlockCovariances error"""
    n_matrices, n_channels, n_times = 2, 12, 100
    x = rndstate.randn(n_matrices, n_channels, n_times)
    cov = BlockCovariances(block_size='[4, 4, 5]')
    with pytest.raises(ValueError):
        cov.fit_transform(x)


def test_cosp_covariances(rndstate):
    """Test fit CospCovariances"""
    n_matrices, n_channels, n_times = 2, 3, 1000
    x = rndstate.randn(n_matrices, n_channels, n_times)
    cov = CospCovariances()
    cov.fit(x)
    covmats = cov.transform(x)
    assert cov.get_params() == dict(
        window=128, overlap=0.75, fmin=None, fmax=None, fs=None
    )
    n_freqs = 65
    assert covmats.shape == (n_matrices, n_channels, n_channels, n_freqs)
    assert is_spsd(covmats.transpose(0, 3, 1, 2))


@pytest.mark.parametrize("coh", coh)
def test_coherences(coh, rndstate):
    """Test fit Coherences"""
    n_matrices, n_channels, n_times = 2, 5, 200
    x = rndstate.randn(n_matrices, n_channels, n_times)

    cov = Coherences(coh=coh)
    assert cov.get_params() == dict(
        window=128, overlap=0.75, fs=None, fmin=None, fmax=None, coh=coh
    )

    if coh == 'lagged':
        cov.set_params(**{'fs': 128, 'fmin': 1, 'fmax': 63})
        n_freqs = 63
    else:
        n_freqs = 65

    cov.fit(x)
    covmats = cov.fit_transform(x)
    assert covmats.shape == (n_matrices, n_channels, n_channels, n_freqs)
    if coh in ["ordinary", "instantaneous"]:
        assert is_spsd(covmats.transpose(0, 3, 1, 2))


@pytest.mark.parametrize("shrinkage", [0.1, 0.9])
def test_shrinkage(shrinkage, rndstate):
    """Test Shrinkage"""
    n_matrices, n_channels, n_times = 2, 3, 100
    x = rndstate.randn(n_matrices, n_channels, n_times)
    covmats = Covariances().fit_transform(x)
    sh = Shrinkage(shrinkage=shrinkage)
    covmats = sh.fit(covmats).transform(covmats)
    assert sh.get_params() == dict(shrinkage=shrinkage)
    assert covmats.shape == (n_matrices, n_channels, n_channels)
    assert is_spd(covmats)
