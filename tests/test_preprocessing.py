import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from pyriemann.spatialfilters import Whitening


def generate_cov(Nt, Ne):
    """Generate a set of cavariances matrices for test purpose"""
    rs = np.random.RandomState(1234)
    diags = 2.0 + 0.1 * rs.randn(Nt, Ne)
    A = 2*rs.rand(Ne, Ne) - 1
    A /= np.atleast_2d(np.sqrt(np.sum(A**2, 1))).T
    covmats = np.empty((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.dot(np.dot(A, np.diag(diags[i])), A.T)
    return covmats


def test_whitening():
    """Test Whitening"""
    n_trials, n_channels, n_components = 20, 6, 3
    cov = generate_cov(n_trials, n_channels)
    rs = np.random.RandomState(1234)
    w = rs.rand(n_trials)
    max_cond = 10

    # Test Init
    whit = Whitening()
    assert whit.metric=='euclid'
    assert whit.dim_red==None
    assert whit.verbose==False

    # Test Fit
    with pytest.raises(ValueError): # len dim_red not equal to 1
        Whitening(dim_red={'n_components': 2, 'expl_var': 0.5}).fit(cov)
    with pytest.raises(ValueError): # n_components not superior to 1
        Whitening(dim_red={'n_components': 0}).fit(cov)
    with pytest.raises(ValueError): # n_components not a int
        Whitening(dim_red={'n_components': 2.5}).fit(cov)
    with pytest.raises(ValueError): # expl_var out of bound
        Whitening(dim_red={'expl_var': 0}).fit(cov)
    with pytest.raises(ValueError): # expl_var out of bound
        Whitening(dim_red={'expl_var': 1.1}).fit(cov)
    with pytest.raises(ValueError): # max_cond not strictly superior to 1
        Whitening(dim_red={'max_cond': 1}).fit(cov)
    with pytest.raises(ValueError): # unknown key
        Whitening(dim_red={'abc': 42}).fit(cov)
    with pytest.raises(ValueError): # unknown type
        Whitening(dim_red='max_cond').fit(cov)
    with pytest.raises(ValueError): # unknown type
        Whitening(dim_red=20).fit(cov)

    whit = Whitening().fit(cov, sample_weight=w)
    assert whit.n_components_ == n_channels
    assert_array_equal(whit.filters_.shape, [n_channels, n_channels])
    assert_array_equal(whit.inv_filters_.shape, [n_channels, n_channels])

    whit = Whitening(
        dim_red={'n_components': n_components}
    ).fit(cov, sample_weight=w)
    assert whit.n_components_ == n_components
    assert_array_equal(whit.filters_.shape, [n_channels, n_components])
    assert_array_equal(whit.inv_filters_.shape, [n_components, n_channels])

    whit = Whitening(dim_red={'expl_var': 0.9}).fit(cov, sample_weight=w)
    assert whit.n_components_ <= n_channels
    assert_array_equal(whit.filters_.shape, [n_channels, whit.n_components_])
    assert_array_equal(whit.inv_filters_.shape,
                       [whit.n_components_, n_channels])

    whit = Whitening(dim_red={'max_cond': max_cond}).fit(cov, sample_weight=w)
    assert whit.n_components_ <= n_channels
    assert_array_equal(whit.filters_.shape, [n_channels, whit.n_components_])
    assert_array_equal(whit.inv_filters_.shape,
                       [whit.n_components_, n_channels])

    # Test transform
    whit = Whitening().fit(cov)
    cov_w = whit.transform(cov)
    assert_array_equal(cov_w.shape, [n_trials, n_channels, n_channels])
    # after whitening, mean = identity
    assert_array_almost_equal(cov_w.mean(axis=0), np.eye(n_channels))

    whit = Whitening(dim_red={'n_components': n_components}).fit(cov)
    cov_w = whit.transform(cov)
    n_components_ = whit.n_components_
    assert_array_equal(cov_w.shape, [n_trials, n_components_, n_components_])
    # after whitening, mean = identity
    assert_array_almost_equal(cov_w.mean(axis=0), np.eye(n_components_))

    whit = Whitening(dim_red={'expl_var': 0.9}).fit(cov)
    cov_w = whit.transform(cov)
    n_components_ = whit.n_components_
    assert_array_equal(cov_w.shape, [n_trials, n_components_, n_components_])
    # after whitening, mean = identity
    assert_array_almost_equal(cov_w.mean(axis=0), np.eye(n_components_))

    whit = Whitening(dim_red={'max_cond': max_cond}).fit(cov)
    cov_w = whit.transform(cov)
    n_components_ = whit.n_components_
    assert_array_equal(cov_w.shape, [n_trials, n_components_, n_components_])
    # after whitening, mean = identity
    mean = cov_w.mean(axis=0)
    assert_array_almost_equal(mean, np.eye(n_components_))
    assert np.linalg.cond(mean) <= max_cond

    # Test inverse_transform
    whit = Whitening().fit(cov)
    cov_iw = whit.inverse_transform(whit.transform(cov))
    assert_array_equal(cov_iw.shape, [n_trials, n_channels, n_channels])
    assert_array_almost_equal(cov, cov_iw)

    whit = Whitening(dim_red={'n_components': n_components}).fit(cov)
    cov_iw = whit.inverse_transform(whit.transform(cov))
    assert_array_equal(cov_iw.shape, [n_trials, n_channels, n_channels])

    whit = Whitening(dim_red={'expl_var': 0.9}).fit(cov)
    cov_iw = whit.inverse_transform(whit.transform(cov))
    assert_array_equal(cov_iw.shape, [n_trials, n_channels, n_channels])

    whit = Whitening(dim_red={'max_cond': max_cond}).fit(cov)
    cov_iw = whit.inverse_transform(whit.transform(cov))
    assert_array_equal(cov_iw.shape, [n_trials, n_channels, n_channels])
