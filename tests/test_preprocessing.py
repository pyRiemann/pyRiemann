import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from pyriemann.spatialfilters import Whitening
from pyriemann.utils.mean import mean_covariance

n_components = 3
expl_var = 0.9
max_cond = 10
dim_red = [
    None,
    {"n_components": n_components},
    {"expl_var": expl_var},
    {"max_cond": max_cond},
]


def test_whitening_init():
    whit = Whitening()
    assert whit.metric == "euclid"
    assert whit.dim_red is None
    assert not whit.verbose


def test_whitening_error(rndstate, get_covmats):
    """Test Whitening"""
    n_matrices, n_channels = 20, 6
    cov = get_covmats(n_matrices, n_channels)
    # Test Fit
    with pytest.raises(ValueError):  # len dim_red not equal to 1
        Whitening(dim_red={"n_components": 2, "expl_var": 0.5}).fit(cov)
    with pytest.raises(ValueError):  # n_components not superior to 1
        Whitening(dim_red={"n_components": 0}).fit(cov)
    with pytest.raises(ValueError):  # n_components not a int
        Whitening(dim_red={"n_components": 2.5}).fit(cov)
    with pytest.raises(ValueError):  # expl_var out of bound
        Whitening(dim_red={"expl_var": 0}).fit(cov)
    with pytest.raises(ValueError):  # expl_var out of bound
        Whitening(dim_red={"expl_var": 1.1}).fit(cov)
    with pytest.raises(ValueError):  # max_cond not strictly superior to 1
        Whitening(dim_red={"max_cond": 1}).fit(cov)
    with pytest.raises(ValueError):  # unknown key
        Whitening(dim_red={"abc": 42}).fit(cov)
    with pytest.raises(ValueError):  # unknown type
        Whitening(dim_red="max_cond").fit(cov)
    with pytest.raises(ValueError):  # unknown type
        Whitening(dim_red=20).fit(cov)


@pytest.mark.parametrize("dim_red", dim_red)
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_whitening_fit(dim_red, metric, rndstate, get_covmats):
    """Test Whitening fit"""
    n_matrices, n_channels = 20, 6
    cov = get_covmats(n_matrices, n_channels)

    w = rndstate.rand(n_matrices)
    whit = Whitening(dim_red=dim_red, metric=metric).fit(cov, sample_weight=w)
    if dim_red is None:
        n_comp = n_channels
    else:
        n_comp = whit.n_components_

    if dim_red and list(dim_red.keys())[0] in ["expl_var", "max_cond"]:
        assert whit.n_components_ <= n_channels
    else:
        assert whit.n_components_ == n_comp
    assert whit.filters_.shape == (n_channels, n_comp)
    assert whit.inv_filters_.shape == (n_comp, n_channels)


@pytest.mark.parametrize("dim_red", dim_red)
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_whitening_transform(dim_red, metric, rndstate, get_covmats):
    """Test Whitening transform"""
    n_matrices, n_channels = 20, 6
    cov = get_covmats(n_matrices, n_channels)
    whit = Whitening(metric=metric).fit(cov)
    cov_w = whit.transform(cov)
    if dim_red is None:
        n_comp = n_channels
    else:
        n_comp = whit.n_components_
    assert cov_w.shape == (n_matrices, n_comp, n_comp)
    # after whitening, mean = identity
    assert_array_almost_equal(
        mean_covariance(cov_w, metric=metric),
        np.eye(n_comp),
    )
    if dim_red is not None and "max_cond" in dim_red.keys():
        assert np.linalg.cond(cov_w.mean(axis=0)) <= max_cond


@pytest.mark.parametrize("dim_red", dim_red)
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_whitening_inverse_transform(dim_red, metric, rndstate, get_covmats):
    """Test Whitening inverse transform"""
    n_matrices, n_channels = 20, 6
    cov = get_covmats(n_matrices, n_channels)
    whit = Whitening(dim_red=dim_red, metric=metric).fit(cov)
    cov_iw = whit.inverse_transform(whit.transform(cov))
    assert cov_iw.shape == (n_matrices, n_channels, n_channels)
    if dim_red is None:
        assert_array_almost_equal(cov, cov_iw)
