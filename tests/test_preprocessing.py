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


def test_whitening_fit_errors(rndstate, get_mats):
    """Test Whitening"""
    n_matrices, n_channels = 2, 6
    mats = get_mats(n_matrices, n_channels, "spd")

    with pytest.raises(ValueError):  # len dim_red not equal to 1
        Whitening(dim_red={"n_components": 2, "expl_var": 0.5}).fit(mats)
    with pytest.raises(ValueError):  # n_components not superior to 1
        Whitening(dim_red={"n_components": 0}).fit(mats)
    with pytest.raises(ValueError):  # n_components not a int
        Whitening(dim_red={"n_components": 2.5}).fit(mats)
    with pytest.raises(ValueError):  # expl_var out of bound
        Whitening(dim_red={"expl_var": 0}).fit(mats)
    with pytest.raises(ValueError):  # expl_var out of bound
        Whitening(dim_red={"expl_var": 1.1}).fit(mats)
    with pytest.raises(ValueError):  # max_cond not strictly superior to 1
        Whitening(dim_red={"max_cond": 1}).fit(mats)
    with pytest.raises(ValueError):  # unknown key
        Whitening(dim_red={"abc": 42}).fit(mats)
    with pytest.raises(ValueError):  # unknown type
        Whitening(dim_red="max_cond").fit(mats)
    with pytest.raises(ValueError):  # unknown type
        Whitening(dim_red=20).fit(mats)


@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize("dim_red", dim_red)
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_whitening_fit(use_weight, dim_red, metric, get_mats, get_weights):
    """Test Whitening fit"""
    n_matrices, n_channels = 20, 6
    mats = get_mats(n_matrices, n_channels, "spd")
    if use_weight:
        weights = get_weights(n_matrices)
    else:
        weights = None

    whit = Whitening(dim_red=dim_red, metric=metric)
    whit.fit(mats, sample_weight=weights)
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


@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize("alpha", [0, 0.5, 1, None])
@pytest.mark.parametrize("dim_red", dim_red)
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_whitening_partial_fit(use_weight, alpha, dim_red, metric,
                               get_mats, get_weights):
    """Test Whitening partial_fit"""
    n_matrices, n_channels = 6, 5
    mats = get_mats(n_matrices, n_channels, "spd")
    if use_weight:
        weights = get_weights(n_matrices)
    else:
        weights = None

    whit = Whitening(dim_red=dim_red, metric=metric)
    whit.fit(mats, sample_weight=weights)
    whit.partial_fit(get_mats(1, n_channels, "spd"), alpha=alpha)

    whit = Whitening(dim_red=dim_red, metric=metric)
    whit.partial_fit(mats, alpha=alpha)


def test_whitening_partial_fit_errors(get_mats):
    n_matrices, n_channels = 3, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    whit = Whitening(dim_red=None, metric="riemann").partial_fit(mats)

    with pytest.raises(ValueError):
        whit.partial_fit(get_mats(2, n_channels + 1, "spd"))
    with pytest.raises(ValueError):
        whit.partial_fit(mats, alpha=42)


@pytest.mark.parametrize("dim_red", dim_red)
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_whitening_transform(dim_red, metric, rndstate, get_mats):
    """Test Whitening transform"""
    n_matrices, n_channels = 20, 6
    mats = get_mats(n_matrices, n_channels, "spd")

    whit = Whitening(metric=metric).fit(mats)
    whitmats = whit.transform(mats)
    if dim_red is None:
        n_comp = n_channels
    else:
        n_comp = whit.n_components_
    assert whitmats.shape == (n_matrices, n_comp, n_comp)
    # after whitening, mean = identity
    assert_array_almost_equal(
        mean_covariance(whitmats, metric=metric),
        np.eye(n_comp),
        decimal=3,
    )
    if dim_red is not None and "max_cond" in dim_red.keys():
        assert np.linalg.cond(whitmats.mean(axis=0)) <= max_cond


@pytest.mark.parametrize("dim_red", dim_red)
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_whitening_inverse_transform(dim_red, metric, rndstate, get_mats):
    """Test Whitening inverse transform"""
    n_matrices, n_channels = 20, 6
    mats = get_mats(n_matrices, n_channels, "spd")

    whit = Whitening(dim_red=dim_red, metric=metric).fit(mats)
    invwhitmats = whit.inverse_transform(whit.transform(mats))
    assert invwhitmats.shape == (n_matrices, n_channels, n_channels)
    if dim_red is None:
        assert_array_almost_equal(mats, invwhitmats)
