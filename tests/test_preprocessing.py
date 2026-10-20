import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from pyriemann.geometry.mean import gmean
from pyriemann.spatialfilters import Whitening


pytestmark = pytest.mark.numpy_only


n_components = 3
expl_var = 0.8
max_cond = 5
dim_reds = [
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


def test_whitening_fit_errors(get_mats):
    n_matrices, n_channels = 2, 6
    X = get_mats(n_matrices, n_channels, "spd")

    with pytest.raises(ValueError):  # len dim_red not equal to 1
        Whitening(dim_red={"n_components": 2, "expl_var": 0.5}).fit(X)
    with pytest.raises(ValueError):  # n_components not superior to 1
        Whitening(dim_red={"n_components": 0}).fit(X)
    with pytest.raises(ValueError):  # n_components not a int
        Whitening(dim_red={"n_components": 2.5}).fit(X)
    with pytest.raises(ValueError):  # expl_var out of bound
        Whitening(dim_red={"expl_var": 0}).fit(X)
    with pytest.raises(ValueError):  # expl_var out of bound
        Whitening(dim_red={"expl_var": 1.1}).fit(X)
    with pytest.raises(ValueError):  # max_cond not strictly superior to 1
        Whitening(dim_red={"max_cond": 1}).fit(X)
    with pytest.raises(ValueError):  # unknown key
        Whitening(dim_red={"abc": 42}).fit(X)
    with pytest.raises(ValueError):  # unknown type
        Whitening(dim_red="max_cond").fit(X)
    with pytest.raises(ValueError):  # unknown type
        Whitening(dim_red=20).fit(X)


@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize("dim_red", dim_reds)
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_whitening_fit(use_weight, dim_red, metric, get_mats, get_weights):
    n_matrices, n_channels = 20, 8
    X = get_mats(n_matrices, n_channels, "spd")
    weights = None
    if use_weight:
        weights = get_weights(n_matrices)

    whit = Whitening(dim_red=dim_red, metric=metric)
    whit.fit(X, sample_weight=weights)
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
@pytest.mark.parametrize("dim_red", dim_reds)
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_whitening_partial_fit(use_weight, alpha, dim_red, metric,
                               get_mats, get_weights):
    n_matrices, n_channels = 6, 7
    X = get_mats(n_matrices, n_channels, "spd")
    weights = None
    if use_weight:
        weights = get_weights(n_matrices)

    whit = Whitening(dim_red=dim_red, metric=metric)
    whit.fit(X, sample_weight=weights)
    whit.partial_fit(get_mats(1, n_channels, "spd"), alpha=alpha)

    whit = Whitening(dim_red=dim_red, metric=metric)
    whit.partial_fit(X, alpha=alpha)


def test_whitening_partial_fit_errors(get_mats):
    n_matrices, n_channels = 3, 3
    X = get_mats(n_matrices, n_channels, "spd")
    whit = Whitening(dim_red=None, metric="riemann").partial_fit(X)

    with pytest.raises(ValueError):
        whit.partial_fit(get_mats(2, n_channels + 1, "spd"))
    with pytest.raises(ValueError):
        whit.partial_fit(X, alpha=42)


@pytest.mark.parametrize("dim_red", dim_reds)
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_whitening_transform(dim_red, metric, get_mats):
    n_matrices, n_channels = 8, 5
    X = get_mats(n_matrices, n_channels, "spd")

    whit = Whitening(metric=metric, dim_red=dim_red).fit(X)
    Xwhit = whit.transform(X)
    if dim_red is None:
        n_comp = n_channels
    else:
        n_comp = whit.n_components_
    W = whit.filters_

    assert Xwhit.shape == (n_matrices, n_comp, n_comp)

    # mean of input matrices is identity after filtering
    M = gmean(X, metric=metric)
    assert_array_almost_equal(W.T @ M @ W, np.eye(n_comp))

    # mean of whitened matrices is identity when the metric is invariant
    # by the transformation: always for euclid; for riemann,
    # only without dim reduction (congruence by an invertible matrix)
    if metric == "euclid" or (metric == "riemann" and n_comp == n_channels):
        assert_array_almost_equal(
            gmean(Xwhit, metric=metric), np.eye(n_comp)
        )

    if dim_red is not None and "max_cond" in dim_red.keys():
        assert np.linalg.cond(M) <= max_cond


@pytest.mark.parametrize("dim_red", dim_reds)
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_whitening_fit_transform(dim_red, metric, get_mats):
    n_matrices, n_channels = 15, 6
    X = get_mats(n_matrices, n_channels, "spd")
    whit = Whitening(metric=metric, dim_red=dim_red)
    Xwhit1 = whit.fit(X).transform(X)
    Xwhit2 = whit.fit_transform(X)
    assert_array_almost_equal(Xwhit1, Xwhit2)


@pytest.mark.parametrize("dim_red", dim_reds)
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_whitening_inverse_transform(dim_red, metric, get_mats):
    n_matrices, n_channels = 20, 6
    X = get_mats(n_matrices, n_channels, "spd")

    whit = Whitening(dim_red=dim_red, metric=metric).fit(X)
    Xinvwhit = whit.inverse_transform(whit.transform(X))
    assert Xinvwhit.shape == (n_matrices, n_channels, n_channels)
    if dim_red is None:
        assert_array_almost_equal(X, Xinvwhit)
