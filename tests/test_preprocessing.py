import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from pyriemann.geometry.geodesic import geodesic
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


def get_spread_mats(get_mats, n_matrices, n_channels):
    """Generate SPD matrices with spread eigenvalues, so that dimension
    reductions based on expl_var and max_cond keep less than n_channels."""
    X = get_mats(n_matrices, n_channels, "spd")
    D = np.diag(np.logspace(0, 1.5, n_channels))
    return D @ X @ D


def get_n_components(dim_red, M):
    """Get expected number of components, from the mean M."""
    n_channels = M.shape[-1]
    if dim_red is None:
        return n_channels
    key, val = next(iter(dim_red.items()))
    eigvals = np.linalg.eigvalsh(M)[::-1]
    if key == "n_components":
        return min(val, n_channels)
    elif key == "expl_var":
        cum_expl_var = np.cumsum(eigvals) / np.sum(eigvals)
        return int(np.argmax(cum_expl_var >= val)) + 1
    elif key == "max_cond":
        return int(np.sum(eigvals[0] / eigvals < val))


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
    X = get_spread_mats(get_mats, n_matrices, n_channels)
    weights = None
    if use_weight:
        weights = get_weights(n_matrices)

    whit = Whitening(dim_red=dim_red, metric=metric)
    whit.fit(X, sample_weight=weights)
    W = whit.filters_

    M = gmean(X, metric=metric, sample_weight=weights)
    n_comp = get_n_components(dim_red, M)
    if dim_red is not None:
        assert n_comp < n_channels  # check that test data are reduced
    assert whit.n_components_ == n_comp
    assert whit.filters_.shape == (n_channels, n_comp)
    assert whit.inv_filters_.shape == (n_comp, n_channels)

    # inverse filters are a left inverse of filters
    assert_array_almost_equal(whit.inv_filters_ @ W, np.eye(n_comp))

    # condition number of the kept part of the mean is lower than max_cond,
    # and adding one more component would exceed it
    if dim_red is not None and "max_cond" in dim_red.keys():
        eigvals = np.linalg.eigvalsh(M)[::-1]
        assert eigvals[0] / eigvals[n_comp - 1] < max_cond
        if n_comp < n_channels:
            assert eigvals[0] / eigvals[n_comp] >= max_cond

    # mean of input matrices is identity after filtering
    assert_array_almost_equal(W.T @ M @ W, np.eye(n_comp))


@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize("alpha", [0, 0.5, 1, None])
@pytest.mark.parametrize("dim_red", dim_reds)
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_whitening_partial_fit(use_weight, alpha, dim_red, metric,
                               get_mats, get_weights):
    n_matrices, n_channels = 6, 7
    X = get_spread_mats(get_mats, n_matrices, n_channels)
    weights = None
    if use_weight:
        weights = get_weights(n_matrices)

    # partial fit after fit
    whit = Whitening(dim_red=dim_red, metric=metric)
    whit.fit(X, sample_weight=weights)
    n_comp = whit.n_components_
    W = whit.filters_.copy()

    X2 = get_spread_mats(get_mats, 2, n_channels)
    whit.partial_fit(X2, alpha=alpha)
    assert whit.n_components_ == n_comp
    assert whit.filters_.shape == (n_channels, n_comp)
    if alpha == 0:
        assert_array_almost_equal(whit.filters_, W)
    else:
        # filters whiten the updated mean
        a = 2 / (n_matrices + 2) if alpha is None else alpha
        M = geodesic(
            gmean(X, metric=metric, sample_weight=weights),
            gmean(X2, metric=metric),
            a,
            metric=metric,
        )
        W = whit.filters_
        assert_array_almost_equal(W.T @ M @ W, np.eye(n_comp))

    # partial fit from scratch is equivalent to fit
    whit = Whitening(dim_red=dim_red, metric=metric)
    whit.partial_fit(X, sample_weight=weights, alpha=alpha)
    if alpha == 0:
        assert not hasattr(whit, "filters_")
    else:
        whit_fit = Whitening(dim_red=dim_red, metric=metric)
        whit_fit.fit(X, sample_weight=weights)
        assert whit.n_components_ == whit_fit.n_components_
        assert_array_almost_equal(whit.filters_, whit_fit.filters_)
        assert_array_almost_equal(whit.inv_filters_, whit_fit.inv_filters_)


@pytest.mark.parametrize("dim_red", dim_reds)
def test_whitening_partial_fit_incremental(dim_red, get_mats):
    """Test that incremental partial fits are equivalent to a full fit, for
    the Euclidean metric where the update of the mean is exact."""
    n_matrices, n_channels = 12, 6
    X = get_spread_mats(get_mats, n_matrices, n_channels)

    whit = Whitening(dim_red=dim_red).fit(X[:4])
    whit.partial_fit(X[4:7])
    whit.partial_fit(X[7:])
    whit_fit = Whitening(dim_red=dim_red).fit(X)
    if dim_red is None or "n_components" in dim_red:
        # otherwise, n_components_ is fixed at first fit and can differ
        assert whit.n_components_ == whit_fit.n_components_
        assert_array_almost_equal(whit.filters_, whit_fit.filters_)
    W = whit.filters_
    assert_array_almost_equal(
        W.T @ np.mean(X, axis=0) @ W, np.eye(whit.n_components_)
    )


def test_whitening_partial_fit_errors(get_mats):
    n_matrices, n_channels = 3, 3
    X = get_mats(n_matrices, n_channels, "spd")
    whit = Whitening(dim_red=None, metric="riemann").partial_fit(X)

    with pytest.raises(ValueError):
        whit.partial_fit(get_mats(2, n_channels + 1, "spd"))
    with pytest.raises(ValueError):
        whit.partial_fit(X, alpha=42)


@pytest.mark.parametrize("dim_red, n_comp", [
    (None, 4),
    ({"n_components": 2}, 2),
    ({"n_components": 10}, 4),
    ({"expl_var": 0.5}, 1),
    ({"expl_var": 0.9}, 3),
    ({"expl_var": 0.95}, 4),
    ({"expl_var": 1.0}, 4),
    ({"max_cond": 1.5}, 1),
    ({"max_cond": 3}, 2),
    ({"max_cond": 5}, 3),
    ({"max_cond": 100}, 4),
])
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_whitening_n_components(dim_red, n_comp, metric):
    """Test number of components on matrices with known mean.

    Eigenvalues of mean are [8, 4, 2, 1], giving cumulative explained variances
    [0.53, 0.8, 0.93, 1] and condition numbers [1, 2, 4, 8].
    """
    X = np.tile(np.diag([8.0, 4.0, 2.0, 1.0]), (3, 1, 1))

    whit = Whitening(dim_red=dim_red, metric=metric).fit(X)
    assert whit.n_components_ == n_comp
    assert whit.transform(X).shape == (3, n_comp, n_comp)

    whit = Whitening(dim_red=dim_red, metric=metric).partial_fit(X)
    assert whit.n_components_ == n_comp


@pytest.mark.parametrize("dim_red", dim_reds)
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_whitening_transform(dim_red, metric, get_mats):
    n_matrices, n_channels = 8, 5
    X = get_spread_mats(get_mats, n_matrices, n_channels)

    whit = Whitening(metric=metric, dim_red=dim_red).fit(X)
    Xwhit = whit.transform(X)
    M = gmean(X, metric=metric)
    n_comp = get_n_components(dim_red, M)
    W = whit.filters_

    assert Xwhit.shape == (n_matrices, n_comp, n_comp)
    assert_array_almost_equal(Xwhit, W.T @ X @ W)

    # mean of whitened matrices is identity when the metric is invariant
    # by the transformation: always for euclid; for riemann,
    # only without dim reduction (congruence by an invertible matrix)
    if metric == "euclid" or (metric == "riemann" and n_comp == n_channels):
        assert_array_almost_equal(
            gmean(Xwhit, metric=metric), np.eye(n_comp)
        )


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
