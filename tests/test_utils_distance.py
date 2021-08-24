from conftest import get_covmats, get_distances
import numpy as np
from numpy.testing import assert_array_almost_equal
from pyriemann.utils.distance import (
    distance_riemann,
    distance_euclid,
    distance_logeuclid,
    distance_logdet,
    distance_kullback,
    distance_kullback_right,
    distance_kullback_sym,
    distance_wasserstein,
    distance,
    pairwise_distance,
    _check_distance_method,
)
from pyriemann.utils.geodesic import geodesic
import pytest
from pytest import approx


def get_dist_func():
    dist_func = [
        distance_riemann,
        distance_logeuclid,
        distance_euclid,
        distance_logdet,
        distance_kullback,
        distance_kullback_right,
        distance_kullback_sym,
        distance_wasserstein,
    ]
    for df in dist_func:
        yield df


@pytest.mark.parametrize("dist", get_distances())
def test_check_distance_str(dist):
    _check_distance_method(dist)


@pytest.mark.parametrize("dist", get_dist_func())
def test_check_distance_func(dist):
    _check_distance_method(dist)


def test_check_distance_error():
    with pytest.raises(ValueError):
        _check_distance_method("universe")
    with pytest.raises(ValueError):
        _check_distance_method(42)


@pytest.mark.parametrize("dist", get_dist_func())
def test_distance_func_eye(dist):
    n_channels = 3
    A = 2 * np.eye(n_channels)
    B = 2 * np.eye(n_channels)
    assert dist(A, B) == approx(0)


@pytest.mark.parametrize("dist", get_dist_func())
def test_distance_func_rand(dist, get_covmats):
    n_trials, n_channels = 2, 6
    covmats = get_covmats(n_trials, n_channels)
    A, C = covmats[0], covmats[1]
    B = geodesic(A, C, alpha=0.5)
    assert dist(A, B) < dist(A, C)


@pytest.mark.parametrize("dist, dfunc", zip(get_distances(), get_dist_func()))
def test_distance_wrapper(dist, dfunc, get_covmats):
    n_trials, n_channels = 2, 5
    covmats = get_covmats(n_trials, n_channels)
    A, B = covmats[0], covmats[1]
    assert distance(A, B, metric=dist) == dfunc(A, B)


def test_pairwise_distance_matrix(get_covmats):
    n_trials, n_channels = 6, 5
    covmats = get_covmats(n_trials, n_channels)
    A, B = covmats[:4], covmats[4:]
    pdist = pairwise_distance(A, B)
    assert pdist.shape == (4, 2)
