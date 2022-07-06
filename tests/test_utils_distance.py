from conftest import get_distances
import numpy as np
import pytest
from pytest import approx

from pyriemann.utils.distance import (
    distance_euclid,
    distance_harmonic,
    distance_kullback,
    distance_kullback_right,
    distance_kullback_sym,
    distance_logdet,
    distance_logeuclid,
    distance_riemann,
    distance_wasserstein,
    distance,
    pairwise_distance,
    _check_distance_method,
)
from pyriemann.utils.geodesic import geodesic


def get_dist_func():
    dist_func = [
        distance_euclid,
        distance_harmonic,
        distance_kullback,
        distance_kullback_right,
        distance_kullback_sym,
        distance_logdet,
        distance_logeuclid,
        distance_riemann,
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
def test_distance_func_geodesic(dist, get_covmats):
    n_matrices, n_channels = 2, 6
    covmats = get_covmats(n_matrices, n_channels)
    A, C = covmats[0], covmats[1]
    B = geodesic(A, C, alpha=0.5)
    assert dist(A, B) < dist(A, C)


@pytest.mark.parametrize("dist", get_dist_func())
def test_distance_func_separability(dist, get_covmats):
    n_matrices, n_channels = 1, 6
    covmats = get_covmats(n_matrices, n_channels)
    assert dist(covmats[0], covmats[0]) == approx(0, abs=1e-7)


@pytest.mark.parametrize(
    "dist", [
        distance_euclid,
        distance_harmonic,
        distance_kullback_sym,
        distance_logdet,
        distance_logeuclid,
        distance_riemann,
        distance_wasserstein,
    ]
)
def test_distance_func_symmetry(dist, get_covmats):
    n_matrices, n_channels = 2, 5
    covmats = get_covmats(n_matrices, n_channels)
    A, B = covmats[0], covmats[1]
    assert dist(A, B) == approx(dist(B, A))


@pytest.mark.parametrize("dist", get_dist_func())
def test_distance_func_triangle_inequality(dist, get_covmats):
    n_matrices, n_channels = 3, 4
    covmats = get_covmats(n_matrices, n_channels)
    A, B, C = covmats[0], covmats[1], covmats[2]
    assert dist(A, B) <= dist(A, C) + dist(C, B)


def test_distance_implementation_kullback(get_covmats):
    n_matrices, n_channels = 2, 6
    covmats = get_covmats(n_matrices, n_channels)
    A, B = covmats[0], covmats[1]
    dist = 0.5*(np.trace(np.linalg.inv(B) @ A) - n_channels
                + np.log(np.linalg.det(B) / np.linalg.det(A)))
    assert distance_kullback(A, B) == approx(dist)


def test_distance_implementation_logdet(get_covmats):
    n_matrices, n_channels = 2, 6
    covmats = get_covmats(n_matrices, n_channels)
    A, B = covmats[0], covmats[1]
    dist = np.sqrt(np.log(np.linalg.det((A + B) / 2.0))
                   - 0.5 * np.log(np.linalg.det(A)*np.linalg.det(B)))
    assert distance_logdet(A, B) == approx(dist)


@pytest.mark.parametrize("dist, dfunc", zip(get_distances(), get_dist_func()))
def test_distance_wrapper(dist, dfunc, get_covmats):
    n_matrices, n_channels = 2, 5
    covmats = get_covmats(n_matrices, n_channels)
    A, B = covmats[0], covmats[1]
    assert distance(A, B, metric=dist) == dfunc(A, B)


@pytest.mark.parametrize("dist", get_dist_func())
def test_distance_wrapper_between_set_and_matrix(dist, get_covmats):
    n_matrices, n_channels = 10, 4
    covmats = get_covmats(n_matrices, n_channels)
    assert distance(covmats, covmats[-1], metric=dist).shape == (n_matrices, 1)


def test_pairwise_distance_matrix(get_covmats):
    n_matrices, n_channels = 6, 5
    covmats = get_covmats(n_matrices, n_channels)
    n_subset = 4
    A, B = covmats[:n_subset], covmats[n_subset:]
    pdist = pairwise_distance(A, B)
    assert pdist.shape == (n_subset, 2)
