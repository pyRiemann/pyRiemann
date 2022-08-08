from conftest import get_metrics
import numpy as np
from pyriemann.utils.tangentspace import (
    exp_map_euclid, exp_map_logeuclid, exp_map_riemann,
    log_map_euclid, log_map_logeuclid, log_map_riemann,
    tangent_space, untangent_space, transport
)
import pytest
from pytest import approx


@pytest.mark.parametrize(
    "fun_map", [exp_map_euclid, exp_map_logeuclid, exp_map_riemann,
                log_map_euclid, log_map_logeuclid, log_map_riemann]
)
def test_maps_shape(fun_map, get_covmats):
    """Test log and exp maps"""
    n_matrices, n_channels = 6, 3
    covmats = get_covmats(n_matrices, n_channels)
    Xt = fun_map(covmats, np.eye(n_channels))
    assert Xt.shape == (n_matrices, n_channels, n_channels)

    n_sets = 2
    covmats_4d = np.asarray([covmats for _ in range(n_sets)])
    Xt = fun_map(covmats_4d, np.eye(n_channels))
    assert Xt.shape == (n_sets, n_matrices, n_channels, n_channels)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_tangent_space_shape(metric, get_covmats):
    """Test tangent space projection"""
    n_matrices, n_channels = 6, 3
    n_ts = (n_channels * (n_channels + 1)) // 2
    covmats = get_covmats(n_matrices, n_channels)
    Xts = tangent_space(covmats, np.eye(n_channels), metric=metric)
    assert Xts.shape == (n_matrices, n_ts)

    n_sets = 2
    covmats_4d = np.asarray([covmats for _ in range(n_sets)])
    Xts = tangent_space(covmats_4d, np.eye(n_channels), metric=metric)
    assert Xts.shape == (n_sets, n_matrices, n_ts)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_untangent_space_shape(metric, rndstate):
    """Test untangent space projection"""
    n_matrices, n_channels = 10, 3
    n_ts = (n_channels * (n_channels + 1)) // 2
    T = rndstate.randn(n_matrices, n_ts)
    covmats = untangent_space(T, np.eye(n_channels), metric=metric)
    assert covmats.shape == (n_matrices, n_channels, n_channels)

    n_sets = 2
    T_4d = np.asarray([T for _ in range(n_sets)])
    covmats = untangent_space(T_4d, np.eye(n_channels), metric=metric)
    assert covmats.shape == (n_sets, n_matrices, n_channels, n_channels)


@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
def test_tangent_and_untangent_space(metric, get_covmats):
    """Test tangent space projection and retro-projection should be the same"""
    def tangent_untangent_space(X, n, metric):
        return untangent_space(tangent_space(X, np.eye(n), metric=metric),
                               np.eye(n), metric=metric)
    n_matrices, n_channels = 10, 3
    covmats = get_covmats(n_matrices, n_channels)
    covmats_ut = tangent_untangent_space(covmats, n_channels, metric)
    assert covmats_ut == approx(covmats)

    n_sets = 2
    covmats_4d = np.asarray([covmats for _ in range(n_sets)])
    covmats_4d_ut = tangent_untangent_space(covmats_4d, n_channels, metric)
    assert covmats_4d_ut == approx(covmats_4d)


@pytest.mark.parametrize("metric", get_metrics())
def test_transport(metric, get_covmats):
    n_matrices, n_channels = 10, 3
    covmats = get_covmats(n_matrices, n_channels)
    ref = np.eye(n_channels)
    covtr = transport(covmats, ref, metric=metric)
    assert covtr.shape == (n_matrices, n_channels, n_channels)
