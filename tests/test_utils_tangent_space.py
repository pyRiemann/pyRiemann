from conftest import get_metrics
import numpy as np
from pyriemann.utils.tangentspace import (
    tangent_space, untangent_space, transport
)
import pytest
from pytest import approx


def test_tangent_space_shape(get_covmats):
    """Test tangent space projection"""
    n_matrices, n_channels = 6, 3
    n_ts = (n_channels * (n_channels + 1)) // 2
    covmats = get_covmats(n_matrices, n_channels)
    Xts = tangent_space(covmats, np.eye(n_channels))
    assert Xts.shape == (n_matrices, n_ts)

    n_sets = 2
    covmats_4d = np.asarray([covmats for _ in range(n_sets)])
    Xts = tangent_space(covmats_4d, np.eye(n_channels))
    assert Xts.shape == (n_sets, n_matrices, n_ts)


def test_untangent_space_shape(rndstate):
    """Test untangent space projection"""
    n_matrices, n_channels = 10, 3
    n_ts = (n_channels * (n_channels + 1)) // 2
    T = rndstate.randn(n_matrices, n_ts)
    covmats = untangent_space(T, np.eye(n_channels))
    assert covmats.shape == (n_matrices, n_channels, n_channels)

    n_sets = 2
    T_4d = np.asarray([T for _ in range(n_sets)])
    covmats = untangent_space(T_4d, np.eye(n_channels))
    assert covmats.shape == (n_sets, n_matrices, n_channels, n_channels)


def test_tangent_and_untangent_space(get_covmats):
    """Test tangent space projection and retro-projection should be the same"""
    def tangent_untangent_space(X, n):
        return untangent_space(tangent_space(X, np.eye(n)), np.eye(n))
    n_matrices, n_channels = 10, 3
    covmats = get_covmats(n_matrices, n_channels)
    covmats_ut = tangent_untangent_space(covmats, n_channels)
    assert covmats_ut == approx(covmats)

    n_sets = 2
    covmats_4d = np.asarray([covmats for _ in range(n_sets)])
    covmats_4d_ut = tangent_untangent_space(covmats_4d, n_channels)
    assert covmats_4d_ut == approx(covmats_4d)


@pytest.mark.parametrize("metric", get_metrics())
def test_transport(metric, get_covmats):
    n_matrices, n_channels = 10, 3
    covmats = get_covmats(n_matrices, n_channels)
    ref = np.eye(n_channels)
    covtr = transport(covmats, ref, metric=metric)
    assert covtr.shape == (n_matrices, n_channels, n_channels)
