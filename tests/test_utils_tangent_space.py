from conftest import get_metrics
import numpy as np
from pyriemann.utils.tangentspace import (
    tangent_space, untangent_space, transport
)
import pytest
from pytest import approx


def test_tangent_space(get_covmats):
    """Test tangent space projection"""
    n_trials, n_channels = 6, 3
    n_ts = (n_channels * (n_channels + 1)) // 2
    covmats = get_covmats(n_trials, n_channels)
    Xts = tangent_space(covmats, np.eye(n_channels))
    assert Xts.shape == (n_trials, n_ts)


def test_untangent_space(rndstate):
    """Test untangent space projection"""
    n_trials, n_channels = 10, 3
    n_ts = (n_channels * (n_channels + 1)) // 2
    T = rndstate.randn(n_trials, n_ts)
    covmats = untangent_space(T, np.eye(n_channels))
    assert covmats.shape == (n_trials, n_channels, n_channels)


def test_tangent_and_untangent_space(get_covmats):
    """Test tangent space projection and retro-projection should be the same"""
    n_trials, n_channels = 10, 3
    covmats = get_covmats(n_trials, n_channels)
    Xts = tangent_space(covmats, np.eye(n_channels))
    covmats_ut = untangent_space(Xts, np.eye(n_channels))
    assert covmats_ut == approx(covmats)


@pytest.mark.parametrize("metric", get_metrics())
def test_transport(metric, get_covmats):
    n_trials, n_channels = 10, 3
    covmats = get_covmats(n_trials, n_channels)
    ref = np.eye(n_channels)
    covtr = transport(covmats, ref, metric=metric)
    assert covtr.shape == (n_trials, n_channels, n_channels)
