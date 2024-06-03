import numpy as np
import pytest

from conftest import requires_matplotlib
from pyriemann.utils.viz import (
    plot_embedding,
    plot_cospectra,
    plot_waveforms
)


@requires_matplotlib
def test_embedding(get_mats):
    """Test ."""
    n_matrices, n_channels = 5, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    plot_embedding(mats, y=None, metric="euclid")
    y = np.ones(mats.shape[0])
    plot_embedding(mats, y=y, metric="euclid")


@requires_matplotlib
def test_embedding_error_raise(get_mats):
    """Test ValueError for unknown embedding type."""
    n_matrices, n_channels = 5, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):
        plot_embedding(mats, y=None, metric="euclid", embd_type="foo")


@requires_matplotlib
def test_cospectra():
    """Test plot_cospectra"""
    n_freqs, n_channels = 16, 3
    cosp = np.random.randn(n_freqs, n_channels, n_channels)
    freqs = np.random.randn(n_freqs)
    plot_cospectra(cosp, freqs)


@requires_matplotlib
@pytest.mark.parametrize("display", ["all", "mean", "mean+/-std", "hist"])
def test_plot_waveforms(display):
    """Test plot_waveforms"""
    n_matrices, n_channels, n_times = 16, 3, 50
    X = np.random.randn(n_matrices, n_channels, n_times)
    plot_waveforms(X, display)
    plot_waveforms(X, display, times=np.arange(n_times))

    X = np.random.randn(n_matrices, 1, n_times)
    plot_waveforms(X, display)
