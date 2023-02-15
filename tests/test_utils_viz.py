from conftest import requires_matplotlib
import numpy as np
import pytest

from pyriemann.utils.viz import (
    plot_embedding,
    plot_cospectra,
    plot_waveforms
)


@requires_matplotlib
def test_embedding(get_covmats):
    """Test ."""
    n_matrices, n_channels = 5, 3
    covmats = get_covmats(n_matrices, n_channels)
    plot_embedding(covmats, y=None, metric="euclid")
    y = np.ones(covmats.shape[0])
    plot_embedding(covmats, y=y, metric="euclid")


@requires_matplotlib
def test_embedding_error_raise(get_covmats):
    """Test ValueError for unknown embedding type."""
    n_matrices, n_channels = 5, 3
    covmats = get_covmats(n_matrices, n_channels)
    with pytest.raises(ValueError):
        plot_embedding(covmats, y=None, metric="euclid", embd_type='foo')


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
