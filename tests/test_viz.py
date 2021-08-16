import pytest
import numpy as np
from conftest import covmats

try:
    from pyriemann.utils.viz import (
        plot_confusion_matrix,
        plot_embedding,
        plot_cospectra,
    )
except ImportError:
    no_plot = True


@pytest.mark.skipif("no_plot" in locals(), reason="Matplotlib not installed")
def test_embedding(covmats):
    """Test Embedding."""
    plot_embedding(covmats, y=None, metric="euclid")
    y = np.ones(covmats.shape[0])
    plot_embedding(covmats, y=y, metric="euclid")


@pytest.mark.skipif("no_plot" in locals(), reason="Matplotlib not installed")
def test_confusion_matrix():
    """Test confusion_matrix"""
    target = np.array([0, 1] * 10)
    preds = np.array([0, 1] * 10)
    plot_confusion_matrix(target, preds, ["a", "b"])


@pytest.mark.skipif("no_plot" in locals(), reason="Matplotlib not installed")
def test_cospectra():
    """Test plot_cospectra"""
    n_freqs, n_channels = 16, 3
    cosp = np.random.randn(n_freqs, n_channels, n_channels)
    freqs = np.random.randn(n_freqs)
    plot_cospectra(cosp, freqs)
