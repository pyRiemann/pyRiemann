from conftest import get_metrics
import numpy as np
from pyriemann.embedding import (SpectralEmbedding,
                                 LocallyLinearEmbedding,
                                 barycenter_weights,
                                 riemann_lle)
import pytest


@pytest.mark.parametrize("metric", get_metrics())
@pytest.mark.parametrize("eps", [None, 0.1])
def test_spectral_embedding(metric, eps, get_covmats):
    """Test SpectralEmbedding."""
    n_matrices, n_channels, n_comp = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    embd = SpectralEmbedding(metric=metric, n_components=n_comp, eps=eps)
    covembd = embd.fit_transform(covmats)
    assert covembd.shape == (n_matrices, n_comp)


def test_spectral_embedding_fit_independence(get_covmats):
    n_matrices, n_channels = 6, 3
    covmats = get_covmats(n_matrices, n_channels)
    embd = SpectralEmbedding()
    embd.fit_transform(covmats)
    # retraining with different size should erase previous fit
    new_covmats = covmats[:, :-1, :-1]
    embd.fit_transform(new_covmats)


def test_rlle_embedding(get_covmats):
    """Test LocallyLinearEmbedding embedding fit_transform."""
    n_matrices, n_channels, n_comp = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    embd = LocallyLinearEmbedding(n_components=n_comp)
    covembd = embd.fit_transform(covmats)
    assert covembd.shape == (n_matrices, n_comp)


def test_rlle_transform(get_covmats):
    """Test LocallyLinearEmbedding embedding transform."""
    n_matrices, n_channels, n_comp = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    embd = LocallyLinearEmbedding(n_components=n_comp)

    embd.fit(covmats)
    new_covmats = get_covmats(n_matrices + 2, n_channels)
    covembd = embd.transform(new_covmats)
    assert covembd.shape == (n_matrices + 2, n_comp)


def test_barycenter_weights(get_covmats):
    """Test barycenter_weights helper function."""
    n_matrices, n_channels = 4, 3
    covmats = get_covmats(n_matrices, n_channels)
    weights = barycenter_weights(covmats, covmats, np.array([[1, 2], [2, 3],
                                                             [3, 0], [0, 1]]))
    assert weights.shape == (n_matrices, 2)


def test_riemann_lle(get_covmats):
    """Test riemann_lle helper function."""
    n_matrices, n_channels, n_comps, n_neighbors = 4, 3, 2, 2
    covmats = get_covmats(n_matrices, n_channels)
    embedding, error = riemann_lle(covmats,
                                   n_neighbors=n_neighbors,
                                   n_components=n_comps)
    assert embedding.shape == (n_matrices, n_comps)
    assert isinstance(error, float)
