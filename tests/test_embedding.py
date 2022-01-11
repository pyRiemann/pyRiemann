from conftest import get_metrics
import numpy as np

from pyriemann.embedding import (SpectralEmbedding,
                                 LocallyLinearEmbedding,
                                 barycenter_weights,
                                 locally_linear_embedding)
import pytest

rembd = [SpectralEmbedding, LocallyLinearEmbedding]
n_comp = [2, 4, 100]


@pytest.mark.parametrize("embd", rembd)
class EmbeddingTestCase:
    def test_embedding_build(self, embd, get_covmats):
        n_matrices, n_channels, n_comp = 8, 3, 4
        covmats = get_covmats(n_matrices, n_channels)

        self.embd_fit(embd, covmats, n_comp)
        self.embd_fit_transform(embd, covmats, n_comp)
        self.embd_fit_independence(embd, covmats, n_comp)
        if 'transform' in embd.__dict__.keys():
            self.embd_transform(embd, covmats, n_comp)
            self.embd_transform_error(embd, covmats, n_comp)
        self.embd_metric_error(embd, covmats, n_comp)


class TestEmbedding(EmbeddingTestCase):
    def embd_fit(self, embedding, covmats, n_components):
        n_matrices, n_channels, n_channels = covmats.shape
        embd = embedding(n_components=n_components)
        embd.fit(covmats)
        assert embd.embedding_.shape == (n_matrices, n_components)

    def embd_fit_transform(self, embedding, covmats, n_components):
        n_matrices, n_channels, n_channels = covmats.shape
        embd = embedding(n_components=n_components)
        transformed = embd.fit_transform(covmats)
        assert transformed.shape == (n_matrices, n_components)

    def embd_transform(self, embedding, covmats, n_components):
        n_matrices, n_channels, n_channels = covmats.shape
        embd = embedding(n_components=n_components)
        embd = embd.fit(covmats)
        transformed = embd.transform(covmats[:-1])
        assert transformed.shape == (n_matrices - 1, n_components)

    def embd_transform_error(self, embedding, covmats, n_components):
        embd = embedding(n_components=n_components)
        embd = embd.fit(covmats)
        with pytest.raises(AssertionError):
            embd.transform(covmats[:-1, :-1, :-1])

    def embd_fit_independence(self, embedding, covmats, n_components):
        n_matrices, n_channels, n_channels = covmats.shape
        embd = embedding(n_components=n_components)
        embd = embd.fit(covmats)
        # retraining with different size should erase previous fit
        new_covmats = covmats[:-1, :-1, :-1]
        embd = embd.fit(new_covmats)
        assert embd.embedding_.shape == (n_matrices - 1, n_components)

    def embd_metric_error(self, embedding, covmats, n_components):
        with pytest.raises(KeyError):
            embd = embedding(n_components=n_components, metric='foooo')
            embd.fit(covmats)


@pytest.mark.parametrize("n_components", n_comp)
@pytest.mark.parametrize("embd", rembd)
def embd_n_comp(n_components, embd, get_covmats):
    n_matrices, n_channels = 8, 3
    covmats = get_covmats(n_matrices, n_channels)
    embd = embd(n_components=n_components)
    if n_matrices <= n_components:
        with pytest.raises(AssertionError):
            embd.fit(covmats)
    else:
        embd.fit(covmats)


@pytest.mark.parametrize("metric", get_metrics())
@pytest.mark.parametrize("eps", [None, 0.1])
def test_spectral_embedding_parameters(metric, eps, get_covmats):
    """Test SpectralEmbedding."""
    n_matrices, n_channels, n_comp = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)
    embd = SpectralEmbedding(metric=metric, n_components=n_comp, eps=eps)
    covembd = embd.fit_transform(covmats)
    assert covembd.shape == (n_matrices, n_comp)


@pytest.mark.parametrize("metric", ['riemann', 'euclid', 'logeuclid'])
@pytest.mark.parametrize("n_neighbors", [2, 4, 8, 16])
@pytest.mark.parametrize("reg", [1e-3, 0])
def test_locally_linear_parameters(metric, n_neighbors, reg, get_covmats):
    """Test SpectralEmbedding."""
    n_matrices, n_channels, n_comp = 6, 3, 2
    covmats = get_covmats(n_matrices, n_channels)

    if n_matrices <= n_neighbors:
        with pytest.raises(AssertionError):
            embd = LocallyLinearEmbedding(metric=metric,
                                          n_components=n_comp,
                                          n_neighbors=n_neighbors)
            embd.fit(covmats)
    else:
        embd = LocallyLinearEmbedding(metric=metric,
                                      n_components=n_comp,
                                      n_neighbors=n_neighbors)
        covembd = embd.fit_transform(covmats)
        assert covembd.shape == (n_matrices, n_comp)


def test_barycenter_weights(get_covmats):
    """Test barycenter_weights helper function."""
    n_matrices, n_channels = 4, 3
    covmats = get_covmats(n_matrices, n_channels)
    weights = barycenter_weights(covmats, covmats, np.array([[1, 2], [2, 3],
                                                             [3, 0], [0, 1]]))
    assert weights.shape == (n_matrices, 2)


def test_locally_linear_embedding(get_covmats):
    """Test locally_linear_embedding helper function."""
    n_matrices, n_channels, n_comps, n_neighbors = 4, 3, 2, 2
    covmats = get_covmats(n_matrices, n_channels)
    embedding, error = locally_linear_embedding(covmats,
                                                n_neighbors=n_neighbors,
                                                n_components=n_comps)
    assert embedding.shape == (n_matrices, n_comps)
    assert isinstance(error, float)
