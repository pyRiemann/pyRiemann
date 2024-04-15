import numpy as np
import pytest
from sklearn.neighbors import NearestCentroid

from conftest import get_metrics
from pyriemann.datasets import make_gaussian_blobs
from pyriemann.embedding import (
    SpectralEmbedding,
    LocallyLinearEmbedding,
    barycenter_weights,
    locally_linear_embedding,
)

from pyriemann.utils.kernel import (kernel,
                                    kernel_euclid, # noqa
                                    kernel_logeuclid, # noqa
                                    kernel_riemann) # noqa

rembd = [SpectralEmbedding, LocallyLinearEmbedding]


@pytest.mark.parametrize("embd", rembd)
class EmbeddingTestCase:
    def test_embedding_build(self, embd, get_mats):
        n_matrices, n_channels, n_comp = 8, 3, 4
        covmats = get_mats(n_matrices, n_channels, "spd")

        self.embd_fit(embd, covmats, n_comp)
        self.embd_fit_transform(embd, covmats, n_comp)
        self.embd_fit_independence(embd, covmats, n_comp)
        if 'transform' in embd.__dict__.keys():
            self.embd_transform(embd, covmats, n_comp)
            self.embd_transform_error(embd, covmats, n_comp)
        self.embd_metric_error(embd, covmats, n_comp)
        self.embd_result(embd)


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
        with pytest.raises(ValueError):
            embd = embedding(n_components=n_components, metric='foooo')
            embd.fit(covmats)

    def embd_result(self, embedding):
        X, y = make_gaussian_blobs(n_matrices=10,
                                   n_dim=2,
                                   class_sep=10.,
                                   class_disp=1)

        embd = embedding(n_components=1)
        X_ = embd.fit_transform(X)
        score = NearestCentroid().fit(X_, y).score(X_, y)
        assert score == 1.


@pytest.mark.parametrize("n_components", [2, 4, 100])
@pytest.mark.parametrize("embd", rembd)
def embd_n_comp(n_components, embd, get_mats):
    n_matrices, n_channels = 8, 3
    covmats = get_mats(n_matrices, n_channels, "spd")
    embd = embd(n_components=n_components)
    if n_matrices <= n_components:
        with pytest.raises(AssertionError):
            embd.fit(covmats)
    else:
        embd.fit(covmats)


@pytest.mark.parametrize("metric", get_metrics())
@pytest.mark.parametrize("eps", [None, 0.1])
def test_spectral_embedding_parameters(metric, eps, get_mats):
    """Test SpectralEmbedding."""
    n_matrices, n_channels, n_comp = 6, 3, 2
    covmats = get_mats(n_matrices, n_channels, "spd")
    embd = SpectralEmbedding(metric=metric, n_components=n_comp, eps=eps)
    covembd = embd.fit_transform(covmats)
    assert covembd.shape == (n_matrices, n_comp)


@pytest.mark.parametrize("metric", ['riemann', 'euclid', 'logeuclid'])
@pytest.mark.parametrize("n_neighbors", [2, 4, 8, 16])
@pytest.mark.parametrize("reg", [1e-3, 0])
@pytest.mark.parametrize("kernel_fct", [kernel, None])
def test_locally_linear_parameters(metric, n_neighbors, reg, kernel_fct,
                                   get_mats):
    """Test LocallyLinearEmbedding."""
    n_matrices, n_channels, n_components = 6, 3, 2
    covmats = get_mats(n_matrices, n_channels, "spd")
    embd = LocallyLinearEmbedding(metric=metric,
                                  n_components=n_components,
                                  n_neighbors=n_neighbors,
                                  kernel=kernel_fct)
    covembd = embd.fit_transform(covmats)
    assert covembd.shape == (n_matrices, n_components)


def test_barycenter_weights(get_mats):
    """Test barycenter_weights helper function."""
    n_matrices, n_channels = 4, 3
    covmats = get_mats(n_matrices, n_channels, "spd")
    weights = barycenter_weights(covmats, covmats, np.array([[1, 2], [2, 3],
                                                             [3, 0], [0, 1]]))
    assert weights.shape == (n_matrices, 2)


def test_locally_linear_embedding(get_mats):
    """Test locally_linear_embedding helper function."""
    n_matrices, n_channels, n_comps, n_neighbors = 4, 3, 2, 2
    covmats = get_mats(n_matrices, n_channels, "spd")
    embedding, error = locally_linear_embedding(covmats,
                                                n_neighbors=n_neighbors,
                                                n_components=n_comps)
    assert embedding.shape == (n_matrices, n_comps)
    assert isinstance(error, float)


@pytest.mark.parametrize("metric", ['riemann', 'euclid', 'logeuclid'])
def test_locally_linear_none_kernel(metric, get_mats):
    """Test LocallyLinearEmbedding."""
    n_matrices, n_channels, n_components = 6, 3, 2
    covmats = get_mats(n_matrices, n_channels, "spd")
    kernel_fun = globals()[f'kernel_{metric}']

    def kfun(X, Y=None, Cref=None, metric=None):
        return kernel_fun(X, Y, Cref=Cref)

    embd = LocallyLinearEmbedding(metric=metric,
                                  n_components=n_components,
                                  kernel=kfun)
    covembd = embd.fit_transform(covmats)

    embd2 = LocallyLinearEmbedding(metric=metric,
                                   n_components=n_components,
                                   kernel=None)
    covembd2 = embd2.fit_transform(covmats)

    assert np.array_equal(covembd, covembd2)
