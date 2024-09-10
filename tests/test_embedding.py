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
from pyriemann.utils.kernel import kernel, kernel_functions

rembd = [SpectralEmbedding, LocallyLinearEmbedding]


@pytest.mark.parametrize("embd", rembd)
def test_embedding(embd, get_mats):
    n_matrices, n_channels, n_comp = 8, 3, 4
    mats = get_mats(n_matrices, n_channels, "spd")

    embd_fit(embd, mats, n_comp)
    embd_fit_transform(embd, mats, n_comp)
    embd_fit_independence(embd, mats, n_comp)
    if "transform" in embd.__dict__.keys():
        embd_transform(embd, mats, n_comp)
        embd_transform_error(embd, mats, n_comp)
    embd_metric_error(embd, mats, n_comp)
    embd_result(embd)


def embd_fit(embedding, mats, n_components):
    n_matrices, n_channels, n_channels = mats.shape
    embd = embedding(n_components=n_components)
    embd.fit(mats)
    assert embd.embedding_.shape == (n_matrices, n_components)

    if embedding is LocallyLinearEmbedding:
        assert embd.data_.shape == (n_matrices, n_channels, n_channels)
        assert isinstance(embd.error_, float)


def embd_fit_transform(embedding, mats, n_components):
    n_matrices, n_channels, n_channels = mats.shape
    embd = embedding(n_components=n_components)
    transformed = embd.fit_transform(mats)
    assert transformed.shape == (n_matrices, n_components)


def embd_transform(embedding, mats, n_components):
    n_matrices, n_channels, n_channels = mats.shape
    embd = embedding(n_components=n_components)
    embd = embd.fit(mats)
    transformed = embd.transform(mats[:-1])
    assert transformed.shape == (n_matrices - 1, n_components)


def embd_transform_error(embedding, mats, n_components):
    embd = embedding(n_components=n_components)
    embd = embd.fit(mats)
    with pytest.raises(ValueError):
        embd.transform(mats[:-1, :-1, :-1])


def embd_fit_independence(embedding, mats, n_components):
    n_matrices, n_channels, n_channels = mats.shape
    embd = embedding(n_components=n_components)
    embd = embd.fit(mats)
    # retraining with different size should erase previous fit
    new_mats = mats[:-1, :-1, :-1]
    embd = embd.fit(new_mats)
    assert embd.embedding_.shape == (n_matrices - 1, n_components)


def embd_metric_error(embedding, mats, n_components):
    with pytest.raises(ValueError):
        embd = embedding(n_components=n_components, metric="foooo")
        embd.fit(mats)


def embd_result(embedding):
    X, y = make_gaussian_blobs(n_matrices=10,
                               n_dim=2,
                               class_sep=10.,
                               class_disp=1)

    embd = embedding(n_components=1)
    X_ = embd.fit_transform(X)
    score = NearestCentroid().fit(X_, y).score(X_, y)
    assert score == 1.


@pytest.mark.parametrize("metric", get_metrics())
@pytest.mark.parametrize("eps", [None, 0.1])
def test_spectral_embedding_parameters(metric, eps, get_mats):
    """Test SpectralEmbedding."""
    n_matrices, n_channels, n_comp = 6, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    embd = SpectralEmbedding(metric=metric, n_components=n_comp, eps=eps)
    covembd = embd.fit_transform(mats)
    assert covembd.shape == (n_matrices, n_comp)


@pytest.mark.parametrize("n_components", [2, 4, 100])
@pytest.mark.parametrize("embd", rembd)
def test_embd_n_comp(n_components, embd, get_mats):
    n_matrices, n_channels = 8, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    embd = embd(n_components=n_components)
    if n_matrices <= n_components:
        with pytest.raises(ValueError):
            embd.fit(mats)
    else:
        embd.fit(mats)


@pytest.mark.parametrize("metric", ["euclid", "logcholesky",
                                    "logeuclid", "riemann"])
@pytest.mark.parametrize("n_neighbors", [2, 4, 8, 16])
@pytest.mark.parametrize("reg", [1e-3, 0])
@pytest.mark.parametrize("kernel_fct", [kernel, None])
def test_locally_linear_parameters(metric, n_neighbors, reg, kernel_fct,
                                   get_mats):
    """Test LocallyLinearEmbedding."""
    n_matrices, n_channels, n_components = n_neighbors + 5, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    embd = LocallyLinearEmbedding(metric=metric,
                                  n_components=n_components,
                                  n_neighbors=n_neighbors,
                                  kernel=kernel_fct)
    covembd = embd.fit_transform(mats)
    assert covembd.shape == (n_matrices, n_components)


def test_barycenter_weights(get_mats):
    """Test barycenter_weights helper function."""
    n_matrices, n_channels = 4, 3
    mats = get_mats(n_matrices, n_channels, "spd")
    weights = barycenter_weights(mats, mats, np.array([[1, 2], [2, 3],
                                                       [3, 0], [0, 1]]))
    assert weights.shape == (n_matrices, 2)


def test_locally_linear_embedding(get_mats):
    """Test locally_linear_embedding helper function."""
    n_matrices, n_channels, n_comps, n_neighbors = 4, 3, 2, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    embedding, error = locally_linear_embedding(mats,
                                                n_neighbors=n_neighbors,
                                                n_components=n_comps)
    assert embedding.shape == (n_matrices, n_comps)
    assert isinstance(error, float)


@pytest.mark.parametrize("metric", ["euclid", "logcholesky",
                                    "logeuclid", "riemann"])
def test_locally_linear_none_kernel(metric, get_mats):
    """Test LocallyLinearEmbedding."""
    n_matrices, n_channels, n_components = 6, 3, 2
    mats = get_mats(n_matrices, n_channels, "spd")

    def kfun(X, Y=None, Cref=None, metric=None):
        return kernel_functions[metric](X, Y, Cref=Cref)

    embd = LocallyLinearEmbedding(metric=metric,
                                  n_components=n_components,
                                  kernel=kfun)
    mats_tf = embd.fit_transform(mats)

    embd2 = LocallyLinearEmbedding(metric=metric,
                                   n_components=n_components,
                                   kernel=None)
    mats_tf2 = embd2.fit_transform(mats)

    assert np.array_equal(mats_tf, mats_tf2)
