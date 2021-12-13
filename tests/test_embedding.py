from conftest import get_metrics
from pyriemann.embedding import SpectralEmbedding, UMAP, _umap_metric_helper
from pyriemann.utils.distance import _check_distance_method
import pytest


@pytest.mark.parametrize("metric", get_metrics())
@pytest.mark.parametrize("eps", [None, 0.1])
def test_spectral_embedding(metric, eps, get_covmats):
    """Test SpectralEmbedding."""
    n_trials, n_channels, n_comp = 6, 3, 2
    covmats = get_covmats(n_trials, n_channels)
    embd = SpectralEmbedding(metric=metric, n_components=n_comp, eps=eps)
    covembd = embd.fit_transform(covmats)
    assert covembd.shape == (n_trials, n_comp)


def test_spectral_embedding_fit_independence(get_covmats):
    n_trials, n_channels = 6, 3
    covmats = get_covmats(n_trials, n_channels)
    embd = SpectralEmbedding()
    embd.fit_transform(covmats)
    # retraining with different size should erase previous fit
    new_covmats = covmats[:, :-1, :-1]
    embd.fit_transform(new_covmats)


@pytest.mark.parametrize("metric", get_metrics())
def test_umap_embedding(metric, get_covmats):
    """Test SpectralEmbedding."""
    n_trials, n_channels, n_comp = 6, 3, 2
    covmats = get_covmats(n_trials, n_channels)
    embd = UMAP(distance_metric=metric, n_components=n_comp)
    covembd = embd.fit_transform(covmats)
    assert covembd.shape == (n_trials, n_comp)


def test_umap_setattr(get_covmats):
    n_trials, n_channels = 6, 3
    covmats = get_covmats(n_trials, n_channels)
    embd = UMAP(n_components=2)
    embd.fit_transform(covmats)
    n_comp = 3
    embd.n_components = n_comp
    # retraining with different size should erase previous fit
    new_covmats = covmats[:-1, :-1, :-1]
    covembd = embd.fit_transform(new_covmats)
    assert covembd.shape == (n_trials - 1, n_comp)


def test_umap_transform(get_covmats):
    n_trials, n_channels = 6, 3
    covmats = get_covmats(n_trials, n_channels)
    n_comp = 2
    embd = UMAP(n_components=n_comp)
    embd.fit(covmats)
    new_covmats = get_covmats(n_trials + 2, n_channels)
    covembd = embd.transform(new_covmats)
    assert covembd.shape == (n_trials + 2, n_comp)


def test_umap_invalid_metric(get_covmats):
    n_comp = 2
    with pytest.raises(ValueError):
        UMAP(n_components=n_comp, distance_metric='shenanigans')


@pytest.mark.parametrize("metric", get_metrics())
def test_umap_metric_helper(get_covmats, metric):
    dist = _check_distance_method(metric)
    n_trials, n_channels = 2, 3
    A, B = get_covmats(n_trials, n_channels)
    Af, Bf = A.flatten(), B.flatten()
    assert dist(A, B) == _umap_metric_helper(Af, Bf, dist)
