from conftest import get_covmats, get_metrics
import numpy as np
from numpy.testing import assert_array_equal
from pyriemann.embedding import Embedding
import pytest


@pytest.mark.parametrize("metric", get_metrics())
@pytest.mark.parametrize("eps", [None, 0.1])
def test_embedding(metric, eps, get_covmats):
    """Test Embedding."""
    n_trials, n_channels, n_comp = 6, 3, 2
    covmats = get_covmats(n_trials, n_channels)
    embd = Embedding(metric=metric, n_components=n_comp, eps=eps)
    covembd = embd.fit_transform(covmats)
    assert covembd.shape == (n_trials, n_comp)
    # assert_array_equal(embd.shape[1, n_components)
