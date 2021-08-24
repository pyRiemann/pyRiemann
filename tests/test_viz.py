from conftest import get_covmats, requires_matplotlib
import numpy as np
from pyriemann.utils.viz import (plot_confusion_matrix, plot_embedding,
                                 plot_cospectra)


def generate_cov(Nt, Ne):
    """Generate a set of cavariances matrices for test purpose."""
    rs = np.random.RandomState(1234)
    diags = 2.0 + 0.1 * rs.randn(Nt, Ne)
    A = 2*rs.rand(Ne, Ne) - 1
    A /= np.atleast_2d(np.sqrt(np.sum(A**2, 1))).T
    covmats = np.empty((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.dot(np.dot(A, np.diag(diags[i])), A.T)
    return covmats, diags, A

@requires_matplotlib
def test_embedding(get_covmats):
    """Test Embedding."""
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)
    plot_embedding(covmats, y=None, metric="euclid")
    y = np.ones(covmats.shape[0])
    plot_embedding(covmats, y=y, metric='euclid')


def test_confusion_matrix():
    """Test confusion_matrix"""
    target = np.array([0, 1] * 10)
    preds = np.array([0, 1] * 10)
    with pytest.warns(DeprecationWarning, match="plot_confusion_matrix is deprecated"):
        plot_confusion_matrix(target, preds, ["a", "b"])


def test_cospectra():
    """Test plot_cospectra"""
    n_freqs, n_channels = 16, 3
    cosp = np.random.randn(n_freqs, n_channels, n_channels)
    freqs = np.random.randn(n_freqs)
    plot_cospectra(cosp, freqs)
