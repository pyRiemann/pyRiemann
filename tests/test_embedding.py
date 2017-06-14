import numpy as np

from pyriemann.embedding import Embedding
from numpy.testing import assert_array_equal


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

def test_embedding():
    """Test Embedding."""
    covmats, diags, A = generate_cov(100, 3)
    embd = Embedding(metric='riemann', n_components=2).fit_transform(covmats)
    assert_array_equal(embd.shape[1], 2)
