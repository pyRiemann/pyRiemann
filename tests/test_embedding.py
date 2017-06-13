import numpy as np

from pyriemann.embedding import Embedding
from numpy.testing import assert_array_equal, assert_almost_equal


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


def test_embedding_firstcomponent():
    """Test Embedding."""
    covmats, diags, A = generate_cov(100, 3)
    u, l = Embedding(metric='riemann').fit_transform(covmats)
    assert_almost_equal(l[0], 1.0)
    assert_array_equal(u[:, 0], np.ones(100))
