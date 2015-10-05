"""Test for channel selection."""
import numpy as np
from pyriemann.channelselection import ElectrodeSelection


def generate_cov(Nt, Ne):
    """Generate a set of cavariances matrices for test purpose."""
    diags = 1.0+0.1*np.random.randn(Nt, Ne)
    covmats = np.empty((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.diag(diags[i])
    return covmats


def test_ElectrodeSelection_transform():
    """Test transform of channelselection."""
    covset = generate_cov(10, 30)
    labels = np.array([0, 1]).repeat(5)
    se = ElectrodeSelection()
    se.fit(covset, labels)
    se.transform(covset)
