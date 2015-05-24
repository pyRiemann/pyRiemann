from numpy.testing import assert_array_equal
import numpy as np

from pyriemann.utils.tangentspace import tangent_space,untangent_space

def generate_cov(Nt,Ne):
    """Generate a set of cavariances matrices for test purpose"""
    diags = 1.0+0.1*np.random.randn(Nt,Ne)
    covmats = np.empty((Nt,Ne,Ne))
    for i in range(Nt):
        covmats[i] = np.diag(diags[i])
    return covmats
    
def test_tangent_space():
    """Test tangent space projection"""
    C = generate_cov(10,3)
    T = tangent_space(C,np.eye(3))
    
def test_untangent_space():
    """Test untangent space projection"""
    T = np.random.randn(10,6)
    covmats = untangent_space(T,np.eye(3))
    
def test_tangent_and_untangent_space():
    """Test tangent space projection and retro-projection should be the same"""
    C = generate_cov(10,3)
    T = tangent_space(C,np.eye(3))
    covmats = untangent_space(T,np.eye(3))
    assert_array_equal(C,covmats)