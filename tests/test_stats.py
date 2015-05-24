import numpy as np
from pyriemann.stats import PermutationTest, PermutationTestTwoWay

def generate_cov(Nt,Ne):
    """Generate a set of cavariances matrices for test purpose"""
    np.random.seed(123)
    diags = 1.0+0.1*np.random.randn(Nt,Ne)
    covmats = np.empty((Nt,Ne,Ne))
    for i in range(Nt):
        covmats[i] = np.diag(diags[i])
    return covmats
    
def test_permutation_test():
    """Test one way permutation test"""
    covset = generate_cov(10,30)
    labels = np.array([0,1]).repeat(5)
    p = PermutationTest(10)
    p.test(covset,labels)
    p.summary()
    p.plot(nbins=2)
    
def test_permutation2way_test():
    """Test two way permutation test"""
    covset = generate_cov(40,2)
    labels = np.array([0,1]).repeat(20)
    labels2 = np.array([4,5,2,3]).repeat(10)
    p = PermutationTestTwoWay(200)
    p.test(covset,labels2,labels)
    p.summary()
    p.plot(nbins=2)