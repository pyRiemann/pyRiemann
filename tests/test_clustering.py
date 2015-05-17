import numpy as np
from pyriemann.clustering import Kmeans

def generate_cov(Nt,Ne):
    """Generate a set of cavariances matrices for test purpose"""
    diags = 1.0+0.1*np.random.randn(Nt,Ne)
    covmats = np.empty((Nt,Ne,Ne))
    for i in range(Nt):
        covmats[i] = np.diag(diags[i])
    return covmats

def test_Kmeans_init():
    """Test init of Kmeans"""
    km = Kmeans(2)
    
def test_Kmeans_fit():
    """Test Fit of Kmeans"""
    covset = generate_cov(20,3)
    km = Kmeans(2)
    km.fit(covset)

def test_Kmeans_predict():
    """Test prediction of Kmeans"""
    covset = generate_cov(20,3)
    km = Kmeans(2)
    km.fit(covset)
    km.predict(covset)
    
def test_Kmeans_transform():
    """Test transform of Kmeans"""
    covset = generate_cov(20,3)
    km = Kmeans(2)
    km.fit(covset)
    km.transform(covset)