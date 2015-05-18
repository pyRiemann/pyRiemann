import numpy as np
from pyriemann.clustering import Kmeans,KmeansPerClassTransform

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
    
def test_Kmeans_fit_with_init():
    """Test Fit of Kmeans wit matric initialization"""
    covset = generate_cov(20,3)
    km = Kmeans(2,init=covset[0:2])
    km.fit(covset)

def test_Kmeans_fit_with_y():
    """Test Fit of Kmeans with a given y"""
    covset = generate_cov(20,3)
    labels = np.array([0,1]).repeat(10)
    km = Kmeans(2)
    km.fit(covset,y=labels)

def test_Kmeans_fit_parallel():
    """Test Fit of Kmeans using paralell"""
    covset = generate_cov(20,3)
    km = Kmeans(2,n_jobs=2)
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
    
def test_KmeansPCT_init():
    """Test init of Kmeans PCT"""
    km = KmeansPerClassTransform(2)
    
def test_KmeansPCT_fit():
    """Test Fit of Kmeans PCT"""
    covset = generate_cov(20,3)
    labels = np.array([0,1]).repeat(10)
    km = KmeansPerClassTransform(2)
    km.fit(covset,labels)
    
def test_KmeansPCT_transform():
    """Test Transform of Kmeans PCT"""
    covset = generate_cov(20,3)
    labels = np.array([0,1]).repeat(10)
    km = KmeansPerClassTransform(2)
    km.fit(covset,labels)
    km.transform(covset)