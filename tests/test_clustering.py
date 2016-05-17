import numpy as np
from pyriemann.clustering import Kmeans, KmeansPerClassTransform, Potato


def generate_cov(Nt, Ne, s=0.1):
    """Generate a set of covariances matrices for test purpose"""
    np.random.seed(0)
    diags = 1.0+s*np.random.randn(Nt, Ne)
    covmats = np.empty((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.diag(diags[i])
    return covmats


def test_Kmeans_init():
    """Test init of Kmeans"""
    km = Kmeans(2)


def test_Kmeans_fit():
    """Test Fit of Kmeans"""
    covset = generate_cov(20, 3)
    km = Kmeans(2)
    km.fit(covset)


def test_Kmeans_fit_with_init():
    """Test Fit of Kmeans wit matric initialization"""
    covset = generate_cov(20, 3)
    km = Kmeans(2, init=covset[0:2])
    km.fit(covset)


def test_Kmeans_fit_with_y():
    """Test Fit of Kmeans with a given y"""
    covset = generate_cov(20, 3)
    labels = np.array([0, 1]).repeat(10)
    km = Kmeans(2)
    km.fit(covset, y=labels)


def test_Kmeans_fit_parallel():
    """Test Fit of Kmeans using paralell"""
    covset = generate_cov(20, 3)
    km = Kmeans(2, n_jobs=2)
    km.fit(covset)


def test_Kmeans_predict():
    """Test prediction of Kmeans"""
    covset = generate_cov(20, 3)
    km = Kmeans(2)
    km.fit(covset)
    km.predict(covset)


def test_Kmeans_transform():
    """Test transform of Kmeans"""
    covset = generate_cov(20, 3)
    km = Kmeans(2)
    km.fit(covset)
    km.transform(covset)


def test_KmeansPCT_init():
    """Test init of Kmeans PCT"""
    km = KmeansPerClassTransform(2)


def test_KmeansPCT_fit():
    """Test Fit of Kmeans PCT"""
    covset = generate_cov(20, 3)
    labels = np.array([0, 1]).repeat(10)
    km = KmeansPerClassTransform(2)
    km.fit(covset, labels)


def test_KmeansPCT_transform():
    """Test Transform of Kmeans PCT"""
    covset = generate_cov(20, 3)
    labels = np.array([0, 1]).repeat(10)
    km = KmeansPerClassTransform(2)
    km.fit(covset, labels)
    km.transform(covset)


def test_Potato_transform():
    """Test transform of Riemannian Potato"""
    covset = generate_cov(20, 3)
    rp = Potato()
    rp.fit(covset)
    rp.transform(covset)

    covset = generate_cov(20, 3)
    rp = Potato(metric='logeuclid', threshold=1, n_iter_max=50)
    rp.fit(covset)
    rp.transform(covset)


def test_Potato_predict():
    """Test transform of Riemannian Potato"""
    covset = generate_cov(20, 3)
    rp = Potato()
    rp.fit(covset, y=None)
    rp.predict(covset)
