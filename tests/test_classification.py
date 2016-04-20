import numpy as np
from nose.tools import assert_raises
from pyriemann.classification import MDM, FgMDM

def generate_cov(Nt,Ne):
    """Generate a set of cavariances matrices for test purpose"""
    np.random.seed(1234)
    diags = 2. + 0.1 * np.random.randn(Nt, Ne)
    covmats = np.empty(shape=(Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.diag(diags[i])
    return covmats

def test_MDM_init():
    """Test init of MDM"""
    mdm = MDM(metric='riemann')

    # Should raise if metric not string or dict
    assert_raises(TypeError,MDM,metric=42)

    # Should raise if metric is not contain bad keys
    assert_raises(KeyError,MDM,metric={'universe' : 42})

    #should works with correct dict
    mdm = MDM(metric={'mean' : 'riemann', 'distance' : 'logeuclid'})

def test_MDM_fit():
    """Test Fit of MDM"""
    covset = generate_cov(100,3)
    labels = np.array([0,1]).repeat(50)
    mdm = MDM(metric='riemann')
    mdm.fit(covset,labels)

def test_MDM_predict():
    """Test prediction of MDM"""
    covset = generate_cov(100,3)
    labels = np.array([0,1]).repeat(50)
    mdm = MDM(metric='riemann')
    mdm.fit(covset,labels)
    mdm.predict(covset)
    mdm = MDM(n_jobs=2)
    mdm.fit(covset,labels)
    mdm.predict(covset)
    mdm.predict_proba(covset)

def test_MDM_fit_predict():
    """Test Fit & predict of MDM"""
    covset = generate_cov(100,3)
    labels = np.array([0,1]).repeat(50)
    mdm = MDM(metric='riemann')
    mdm.fit_predict(covset, labels)

def test_MDM_transform():
    """Test transform of MDM"""
    covset = generate_cov(100,3)
    labels = np.array([0,1]).repeat(50)
    mdm = MDM(metric='riemann')
    mdm.fit(covset,labels)
    mdm.transform(covset)


def test_FgMDM_init():
    """Test init of FgMDM"""
    mdm = FgMDM(metric='riemann')
    mdm = FgMDM(metric={'mean':'logeuclid', 'distance':'riemann'})

    assert_raises(KeyError, FgMDM, metric={'foo':'bar'})
    assert_raises(TypeError, FgMDM, metric=42)

def test_FgMDM_fit_transform():
    """Test fit and transform of FgMDM"""
    covset = generate_cov(100,3)
    labels = np.array([0,1]).repeat(50)
    mdm = FgMDM(metric='riemann')
    mdm.fit(covset,labels)
    mdm.transform(covset)

def test_FgMDM_predict():
    """Test prediction of FgMDM"""
    covset = generate_cov(100,3)
    labels = np.array([0,1]).repeat(50)
    mdm = FgMDM(metric='riemann')
    mdm.fit(covset,labels)
    mdm.predict(covset)


def test_TSc_init():
    """Test init of TSClassifier"""
    tsc = TSclassifier(metric='riemann')
    tsc = TSclassifier(tsupdate=True)
    assert_raises(TypeError, TSclassifier, clf=42)

def test_TSc_fit_transform():
    """Test fit and transform of TSClassifier"""
    covset = generate_cov(100,3)
    labels = np.array([0,1]).repeat(50)
    tsc = TSclassifier(metric='riemann')
    tsc.fit(covset,labels)
    tsc.transform(covset)

def test_TSc_predict():
    """Test prediction of TSClassifier"""
    covset = generate_cov(100,3)
    labels = np.array([0,1]).repeat(50)
    ts = TSclassifier(metric='riemann')
    tsc.fit(covset,labels)
    tsc.predict(covset)
    tsc.predict_proba(covset)


def test_kNN_init():
    """Test init of KNearestNeighbor"""
    knn = KNearestNeighbor(metric='riemann')

def test_kNN_fit_transform():
    """Test fit of KNearestNeighbor"""
    covset = generate_cov(100,3)
    labels = np.array([0,1]).repeat(50)
    knn = KNearestNeighbor(metric='riemann')
    knn.fit(covset,labels)

def test_kNN_predict():
    """Test prediction of KNearestNeighbor"""
    covset = generate_cov(100,3)
    labels = np.array([0,1]).repeat(50)
    ts = KNearestNeighbor(metric='riemann')
    knn.fit(covset,labels)
    knn.predict(covset)
