import numpy as np
from nose.tools import assert_raises
from pyriemann.classification import MDM,FgMDM

def generate_cov(Nt,Ne):
    """Generate a set of cavariances matrices for test purpose"""
    diags = 1.0+0.1*np.random.randn(Nt,Ne)
    covmats = np.empty((Nt,Ne,Ne))
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

def test_MDM_fit_predict():
    """Test Fit & predict of MDM"""
    covset = generate_cov(100,3)
    labels = np.array([0,1]).repeat(50)
    mdm = MDM(metric='riemann')
    mdm.fit_predict(covset,labels)

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

def test_FgMDM_fit():
    """Test Fit of FgMDM"""
    covset = generate_cov(100,3)
    labels = np.array([0,1]).repeat(50)
    mdm = FgMDM(metric='riemann')
    mdm.fit(covset,labels)

def test_FgMDM_predict():
    """Test prediction of FgMDM"""
    covset = generate_cov(100,3)
    labels = np.array([0,1]).repeat(50)
    mdm = FgMDM(metric='riemann')
    mdm.fit(covset,labels)
    mdm.predict(covset)
