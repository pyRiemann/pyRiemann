import numpy as np
from pyriemann.spatialfilters import Xdawn, CSP


def test_Xdawn_init():
    """Test init of Xdawn"""
    xd = Xdawn()


def test_Xdawn_fit():
    """Test Fit of Xdawn"""
    X = np.random.randn(100, 3, 10)
    labels = np.array([0, 1]).repeat(50)
    xd = Xdawn()
    xd.fit(X, labels)


def test_Xdawn_transform():
    """Test transform of Xdawn"""
    X = np.random.randn(100, 3, 10)
    labels = np.array([0, 1]).repeat(50)
    xd = Xdawn()
    xd.fit(X, labels)
    xd.transform(X)


def test_CSP():
    """Test methods of CSP"""
    csp = CSP()
    X = np.random.randn(100, 3, 10)
    labels = np.array([0, 1]).repeat(50)

def test_CSP_multiclass():
    """Test multiclass CSP"""
    csp = CSP()
    X = np.random.randn(100, 3, 10)
    labels = np.array([0, 1, 2, 3]).repeat(25)
    
    
