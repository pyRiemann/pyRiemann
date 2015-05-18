import numpy as np
from pyriemann.spatialfilters import Xdawn

def test_Xdawn_init():
    """Test init of Xdawn"""
    xd = Xdawn()
    
def test_Xdawn_fit():
    """Test Fit of Xdawn"""
    x = np.random.randn(100,3,10)
    labels = np.array([0,1]).repeat(50)
    xd = Xdawn()
    xd.fit(x,labels)

def test_Xdawn_transform():
    """Test transform of Xdawn"""
    x = np.random.randn(100,3,10)
    labels = np.array([0,1]).repeat(50)
    xd = Xdawn()
    xd.fit(x,labels)
    xd.transform(x)
