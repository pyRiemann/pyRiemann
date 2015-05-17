from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal
from nose.tools import assert_true, assert_raises, assert_not_equal
import numpy as np

from pyriemann.utils.geodesic import (geodesic_riemann,geodesic_euclid,geodesic_logeuclid,geodesic)

#### Riemannian metric 
def test_geodesic_riemann_0():
    """Test riemannian geodesic when alpha = 0"""
    A = 0.5*np.eye(3)
    B = 2*np.eye(3)
    assert_array_almost_equal(geodesic_riemann(A,B,0),A)
    
def test_geodesic_riemann_1():
    """TTest riemannian geodesic when alpha = 1"""
    A = 0.5*np.eye(3)
    B = 2*np.eye(3)
    assert_array_almost_equal(geodesic_riemann(A,B,1),B)
    
def test_geodesic_riemann_middle():
    """Test riemannian geodesic when alpha = 0.5"""
    A = 0.5*np.eye(3)
    B = 2*np.eye(3)
    Ctrue = np.eye(3)
    assert_array_almost_equal(geodesic_riemann(A,B,0.5),Ctrue)
    
#### euclidean metric 
def test_geodesic_euclid_0():
    """Test euclidean geodesic when alpha = 0"""
    A = 0.5*np.eye(3)
    B = 2*np.eye(3)
    assert_array_almost_equal(geodesic_euclid(A,B,0),A)
    
def test_geodesic_euclid_1():
    """TTest euclidean geodesic when alpha = 1"""
    A = 0.5*np.eye(3)
    B = 2*np.eye(3)
    assert_array_almost_equal(geodesic_euclid(A,B,1),B)
    
def test_geodesic_euclid_middle():
    """Test euclidean geodesic when alpha = 0.5"""
    A = 1*np.eye(3)
    B = 2*np.eye(3)
    Ctrue = 1.5*np.eye(3)
    assert_array_almost_equal(geodesic_euclid(A,B,0.5),Ctrue)
    
#### log-euclidean metric 
def test_geodesic_logeuclid_0():
    """Test log euclidean geodesic when alpha = 0"""
    A = 0.5*np.eye(3)
    B = 2*np.eye(3)
    assert_array_almost_equal(geodesic_logeuclid(A,B,0),A)
    
def test_geodesic_logeuclid_1():
    """TTest log euclidean geodesic when alpha = 1"""
    A = 0.5*np.eye(3)
    B = 2*np.eye(3)
    assert_array_almost_equal(geodesic_logeuclid(A,B,1),B)
    
def test_geodesic_logeuclid_middle():
    """Test log euclidean geodesic when alpha = 0.5"""
    A = 0.5*np.eye(3)
    B = 2*np.eye(3)
    Ctrue = 1*np.eye(3)
    assert_array_almost_equal(geodesic_logeuclid(A,B,0.5),Ctrue)
    
### global geodesic

def test_geodesic_riemann():
    """Test riemannian geodesic when alpha = 0.5 for global function"""
    A = 0.5*np.eye(3)
    B = 2*np.eye(3)
    Ctrue = 1*np.eye(3)
    assert_array_almost_equal(geodesic(A,B,0.5,metric='riemann'),Ctrue)
    
def test_geodesic_euclid():
    """Test euclidean geodesic when alpha = 0.5 for global function"""
    A = 1*np.eye(3)
    B = 2*np.eye(3)
    Ctrue = 1.5*np.eye(3)
    assert_array_almost_equal(geodesic(A,B,0.5,metric='euclid'),Ctrue)
    
def test_geodesic_logeuclid():
    """Test riemannian geodesic when alpha = 0.5 for global function"""
    A = 0.5*np.eye(3)
    B = 2*np.eye(3)
    Ctrue = 1*np.eye(3)
    assert_array_almost_equal(geodesic(A,B,0.5,metric='logeuclid'),Ctrue)

