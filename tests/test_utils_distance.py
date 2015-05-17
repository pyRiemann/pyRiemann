from numpy.testing import assert_array_almost_equal,assert_array_equal
from nose.tools import assert_equal
import numpy as np

from pyriemann.utils.distance import   (distance_riemann,
                                        distance_euclid,
                                        distance_logeuclid,
                                        distance_logdet,
                                        distance)
                                        
                                        
def test_distance_riemann():
    """Test riemannian distance"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert_array_almost_equal(distance_riemann(A,B),0)
    
def test_distance_euclid():
    """Test euclidean distance"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert_equal(distance_euclid(A,B),0)
    
def test_distance_logeuclid():
    """Test logeuclid distance"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert_equal(distance_logeuclid(A,B),0)
    
def test_distance_logdet():
    """Test logdet distance"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert_equal(distance_logdet(A,B),0)
    
def test_distance_generic_riemann():
    """Test riemannian distance for generic function"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert_equal(distance(A,B,metric='riemann'),distance_riemann(A,B))
    
def test_distance_generic_euclid():
    """Test euclidean distance for generic function"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert_equal(distance(A,B,metric='euclid'),distance_euclid(A,B))
    
def test_distance_generic_logdet():
    """Test logdet distance for generic function"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert_equal(distance(A,B,metric='logdet'),distance_logdet(A,B))
    
def test_distance_generic_logeuclid():
    """Test logeuclid distance for generic function"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert_equal(distance(A,B,metric='logeuclid'),distance_logeuclid(A,B))
    