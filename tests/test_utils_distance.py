from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
import numpy as np

from pyriemann.utils.distance import (distance_riemann,
                                      distance_euclid,
                                      distance_logeuclid,
                                      distance_logdet,
                                      distance_kullback,
                                      distance_kullback_right,
                                      distance_kullback_sym,
                                      distance_wasserstein,
                                      distance, pairwise_distance,
                                      _check_distance_method)


def test_check_distance_methd():
    """Test _check_distance_method"""
    _check_distance_method('riemann')
    _check_distance_method(distance_riemann)
    with pytest.raises(ValueError):
        _check_distance_method('666')
    with pytest.raises(ValueError):
        _check_distance_method(42)


def test_distance_riemann():
    """Test riemannian distance"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert_array_almost_equal(distance_riemann(A, B), 0)


def test_distance_kullback():
    """Test kullback divergence"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert_array_almost_equal(distance_kullback(A, B), 0)
    assert_array_almost_equal(distance_kullback_right(A, B), 0)
    assert_array_almost_equal(distance_kullback_sym(A, B), 0)


def test_distance_euclid():
    """Test euclidean distance"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert distance_euclid(A, B) == 0


def test_distance_logeuclid():
    """Test logeuclid distance"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert distance_logeuclid(A, B) == 0


def test_distance_wasserstein():
    """Test wasserstein distance"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert distance_wasserstein(A, B) == 0


def test_distance_logdet():
    """Test logdet distance"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert distance_logdet(A, B) == 0


def test_distance_generic_riemann():
    """Test riemannian distance for generic function"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert distance(A, B, metric='riemann') == distance_riemann(A, B)


def test_distance_generic_euclid():
    """Test euclidean distance for generic function"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert distance(A, B, metric='euclid') == distance_euclid(A, B)


def test_distance_generic_logdet():
    """Test logdet distance for generic function"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert distance(A, B, metric='logdet') == distance_logdet(A, B)


def test_distance_generic_logeuclid():
    """Test logeuclid distance for generic function"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert distance(A, B, metric='logeuclid') == distance_logeuclid(A, B)


def test_distance_generic_kullback():
    """Test logeuclid distance for generic function"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert distance(A, B, metric='kullback') == distance_kullback(A, B)
    assert distance(A, B, metric='kullback_right') == \
        distance_kullback_right(A, B)
    assert distance(A, B, metric='kullback_sym') == \
        distance_kullback_sym(A, B)


def test_distance_generic_custom():
    """Test custom distance for generic function"""
    A = 2*np.eye(3)
    B = 2*np.eye(3)
    assert distance(
        A, B, metric=distance_logeuclid) == distance_logeuclid(A, B)


def test_pairwise_distance_matrix():
    """Test pairwise distance"""
    A = np.array([2*np.eye(3), 3*np.eye(3)])
    B = np.array([2*np.eye(3), 3*np.eye(3)])
    pairwise_distance(A, B)
