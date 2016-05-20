from numpy.testing import assert_array_almost_equal
from nose.tools import (assert_equal, assert_raises, assert_almost_equal,
                        assert_true)
from scipy.linalg import eig, norm
import numpy as np


from pyriemann.utils.base import (sqrtm, invsqrtm, logm, expm, powm,
                                  generate_spd_matrices)

def test_sqrtm():
    """Test matrix square root"""
    C = 2*np.eye(3)
    Ctrue = np.sqrt(2)*np.eye(3)
    assert_array_almost_equal(sqrtm(C),Ctrue)


def test_invsqrtm():
    """Test matrix inverse square root"""
    C = 2*np.eye(3)
    Ctrue = (1.0/np.sqrt(2))*np.eye(3)
    assert_array_almost_equal(invsqrtm(C),Ctrue)
    

def test_logm():
    """Test matrix logarithm"""
    C = 2*np.eye(3)
    Ctrue = np.log(2)*np.eye(3)
    assert_array_almost_equal(logm(C),Ctrue)
    

def test_expm():
    """Test matrix exponential"""
    C = 2*np.eye(3)
    Ctrue = np.exp(2)*np.eye(3)
    assert_array_almost_equal(expm(C),Ctrue)


def test_powm():
    """Test matrix power"""
    C = 2*np.eye(3)
    Ctrue = (2**0.5)*np.eye(3)
    assert_array_almost_equal(powm(C,0.5),Ctrue)


def test_generate_spd_matrices():
    """Test generate_spd_matrices"""
    assert_raises(ValueError, generate_spd_matrices, constraint=42)
    
    n_samples, ndim = 10, 3
    constraint_list = [None, 'unit_norm', 'condition_number']
    # Check size
    for c in constraint_list:
        covmats = generate_spd_matrices(n_samples, ndim, constraint=c)
        assert_equal(covmats.shape, (n_samples, ndim, ndim))

    # Check that all eigenvalues are positive
    for c in constraint_list:
        covmats = generate_spd_matrices(n_samples, ndim, constraint=c)
        for i in range(n_samples):
            w, _ = eig(covmats[i, :, :])
            assert_true(np.all(w > 0))

    # Check the constraint
    covmats = generate_spd_matrices(n_samples, ndim, constraint='unit_norm')
    for i in range(n_samples):
        assert_almost_equal(norm(covmats[i, :, :]), 1.)

    
