import numpy as np
from scipy.linalg import eigh, norm, eig
from numpy import diag, exp, log, multiply, sqrt, eye, zeros, real
from numpy.random import randn, rand

###############################################################
# Basic Functions
###############################################################


def sqrtm(Ci):
    """Return the matrix square root of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix square root

    """
    D, V = eigh(Ci)
    D = diag(sqrt(D))
    Out = V.dot(D.dot(V.T))
    return Out


def logm(Ci):
    """Return the matrix logarithm of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix logarithm

    """
    D, V = eigh(Ci)
    Out = multiply(V, log(D)).dot(V.T)
    return Out


def expm(Ci):
    """Return the matrix exponential of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix exponential

    """
    D, V = eigh(Ci)
    D = diag(exp(D))
    Out = V.dot(D.dot(V.T))
    return Out


def invsqrtm(Ci):
    """Return the inverse matrix square root of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{-1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the inverse matrix square root

    """
    D, V = eigh(Ci)
    D = diag(1.0 / sqrt(D))
    Out = V.dot(D.dot(V.T))
    return Out


def powm(Ci, alpha):
    """Return the matrix power :math:`\\alpha` of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{\\alpha} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :param alpha: the power to apply
    :returns: the matrix power

    """
    D, V = eigh(Ci)
    D = diag(D**alpha)
    Out = V.dot(D.dot(V.T))
    return Out


def check_version(library, min_version):
    """Check minimum library version required

    Parameters
    ----------
    library : str
        The library name to import. Must have a ``__version__`` property.
    min_version : str
        The minimum version string. Anything that matches
        ``'(\\d+ | [a-z]+ | \\.)'``

    Returns
    -------
    ok : bool
        True if the library exists with at least the specified version.

    Adapted from MNE-Python: http://github.com/mne-tools/mne-python
    """
    from distutils.version import LooseVersion
    ok = True
    try:
        library = __import__(library)
    except ImportError:
        ok = False
    else:
        this_version = LooseVersion(library.__version__)
        if this_version < min_version:
            ok = False
    return ok


def generate_spd_matrices(n_samples=10, ndim=3, constraint=None,
                          condition_number=1e1):
    """Return random SPD matrices, with different constraints.

    Symmetric Positive Definite real matrices belongs to :math:`Sym^+_d`
    
    .. math::
             Sym^+_d = \{ C in R^{d \times d} | C^t = C and z^t C z > 0 for all z \in R^{d} \backslash 0 \}

    When constraint is None, return n_samples SPD matrices.
    When it is 'unit_norm', return SPD matrices with norm(C) = 1
    When 'condition_number', return matrices with specified condition number
    
    Parameters
    ----------
    ndim : int
        Dimension of the SPD matrices
    n_samples : int
        Number of SPD matrices to return
    constraint: str
        Either None, 'unit_norm' or 'condition_number'
    condition_number: int
        Parameter used when constraint='condition_number'

    Returns
    -------
    covmats : array, shape(n_samples, ndim, ndim)
        The generated SPD matrices

    This is the adaptation of the Matlab code proposed by Dario Bini and
    Bruno Iannazzo, http://bezout.dm.unipi.it/software/mmtoolbox/
    """
    covmats = zeros(shape=(n_samples, ndim, ndim))
    if constraint is None:
        for i in range(n_samples):
            W = randn(ndim, ndim)
            A = sqrtm(W.dot(W.T))
            covmats[i, :, :] = (A + A.T) / 2.
    elif constraint is 'unit_norm':
        for i in range(n_samples):
            W = randn(ndim, ndim) - rand(ndim, ndim)
            W = W.T.dot(W)
            A = W/norm(W)
            covmats[i, :, :] = (A + A.T) / 2.
    elif constraint is 'condition_number':
        for i in range(n_samples):
            W = rand(ndim, ndim) - rand(ndim, ndim)
            W = W.T.dot(W)
            W -= eye(ndim)*min(real(eig(W, right=False)))
            W /= norm(W)
            W += eye(ndim)/(condition_number-1)
            covmats[i, :, :] = W /norm(W)
    else:
        raise (ValueError, "Unknown constraint")
    return covmats
