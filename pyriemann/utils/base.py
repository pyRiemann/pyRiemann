import numpy as np
import scipy

from numpy.core.numerictypes import typecodes


def _matrix_operator(Ci, operator):
    """matrix equivalent of an operator."""
    if Ci.dtype.char in typecodes['AllFloat'] and not np.isfinite(Ci).all():
        raise ValueError(
            "Covariance matrices must be positive definite. Add "
            "regularization to avoid this error.")
    eigvals, eigvects = scipy.linalg.eigh(Ci, check_finite=False)
    eigvals = np.diag(operator(eigvals))
    Out = np.dot(np.dot(eigvects, eigvals), eigvects.T)
    return Out


def sqrtm(Ci):
    r"""Return the matrix square root of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix square root

    """  # noqa
    return _matrix_operator(Ci, np.sqrt)


def logm(Ci):
    r"""Return the matrix logarithm of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the covariance matrix
    :returns: the matrix logarithm

    """
    return _matrix_operator(Ci, np.log)


def expm(Ci):
    r"""Return the matrix exponential of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix exponential

    """
    return _matrix_operator(Ci, np.exp)


def invsqrtm(Ci):
    r"""Return the inverse matrix square root of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{-1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the inverse matrix square root

    """  # noqa
    def isqrt(x): return 1. / np.sqrt(x)
    return _matrix_operator(Ci, isqrt)


def powm(Ci, alpha):
    r"""Return the matrix power :math:`\\alpha` of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{\\alpha} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :param alpha: the power to apply
    :returns: the matrix power

    """  # noqa
    def power(x): return x**alpha
    return _matrix_operator(Ci, power)
