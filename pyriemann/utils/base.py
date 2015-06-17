import numpy
import scipy

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
    D, V = scipy.linalg.eigh(Ci)
    D = numpy.matrix(numpy.diag(numpy.sqrt(D)))
    V = numpy.matrix(V)
    Out = numpy.matrix(V * D * V.T)
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
    D, V = scipy.linalg.eigh(Ci)
    Out = numpy.dot(numpy.multiply(V, numpy.log(D)), V.T)
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
    D, V = scipy.linalg.eigh(Ci)
    D = numpy.matrix(numpy.diag(numpy.exp(D)))
    V = numpy.matrix(V)
    Out = numpy.matrix(V * D * V.T)
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
    D, V = scipy.linalg.eigh(Ci)
    D = numpy.matrix(numpy.diag(1.0 / numpy.sqrt(D)))
    V = numpy.matrix(V)
    Out = numpy.matrix(V * D * V.T)
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
    D, V = scipy.linalg.eigh(Ci)
    D = numpy.matrix(numpy.diag(D**alpha))
    V = numpy.matrix(V)
    Out = numpy.matrix(V * D * V.T)
    return Out
