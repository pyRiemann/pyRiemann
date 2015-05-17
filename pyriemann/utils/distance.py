import numpy
from numpy.linalg import eigvalsh

from .base import sqrtm,invsqrtm,powm,logm,expm
###############################################################
# distances 	
###############################################################	

def distance_euclid(A,B):
    """Return the Euclidean distance (Froebenius norm) between 
    two covariance matrices A and B :
    
    .. math::
            d = \Vert \mathbf{A} - \mathbf{B} \Vert_F
    
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Eclidean distance between A and B 
    
    """
    return numpy.linalg.norm(A-B,ord='fro')
   
def distance_logeuclid(A,B):
    """Return the Log Euclidean distance between 
    two covariance matrices A and B :
    
    .. math::
            d = \Vert \log(\mathbf{A}) - \log(\mathbf{B}) \Vert_F
    
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Log-Eclidean distance between A and B 
    
    """
    return distance_euclid(logm(A),logm(B))
    
def distance_riemann(A,B):
    """Return the Riemannian distance between 
    two covariance matrices A and B :
    
    .. math::
            d = {\left( \sum_i \log(\lambda_i)^2 \\right)}^{-1/2}
    
    where :math:`\lambda_i` are the joint eigenvalues of A and B 
    
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Riemannian distance between A and B 
    
    """
    return numpy.sqrt((numpy.log(eigvalsh(A,B))**2).sum())
    
def distance_logdet(A,B):
    """Return the Log-det distance between 
    two covariance matrices A and B :
    
    .. math::
            d = \sqrt{\log(\det(\\frac{\mathbf{A}+\mathbf{B}}{2}))} - 0.5 \\times \log(\det(\mathbf{A} \\times \mathbf{B}))
    
    
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Log-Euclid distance between A and B 
    
    """
    return numpy.sqrt(numpy.log(numpy.linalg.det((A+B)/2))-0.5*numpy.log(numpy.linalg.det(numpy.dot(A,B))))

                        
def distance(A,B,metric='riemann'):
    """Return the distance between 
    two covariance matrices A and B according to the metric :
    
   
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' , 'logeuclid' , 'euclid' , 'logdet' 
    :returns: the distance between A and B 
    
    """
    
    distance_methods = {'riemann' : distance_riemann,
                       'logeuclid': distance_logeuclid,
                        'euclid' : distance_euclid,
                        'logdet' : distance_logdet}
    d = distance_methods[metric](A,B)
    return d
	