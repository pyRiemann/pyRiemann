import numpy
from numpy import matrix, sqrt, diag, log, exp, mean, eye, triu_indices_from, zeros, cov, concatenate, triu
from numpy.linalg import norm, inv, eigvals, det
from numpy.linalg import eigh as eig
from scipy.linalg import eigvalsh


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
    D,V = eig(Ci)
    D = matrix(diag(sqrt(D)))
    V = matrix(V)
    Out = matrix(V*D*V.T)
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
    D,V = eig(Ci)
    Out = numpy.dot(numpy.multiply(V,log(D)),V.T)
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
    D,V = eig(Ci)
    D = matrix(diag(exp(D)))
    V = matrix(V)
    Out = matrix(V*D*V.T)
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
    D,V = eig(Ci)
    D = matrix(diag(1.0/sqrt(D)))
    V = matrix(V)
    Out = matrix(V*D*V.T)
    return Out

def powm(Ci,alpha):
    """Return the matrix power :math:`\\alpha` of a covariance matrix defined by :
    
    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{\\alpha} \mathbf{V}^T
            
    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`
   
    :param Ci: the coavriance matrix
    :param alpha: the power to apply 
    :returns: the matrix power
    
    """
    D,V = eig(Ci)
    D = matrix(diag(D**alpha))
    V = matrix(V)
    Out = matrix(V*D*V.T)
    return Out		
###############################################################
# geodesic 	
###############################################################	
def geodesic(A,B,alpha,metric='riemann'):
    """Return the matrix at the position alpha on the geodesic between A and B according to the metric :
    
   
    :param A: the first coavriance matrix
    :param B: the second coavriance matrix
    :param alpha: the position on the geodesic
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' , 'logeuclid' , 'euclid'
    :returns: the covariance matrix on the geodesic
    
    """
    options = {'riemann' : geodesic_riemann,
               'logeuclid': geodesic_logeuclid,
               'euclid' : geodesic_euclid}
    C = options[metric](covmats,*args)
    return C

def geodesic_riemann(A,B,alpha=0.5):
    """Return the matrix at the position alpha on the riemannian geodesic between A and B  :
    
    .. math::
            \mathbf{C} = \mathbf{A}^{1/2} \left( \mathbf{A}^{-1/2} \mathbf{B} \mathbf{A}^{-1/2} \\right)^\\alpha \mathbf{A}^{1/2}
    
    C is equal to A if alpha = 0 and B if alpha = 1
    
    :param A: the first coavriance matrix
    :param B: the second coavriance matrix
    :param alpha: the position on the geodesic
    :returns: the covariance matrix
    
    """
    A = matrix(sqrtm(C0))
    B = matrix(invsqrtm(C0))
    C = B*C1*B
    D = matrix(powm(C,alpha))
    E = matrix(A*D*A)
    return E
    
def geodesic_euclid(A,B,alpha=0.5):
    """Return the matrix at the position alpha on the euclidean geodesic between A and B  :
    
    .. math::
            \mathbf{C} = (1-\\alpha) \mathbf{A} + \alpha \mathbf{B} 
    
    C is equal to A if alpha = 0 and B if alpha = 1
    
    :param A: the first coavriance matrix
    :param B: the second coavriance matrix
    :param alpha: the position on the geodesic
    :returns: the covariance matrix
    
    """
    return (1-alpha)*A + alpha*B
    
def geodesic_logeuclid(A,B,alpha=0.5):
    """Return the matrix at the position alpha on the log euclidean geodesic between A and B  :
    
    .. math::
            \mathbf{C} =  \exp \left( (1-\\alpha) \log(\mathbf{A}) + \alpha \log(\mathbf{B}) \\right)
    
    C is equal to A if alpha = 0 and B if alpha = 1
    
    :param A: the first coavriance matrix
    :param B: the second coavriance matrix
    :param alpha: the position on the geodesic
    :returns: the covariance matrix
    
    """
    return expm( (1-alpha)*logm(A) + alpha*logm(B))
###############################################################
# Tangent Space	
###############################################################		
def tangent_space(covmats,Cref):
    """Project a set of covariance matrices in the tangent space according to the given reference point Cref
    
    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels 
    :param Cref: The reference covariance matrix
    :returns: the Tangent space , a matrix of Ntrials X (Nchannels*(Nchannels+1)/2)
    
    """
    Nt,Ne,Ne = covmats.shape
    Cm12 = invsqrtm(Cref)
    idx = triu_indices_from(Cref)
    T = zeros((Nt,Ne*(Ne+1)/2))
    coeffs = (sqrt(2)*triu(numpy.ones((Ne,Ne)),1) + numpy.eye(Ne))[idx]
    for index in range(Nt):
        tmp = numpy.dot(numpy.dot(Cm12,covmats[index,:,:]),Cm12)
        tmp = logm(tmp)
        T[index,:] = numpy.multiply(coeffs,tmp[idx])
    return T
 
def untangent_space(T,Cref):
    """Project a set of Tangent space vectors in the manifold according to the given reference point Cref
    
    :param T: the Tangent space , a matrix of Ntrials X (Nchannels*(Nchannels+1)/2)
    :param Cref: The reference covariance matrix
    :returns: A set of Covariance matrix, Ntrials X Nchannels X Nchannels 
    
    """
    Nt,Nd = T.shape
    Ne = int((sqrt(1+8*Nd)-1)/2)
    C12 = sqrtm(Cref)
    
    idx = triu_indices_from(Cref)
    covmats = zeros((Nt,Ne,Ne))
    covmats[:,idx[0],idx[1]] = T
    for i in range(Nt):
        covmats[i] = diag(diag(covmats[i])) + triu(covmats[i],1)/sqrt(2) + triu(covmats[i],1).T/sqrt(2)
        covmats[i] = expm(covmats[i])
        covmats[i] = numpy.dot(numpy.dot(C12,covmats[i]),C12)
        
    return covmats

###############################################################
# Means 	
###############################################################	

def mean_riemann(covmats,tol=10e-9,maxiter=50,init=None):
    """Return the mean covariance matrix according to the Riemannian metric.
    The procedure is similar to a gradient descent minimizing the sum of 
    riemannian distance to the mean.
    
    .. math::
            \mathbf{C} = \\arg\min{(\sum_i \delta_R ( \mathbf{C} , \mathbf{C}_i)^2)} 
   
    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels 
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param init: A covariance matrix used to initialize the gradient descent. If None the Arithmetic mean is used
    :returns: the mean covariance matrix
    
    """
    #init 
    Nt,Ne,Ne = covmats.shape
    if init is None:
        C = numpy.mean(covmats,axis=0)
    else:
        C = init
    k=0
    J = eye(2)
    nu = 1.0
    tau = 10e19
    crit = norm(J,ord='fro')
    # stop when J<10^-9 or max iteration = 50
    while (crit>tol) and (k<maxiter) and (nu>tol):
        k=k+1
        C12 = sqrtm(C)
        Cm12 = invsqrtm(C)
        T = zeros((Ne,Ne))
       
        for index in range(Nt):
            tmp =  numpy.dot(numpy.dot(Cm12,covmats[index,:,:]),Cm12)
            T += logm(matrix(tmp))
        
        #J = mean(T,axis=0)
        J = T/Nt        
        crit = norm(J,ord='fro')
        h = nu*crit
        C = matrix(C12*expm(nu*J)*C12)
        if h < tau:            
            nu = 0.95*nu
            tau = h
        else:
            nu = 0.5*nu
        
    return C	
    
def mean_logeuclid(covmats):
    """Return the mean covariance matrix according to the log-euclidean metric :
    
    .. math::
            \mathbf{C} = \exp{(\\frac{1}{N} \sum_i \log{\mathbf{C}_i})}
   
    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels 
    :returns: the mean covariance matrix
    
    """
    Nt,Ne,Ne = covmats.shape 
    T = zeros((Ne,Ne))
    for index in range(Nt):
        T+= logm(matrix(covmats[index,:,:]))
    C = expm(T/Nt)
    
    return C    

def mean_logdet(covmats,tol=10e-5,maxiter=50,init=None):
    """Return the mean covariance matrix according to the logdet metric.
    This is an iterative procedure where the update is:
    
    .. math::
            \mathbf{C} = \left(\sum_i \left( 0.5 \mathbf{C} + 0.5 \mathbf{C}_i \\right)^{-1} \\right)^{-1}
   
    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels 
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param init: A covariance matrix used to initialize the iterative procedure. If None the Arithmetic mean is used
    :returns: the mean covariance matrix
    
    """
    Nt,Ne,Ne = covmats.shape
    if init is None:
        C = mean(covmats,axis=0)
    else:
        C = init
    k=0
    J = eye(2)
    crit = norm(J,ord='fro')
    # stop when J<10^-9 or max iteration = 50
    while (crit>tol) and (k<maxiter) :
        k=k+1

        J = zeros((Ne,Ne))

        for Ci in covmats:
            J += inv(0.5*Ci + 0.5*C);
            
        J = J/Nt
        Cnew = inv(J)
        crit = norm(Cnew-C,ord='fro')
        
        C = Cnew
    if k==maxiter:
        print 'Max iter reach'
    return C	

def mean_euclid(covmats):
    """Return the mean covariance matrix according to the euclidean metric :
    
    .. math::
            \mathbf{C} = \\frac{1}{N} \sum_i \mathbf{C}_i
   
    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels 
    :returns: the mean covariance matrix
    
    """
    return mean(covmats,axis=0)    
    
def mean_identity(covmats): 
    """Return the identity matrix corresponding to the covmats sit size
    
    .. math::
            \mathbf{C} = \mathbf{I}_d
   
    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels 
    :returns: the identity matrix of size Nchannels
    
    """
    C = eye(covmats.shape[1])
    return C
    
def mean_covariance(covmats,metric='riemann',*args):
    """Return the mean covariance matrix according to the metric
    
   
    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels 
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' , 'logeuclid' , 'euclid' , 'logdet', 'indentity'
    :param args: the argument passed to the sub function 
    :returns: the mean covariance matrix
    
    """
    options = {'riemann' : mean_riemann,
               'logeuclid': mean_logeuclid,
               'euclid' : mean_euclid,
               'identity' : mean_identity,
               'logdet' : mean_logdet}
    C = options[metric](covmats,*args)
    return C
    

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
    return norm(A-B,ord='fro')
   
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
    return sqrt((log(eigvalsh(A,B))**2).sum())
    
def distance_logdet(A,B):
    """Return the Log-det distance between 
    two covariance matrices A and B :
    
    .. math::
            d = \sqrt{\log(\det(\\frac{\mathbf{A}+\mathbf{B}}{2}))} - 0.5 \\times \log(\det(\mathbf{A} \\times \mathbf{B}))
    
    
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Log-Euclid distance between A and B 
    
    """
    return sqrt(log(det((A+B)/2))-0.5*log(det(numpy.dot(A,B))))
    
distance_methods = {'riemann' : distance_riemann,
                   'logeuclid': distance_logeuclid,
                    'euclid' : distance_euclid,
                    'logdet' : distance_logdet}
                        
def distance(A,B,metric='riemann'):
    """Return the distance between 
    two covariance matrices A and B according to the metric :
    
   
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' , 'logeuclid' , 'euclid' , 'logdet' 
    :returns: the distance between A and B 
    
    """
    d = distance_methods[metric](A,B)
    return d
	