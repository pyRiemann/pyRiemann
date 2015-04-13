import numpy
from numpy import matrix, sqrt, diag, log, exp, mean, eye, triu_indices_from, zeros, cov, concatenate, triu
from numpy.linalg import norm, inv, eigvals
from numpy.linalg import eigh as eig
from scipy.linalg import eigvalsh
from sklearn.covariance import LedoitWolf
def sqrtm(Ci):
	D,V = eig(Ci)
	D = matrix(diag(sqrt(D)))
	V = matrix(V)
	Out = matrix(V*D*V.T)
	return Out

def logm(Ci):
    D,V = eig(Ci)
    Out = numpy.dot(numpy.multiply(V,log(D)),V.T)
    return Out
	
def expm(Ci):
	D,V = eig(Ci)
	D = matrix(diag(exp(D)))
	V = matrix(V)
	Out = matrix(V*D*V.T)
	return Out

def invsqrtm(Ci):
	D,V = eig(Ci)
	D = matrix(diag(1.0/sqrt(D)))
	V = matrix(V)
	Out = matrix(V*D*V.T)
	return Out

def powm(Ci,alpha):
	D,V = eig(Ci)
	D = matrix(diag(D**alpha))
	V = matrix(V)
	Out = matrix(V*D*V.T)
	return Out		
	
	
def distance(C0,C1):
    return sqrt((log(eigvalsh(C0,C1))**2).sum())
	
def geodesic(C0,C1,alpha):
    A = matrix(sqrtm(C0))
    B = matrix(invsqrtm(C0))
    C = B*C1*B
    D = matrix(powm(C,alpha))
    E = matrix(A*D*A)
    return E
	
def tangent_space(covmats,ref):
    Nt,Ne,Ne = covmats.shape
    Cm12 = invsqrtm(ref)
    idx = triu_indices_from(ref)
    T = zeros((Nt,Ne*(Ne+1)/2))
    coeffs = (sqrt(2)*triu(numpy.ones((Ne,Ne)),1) + numpy.eye(Ne))[idx]
    for index in range(Nt):
        tmp = numpy.dot(numpy.dot(Cm12,covmats[index,:,:]),Cm12)
        tmp = logm(tmp)
        T[index,:] = numpy.multiply(coeffs,tmp[idx])
    return T
 
def untangent_space(T,ref):
    Nt,Nd = T.shape
    Ne = int((sqrt(1+8*Nd)-1)/2)
    C12 = sqrtm(ref)
    
    idx = triu_indices_from(ref)
    covmats = zeros((Nt,Ne,Ne))
    covmats[:,idx[0],idx[1]] = T
    for i in range(Nt):
        covmats[i] = diag(diag(covmats[i])) + triu(covmats[i],1)/sqrt(2) + triu(covmats[i],1).T/sqrt(2)
        covmats[i] = expm(covmats[i])
        covmats[i] = numpy.dot(numpy.dot(C12,covmats[i]),C12)
        
    return covmats
 
def riemann_mean(covmats,tol=10e-9,maxiter=50):
    #init 
    Nt,Ne,Ne = covmats.shape
    C = numpy.mean(covmats,axis=0)
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
    
def logeuclid_mean(covmats):
    Nt,Ne,Ne = covmats.shape 
    T = zeros((Ne,Ne))
    for index in range(Nt):
        T+= logm(matrix(covmats[index,:,:]))
    C = expm(T/Nt)
    
    return C    

def logdet_mean(covmats,tol=10e-5,maxiter=50):
    #init 
    Nt,Ne,Ne = covmats.shape
    C = mean(covmats,axis=0)
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

def euclid_mean(covmats):
    return mean(covmats,axis=0)    
    
def identity(covmats):    
    C = eye(covmats.shape[1])
    return C
    
def mean_covariance(covmats,metric='riemann',*args):
    options = {'riemann' : riemann_mean,
               'logeuclid': logeuclid_mean,
               'euclid' : euclid_mean,
               'identity' : identity,
               'logdet' : logdet_mean}
    C = options[metric](covmats,*args)
    return C
    
    