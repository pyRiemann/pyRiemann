from numpy import matrix, sqrt, diag, log, exp, mean, eye, triu_indices_from, zeros, cov, concatenate, triu
from numpy.linalg import norm
from numpy.linalg import eigh as eig

def sqrtm(Ci):
	D,V = eig(Ci)
	D = matrix(diag(sqrt(D)))
	V = matrix(V)
	Out = matrix(V*D*V.T)
	return Out

def logm(Ci):
	D,V = eig(Ci)
	D = matrix(diag(log(D)))
	V = matrix(V)
	Out = matrix(V*D*V.T)
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
    P = invsqrtm(C0)
    A = P*C1*P
    D,V = eig(A)
    l = log(D)
    d = norm(l)
    return d
	
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
    for index in range(Nt):
        tmp = logm(matrix(Cm12*covmats[index,:,:]*Cm12))
        #fixme : not very efficient        
        tmp = sqrt(2)*triu(tmp,1) + diag(diag(tmp))
        T[index,:] = tmp[idx]
    return T
 
def riemann_mean(covmats,tol=10e-9,maxiter=50):
    #init 
    Nt,Ne,Ne = covmats.shape
    C = mean(covmats,axis=0)
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
        T = zeros((Nt,Ne,Ne))

        for index in range(Nt):
            T[index,:,:] = logm(matrix(Cm12*covmats[index,:,:]*Cm12))
            
        J = mean(T,axis=0)	
        crit = norm(J,ord='fro')
        h = nu*crit
        if h < tau:            
            C = matrix(C12*expm(nu*J)*C12)
            nu = 0.95*nu
            tau = h
        else:
            nu = 0.5*nu
    return C	
    
def logeuclid_mean(covmats):
    Nt,Ne,Ne = covmats.shape 
    T = zeros((Ne,Ne,Nt))
    for index in range(Nt):
        T[:,:,index] = logm(matrix(covmats[index,:,:]))
    C = expm(mean(T,axis=2))
    
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
               'identity' : identity}
    C = options[metric](covmats,*args)
    return C
    
def covariances(X):
     Nt,Ne,Ns = X.shape
     covmats = zeros((Nt,Ne,Ne))
     for i in range(Nt):
         covmats[i,:,:] = cov(X[i,:,:])
     return covmats
     
def covariances_EP(X,P):
     Nt,Ne,Ns = X.shape
     Np,Ns = P.shape
     covmats = zeros((Nt,Ne+Np,Ne+Np))
     for i in range(Nt):
         covmats[i,:,:] = cov(concatenate((P,X[i,:,:]),axis=0))
     return covmats
