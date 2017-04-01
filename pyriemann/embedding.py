"""Embedding covariance matrices via manifold learning techniques."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.utils.distance import distance
    
class Embedding(BaseEstimator, TransformerMixin):

    """Embed SPD matrices into an Euclidean space of smaller dimension.

    It uses diffusion maps to embed the SPD matrices into an Euclidean space.
    The Euclidean distance between points in this new space approximates 
    the Diffusion distance (also called commute distance) between vertices 
    of a graph where each SPD matrix is a vertex. 

    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric to be used for defining pairwise distance between 
        covariance matrices. 

    """

    def __init__(self, metric='riemann'):
        """Init."""
        self.metric = metric

    def fit(self, X):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        self : Embedding instance
        The Embedding instance.
        """
        return self

    def transform(self, X, eps=None, tdiff=0.0):
        """Calculates the coordinates of the embedded points.

        Parameters
        ----------
        X :   ndarray, shape (n_trials, n_channels, n_channels)
              ndarray of SPD matrices.
        eps:  float (default: None)
              the scaling of the Gaussian kernel. If none is given
              it will use the square of the median of pairwise distances between
              points (same criterium used in the R package implementation).            
        tiff: float (default: 0.0)
              diffusion time to be considered. It allows for multiscale analysis
              but usually tdiff=0.0 gives enough information.

        Returns
        -------
        u : ndarray, shape (n_trials, n_trials)
            ndarray with the embeddings of the covariance matrices.
        l : ndarray, shape (n_trials)
            ndarray with the eigenvalues of the diffusion matrix.            
        """
        
        u,l = get_Embedding(X, self.metric, eps, tdiff)
        
        return u,l   
        
def make_distanceMatrix(points, metric):

    # make matrix with pairwise distances between points
    Npoints = points.shape[0]
    distmatrix = np.zeros((Npoints, Npoints))
    for ii,pi in enumerate(points):
        for jj,pj in enumerate(points):
            distmatrix[ii,jj] = distance(pi, pj, metric=metric)
            
    return distmatrix       

def make_kernelMatrix(distmatrix, eps):
    
    # make kernel matrix from the distance matrix
    kernel = np.exp(-distmatrix**2/(4*eps))    

    # renormalize the kernel matrix
    q = np.dot(kernel, np.ones(len(kernel)))
    kernel_r = np.divide(kernel, np.outer(q,q)) 
        
    return kernel_r

def make_transitionMatrix(kernel):

    # normalize rows of the kernel so it becomes a prob transition matrix    
    d = np.sqrt(np.dot(kernel, np.ones(len(kernel))))
    P = np.divide(kernel, np.outer(d, d))  
    
    return P

def get_Embedding(points, metric, eps=None, tdiff=0):
    
    # this implementation follows the algorithm in page 34 of 
    # Stephane's Lafon PhD thesis "Diffusion Maps and Geometric Harmonics"
    
    # eps is the scaling of the gaussian kernel and defines locality
    # tdiff is the diffusion time to be considered at the output
    # (tdiff = 0 is usually enough for a first investigation of data)
    
    # from the set of points build the prob transition matrix along the graph    
    d = make_distanceMatrix(points, metric)  
    if eps is None:
        eps = np.median(d)**2/2
    K = make_kernelMatrix(distmatrix=d, eps=eps)
    P = make_transitionMatrix(K)
    
    # the eigendecomposition will give the spectral embedding of the points
    u,s,v = np.linalg.svd(P)    
    
    # because of the matrix normalizations, the actual embedding has to be
    # corrected by dividing its coordinates by the first left singular vector
    phi = np.copy(u)
    for i in range(len(u)):
        phi[:,i] = (s[i]**tdiff)*np.divide(u[:,i], u[:,0])
    
    return phi, s    





    
    
