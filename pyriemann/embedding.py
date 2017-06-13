"""Embedding covariance matrices via manifold learning techniques."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import spectral_embedding
from pyriemann.utils.distance import distance, pairwise_distance
    
class CovEmbedding(BaseEstimator):

    """Embed SPD matrices into an Euclidean space of smaller dimension.

    It uses diffusion maps to embed the SPD matrices into an Euclidean space.
    The Euclidean distance between points in this new space approximates 
    the Diffusion distance (also called commute distance) between vertices 
    of a graph where each SPD matrix is a vertex. 

    Parameters
    ----------
    n_components : integer, default: 2
        The dimension of the projected subspace.
        
    metric : string | dict (default: 'riemann')
        The type of metric to be used for defining pairwise distance between 
        covariance matrices. 
    eps:  float (default: None)
          the scaling of the Gaussian kernel. If none is given
          it will use the square of the median of pairwise distances between
          points.                   

    """

    def __init__(self, n_components=2, metric='riemann', eps=None):
        """Init."""
        self.metric = metric
        self.n_components = n_components
        self.eps = eps
        
    def _get_affinity_matrix(self, X, eps):
        
        # make matrix with pairwise distances between points
        Npoints = X.shape[0]
        distmatrix = np.zeros((Npoints, Npoints))
        for ii,pi in enumerate(X):
            for jj,pj in enumerate(X):
                distmatrix[ii,jj] = distance(pi, pj, metric=self.metric)                                 
        
        # determine which scale for the gaussian kernel
        if self.eps is None:
            eps = np.median(distmatrix)**2/2                           
                           
        # make kernel matrix from the distance matrix
        kernel = np.exp(-distmatrix**2/(4*eps))    
    
        # normalize the kernel matrix 
        q = np.dot(kernel, np.ones(len(kernel)))
        kernel_n = np.divide(kernel, np.outer(q,q)) 
        
        self.affinity_matrix_ = kernel_n                           
        return self.affinity_matrix_

    def fit(self, X, y=None):
        """Fit the model from data in X.        

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        self : Embedding instance
        The Embedding instance.
        """                   
        
        affinity_matrix = self._get_affinity_matrix(X, self.eps)
        self.embedding_ = spectral_embedding(adjacency=affinity_matrix, 
                                             n_components=self.n_components,                                              
                                             norm_laplacian=True)
        
        return self

    def fit_transform(self, X, y=None):
        """Calculates the coordinates of the embedded points.

        Parameters
        ----------
        X :   ndarray, shape (n_trials, n_channels, n_channels)
              ndarray of SPD matrices.

        Returns
        -------
        X_new: array-like, shape (n_samples, n_components)           
        """
        
        self.fit(X)    
        return self.embedding_  
        












