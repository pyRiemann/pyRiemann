import numpy

from scipy.linalg import eigh
from sklearn.base  import BaseEstimator, TransformerMixin

###############################################################################
class Xdawn(BaseEstimator,TransformerMixin):
    """ 
    Compute double xdawn, project the signal

    """    
    def __init__(self,nfilter=4,applyfilters=True,classes=None):
        self.nfilter = nfilter
        self.applyfilters = applyfilters
        self.classes = classes
    
    def fit(self,X,y):
        Nt,Ne,Ns = X.shape
        
        if self.classes is None:
            self.classes = numpy.unique(y)
                    
        #FIXME : too many reshape operation         
        tmp = X.transpose((1,2,0))
        Cx = numpy.matrix(numpy.cov(tmp.reshape(Ne,Ns*Nt)))
        
        self.P = []
        self.V = []
        self._patterns = []
        for c in self.classes:
            # Prototyped responce for each class
            P = numpy.mean(X[y==c,:,:],axis=0)
        
            # Covariance matrix of the prototyper response & signal
            C = numpy.matrix(numpy.cov(P))

            # Spatial filters
            evals, evecs = eigh(C, Cx)
            evecs = evecs[:, numpy.argsort(evals)[::-1]]  # sort eigenvectors
            evecs /= numpy.apply_along_axis(numpy.linalg.norm, 0, evecs)
            V = evecs            
            A = numpy.linalg.pinv(V.T)                
            # create the reduced prototyped response
            self.V.append(V[:,0:self.nfilter].T)
            self._patterns.append(A[:,0:self.nfilter].T)
            self.P.append(numpy.dot(V[:,0:self.nfilter].T,P))
            
        
        self.P = numpy.concatenate(self.P,axis=0)
        self.V = numpy.concatenate(self.V,axis=0)
        self._patterns = numpy.concatenate(self._patterns,axis=0)
        
    
    def transform(self,X):
        X = numpy.dot(self.V,X)
        X = X.transpose((1,0,2))
        return X
    
    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)   
        