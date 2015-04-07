import numpy

from .utils import covariances, cospectrum,nextpow2
from .spatialfilters import Xdawn
from sklearn.base  import BaseEstimator, TransformerMixin

###############################################################################
class Covariances(BaseEstimator,TransformerMixin):
    """ 
    compute the covariances matrices

    """    
    def __init__(self):
        pass
        
    def fit(self,X,y):
        pass
    
    def transform(self,X):
            
        covmats = covariances(X)
        return covmats
    
    def fit_transform(self,X,y):
        return self.transform(X)

###############################################################################
class ERPCovariances(BaseEstimator,TransformerMixin):
    """ 
    Compute xdawn, project the signal and compute the covariances

    """    
    def __init__(self,classes=None):
        self.classes = classes
        
    def fit(self,X,y):
        
        if self.classes is not None:
            classes = self.classes
        else:
            classes = numpy.unique(y)
        
        
        
        self.P = []
        for c in classes:
            # Prototyped responce for each class
            P = numpy.mean(X[y==c,:,:],axis=0)
            self.P.append(P)
        #self.P = self.P[1]           
        self.P = numpy.concatenate(self.P,axis=0)
    
    def transform(self,X):
            
        covmats = covariances_EP(X,self.P)
        return covmats
    
    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)

###############################################################################
class XdawnCovariances(BaseEstimator,TransformerMixin):
    """ 
    Compute xdawn, project the signal and compute the covariances

    """    
    def __init__(self,nfilter=4,applyfilters=True):
        self.Xd = Xdawn(nfilter=nfilter,applyfilters=applyfilters)
        
    def fit(self,X,y):
        self.Xd.fit(X,y)
        
    
    def transform(self,X):
        X = self.Xd.transform(X)
            
        covmats = covariances_EP(X,self.Xd.P)
        return covmats
    
    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)

###############################################################################
class CospCovariances(BaseEstimator,TransformerMixin):
    """ 
    compute the cospectral covariance matrices

    """    
    def __init__(self,window=128,overlap=0.75,fmin = None,fmax = None,fs = None,phase_correction=False):
        self._window = nextpow2(window)
        self._overlap = overlap
        self._fmin = fmin
        self._fmax = fmax
        self._fs = fs
        
        self._phase_corr = phase_correction
        
    def fit(self,X,y):
        pass
    
    def transform(self,X):
        
        Nt,Ne,_ = X.shape
        out = []
        
        for i in range(Nt):
            S = cospectrum(X[i],window=self._window,overlap=self._overlap,fmin=self._fmin,fmax=self._fmax,fs=self._fs)
            out.append(S.real)
        
        return numpy.array(out)
    
    def fit_transform(self,X,y):
        return self.transform(X)