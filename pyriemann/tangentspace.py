from .utils import mean_covariance, tangent_space, untangent_space

import numpy
from sklearn.base  import BaseEstimator, TransformerMixin
from sklearn.lda import LDA

class TangentSpace(BaseEstimator, TransformerMixin):

    def __init__(self,metric='riemann',tsupdate = False):

        self.metric = metric
        self.tsupdate = tsupdate 

        
    def fit(self,X,y=None):
        # compute mean covariance
        self.Cr = mean_covariance(X,metric=self.metric)
        
    def transform(self,X):
       
        if self.tsupdate:
            Cr = mean_covariance(X,metric=self.metric)
        else:
            Cr = self.Cr
        return tangent_space(X,Cr)

    def fit_transform(self,X,y=None):
        # compute mean covariance
        self.Cr = mean_covariance(X,metric=self.metric)
        return tangent_space(X,self.Cr)
    
    def inverse_transform(self,X,y=None):
        return untangent_space(X,self.Cr)

########################################################################        
class FGDA(BaseEstimator, TransformerMixin):

    def __init__(self,metric='riemann',tsupdate = False):

        self._ts = TangentSpace(tsupdate=tsupdate)
        
    def fit(self,X,y=None):
        self.classes= numpy.unique(y)
        self._lda = LDA(n_components=len(self.classes)-1,solver='eigen', shrinkage='auto')
        
        ts = self._ts.fit_transform(X)
        self._lda.fit(ts,y)
        
        W = self._lda.coef_.copy()
        self._W = numpy.dot(numpy.dot(W.T,numpy.linalg.pinv(numpy.dot(W,W.T))),W)
        
    def transform(self,X):
       ts = self._ts.transform(X)
       ts = numpy.dot(ts,self._W)
       cov = self._ts.inverse_transform(ts)
       return cov

    def fit_transform(self,X,y=None):
       self.classes= numpy.unique(y)
       self._lda = LDA(n_components=len(self.classes)-1,solver='lsqr', shrinkage='auto')
       
       ts = self._ts.fit_transform(X)
       self._lda.fit(ts,y)
       
       W = self._lda.coef_
       self._W = numpy.dot(numpy.dot(W.T,numpy.linalg.pinv(numpy.dot(W,W.T))),W)
       ts = numpy.dot(ts,self._W)
       cov = self._ts.inverse_transform(ts)
       return cov