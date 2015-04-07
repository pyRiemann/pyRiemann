import numpy
from sklearn.base  import BaseEstimator, ClassifierMixin

from .utils import mean_covariance, distance
from .tangentspace import TangentSpace,FGDA

#######################################################################
class MDM(BaseEstimator, ClassifierMixin):
    
    def __init__(self,metric='riemann'):
        self.metric = metric
        
    def fit(self,X,y):

        self.classes= numpy.unique(y)
        
        self.covmeans = []
        
        for l in self.classes:
            self.covmeans.append(mean_covariance(X[y==l,:,:],metric=self.metric))
        
    def predict(self,covtest):
        Nt = covtest.shape[0]
        Nc = len(self.classes)
        dist = numpy.zeros((Nc,Nt))
        
        for m in range(Nc):
            for k in range(Nt):
                dist[m,k] = distance(covtest[k,:,:],self.covmeans[m])
                
        return self.classes[dist.argmin(axis=0)]
        
    def predict_proba(self,covtest):
        Nt = covtest.shape[0]
        Nc = len(self.classes)
        dist = numpy.zeros((Nc,Nt))
        
        for m in range(Nc):
            for k in range(Nt):
                dist[m,k] = distance(covtest[k,:,:],self.covmeans[m])
                
        return -numpy.diff(dist,axis=0).T
        
#######################################################################
class FgMDM(BaseEstimator, ClassifierMixin):
    
    def __init__(self,metric='riemann',tsupdate = False):
        self._mdm = MDM()
        self._fgda = FGDA(metric=metric,tsupdate=tsupdate)
        
    def fit(self,X,y):
        cov = self._fgda.fit_transform(X,y)
        self._mdm.fit(cov,y)        
        
        
    def predict(self,X):
        cov = self._fgda.transform(X)
        return self._mdm.predict(cov)
        
    def predict_proba(self,X):
        cov = self._fgda.transform(X)
        return self._mdm.predict_proba(cov)