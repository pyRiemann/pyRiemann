from .utils import mean_covariance, distance

import numpy
from sklearn.base  import BaseEstimator, TransformerMixin

class ElectrodeSelection(BaseEstimator, TransformerMixin):

    def __init__(self,nelec = 16,metric='riemann'):
        self.nelec = nelec
        self.metric = metric
        self.subelec = -1
        self.dist = []
    
    def fit(self,X,y=None):
        self.classes= numpy.unique(y)
        self.covmeans = []
        for l in self.classes:
            self.covmeans.append(mean_covariance(X[y==l,:,:],metric=self.metric))
        
        Ne,_ = self.covmeans[0].shape
        
        self.subelec = range(0,Ne,1) 
        while (len(self.subelec))>self.nelec :
            di = numpy.zeros((len(self.subelec),1))
            for idx in range(len(self.subelec)):
                sub = self.subelec[:]
                sub.pop(idx)
                di[idx] = 0
                for i in range(len(self.covmeans)):
                    for j in range(i+1,len(self.covmeans)):
                         di[idx] += distance(self.covmeans[i][:,sub][sub,:],self.covmeans[j][:,sub][sub,:])
            #print di
            torm = di.argmax()
            self.dist.append(di.max())
            self.subelec.pop(torm)        
        #print self.subelec
        
    def transform(self,X):
       return X[:,self.subelec,:][:,:,self.subelec]

    def fit_transform(self,X,y=None):
        self.fit(X,y)
        return self.transform(X)