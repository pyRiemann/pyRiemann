from sklearn.base  import BaseEstimator, ClassifierMixin

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