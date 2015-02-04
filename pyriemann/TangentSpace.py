from sklearn.base  import BaseEstimator, TransformerMixin

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
