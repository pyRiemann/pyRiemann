from sklearn.neighbors import DistanceMetric
from .utils import distance

import numpy
import pandas
from pylab import hist,plt
from sklearn.base  import BaseEstimator
from math import factorial
from sklearn.metrics import pairwise_distances


def space_perms(N,B=0,connected=0):
    
    if (numpy.max(N)>10)&(B==0):
        print('Warning, to many perms')
        B=1000
    SPACE = None
    for j in range(len(N)):
        n=N[j]
        if (j==0) | (connected==0):
            if B==0:
                Space=numpy.random.permutation(range(n))
                Space=Space[::-1,::-1]
            else:
                Space=numpy.zeros((B+1,n))
                for i in range(B):
                    Space[i,:] = numpy.random.permutation(n)
                Space[B,:] = range(n)
        
        if SPACE is None:
            SPACE = Space
        else:
            SPACE = numpy.concatenate((SPACE,Space+SPACE.shape[1]),axis=1)

    return SPACE
 
#######################################################################
class RiemannDistanceMetric(DistanceMetric):

    def __init__(self,metric='riemann'):
        self.metric = metric
    
    def pairwise(self,X,Y=None):
        
        if Y is None:
            Ntx,_,_ = X.shape
            dist = numpy.zeros((Ntx,Ntx))
            for i in range(Ntx):
                for j in range(i+1,Ntx):
                    dist[i,j] = distance(X[i],X[j])
            dist += dist.T
        else:
            Ntx,_,_ = X.shape
            Nty,_,_ = Y.shape
            dist = numpy.zeros((Ntx,Nty))
            for i in range(Ntx):
                for j in range(i,Nty):
                    dist[i,j] = distance(X[i],Y[j])
        return dist
        
    def get_metric(self):
        return "riemann"

#######################################################################
class SeparabilityIndex(BaseEstimator):
    def __init__(self,method='',metric='riemann'):
        self.method = method
        self.metric = 'riemann'
    
    def fit(self,X,y=None):
        self.pairs = RiemannDistanceMetric(self.metric).pairwise(X)
        
    def score(self,y):
        groups = numpy.unique(y)
        a = len(groups)
        Ntx = len(y)
        self.a = a
        self.Ntx = Ntx
        self._SST = (self.pairs**2).sum()/(2*Ntx)
        pattern = numpy.zeros((Ntx,Ntx))
        for g in groups:
            pattern += numpy.outer(y==g,y==g)/(numpy.float(numpy.sum(y==g)))
        
        self._SSW = ((self.pairs**2)*(pattern)).sum()/2
        
        self._SSA = self._SST-self._SSW
        
        
        self._F = (self._SSA/(a-1))/(self._SSW/(Ntx-a))   
        
        return self._F
#######################################################################
class SeparabilityIndexTwoFactor(BaseEstimator):
    def __init__(self,method='',metric='riemann'):
        self.method = method
        self.metric = 'riemann'
    
    def fit(self,X,y=None):
        self.pairs = RiemannDistanceMetric(self.metric).pairwise(X)
        #self.pairs = pairwise_distances(X,X)
    def score(self,fact1,fact2):
        groups1 = numpy.unique(fact1)
        groups2 = numpy.unique(fact2)
        
        a1 = len(groups1)
        a2 = len(groups2)
        Ntx = len(fact1)
        self.a1 = a1
        self.a2 = a2
        self.Ntx = Ntx
        self._SST = (self.pairs**2).sum()/(2*Ntx)
        
        #first factor
        pattern = numpy.zeros((Ntx,Ntx))
        y = fact1
        for g in groups1:
            pattern += numpy.outer(y==g,y==g)/(numpy.float(numpy.sum(y==g)))
        
        self._SSW1 = ((self.pairs**2)*(pattern)).sum()/2
        
        
        #second factor
        pattern = numpy.zeros((Ntx,Ntx))
        y = fact2
        for g in groups2:
            pattern += numpy.outer(y==g,y==g)/(numpy.float(numpy.sum(y==g)))
        
        self._SSW2 = ((self.pairs**2)*(pattern)).sum()/2
        
        # Co factor
        pattern = numpy.zeros((Ntx,Ntx))
        for g1 in groups1:
            for g2 in groups2:
                truc = (fact1==g1)&(fact2==g2)
                pattern += numpy.outer(truc,truc)/(numpy.float(numpy.sum(truc)))
        
        self._SSi = ((self.pairs**2)*(pattern)).sum()/2
        
        #self._SS1 =  self._SSW2 - self._SSi 
        #self._SS2 =  self._SSW1 - self._SSi 
        
        self._SS1 =  self._SST - self._SSW1 
        self._SS2 =  self._SST - self._SSW2 
        
        #self._SSR = self._SST - self._SS1 - self._SS2
        self._SSR = self._SSi
        
        self._SS12 = self._SST - self._SS1 - self._SS2 - self._SSR
        #self._SS12 = self._SST -self._SSW2  -(self._SSR - self._SSi)
        
                
        self._F1 = (self._SS1/(a1-1))/(self._SSR/(Ntx-a1*a2))
        self._F2 = (self._SS2/(a2-1))/(self._SSR/(Ntx-a1*a2))
        self._F12 = (self._SS12/((a1-1)*(a2-1)))/(self._SSR/(Ntx-a1*a2))      
        
        return self._F1,self._F2,self._F12        
#######################################################################
class PermutationTest(BaseEstimator):
    def __init__(self,n_perms = 100,sep_index = SeparabilityIndex(),random_state=42):

        self.n_perms = n_perms
        self.random_state = random_state
        self.SepIndex = sep_index
    

    def test(self,X,y):
        numpy.random.seed(self.random_state)
        self.SepIndex.fit(X)
        self.F = numpy.zeros(self.n_perms+1)
        for i in range(self.n_perms):
            perms = numpy.random.permutation(y)
            self.F[i+1] = self.SepIndex.score(perms)
        
        self.F[0] = self.SepIndex.score(y)
        
        self.p_value = (self.F[0]<=self.F).sum()/numpy.float(self.n_perms)

        return self.p_value,self.F
    
    def n_combinaisons(self,y):
        a = len(numpy.unique(y))
        n = len(y)/a
        factorial(a*n)/(factorial(a)*(factorial(n))**a)
    
    def generate_perms(self,y,n_perms):
        pass
        
    
    def summary(self):
        sep = self.SepIndex
        a = sep.a
        Ntx = sep.Ntx
        
        df = [(a-1),Ntx-a,Ntx-1]
        SS = [sep._SSA,sep._SSW,sep._SST]
        MS = numpy.array(SS)/numpy.array(df)
        F  = [self.F[0],numpy.nan,numpy.nan]
        p  = [self.p_value,numpy.nan,numpy.nan]
        
        cols = ['df','SS','MS','F','p-value']
        index = ['Labels','Residual','Total']
        
        data = numpy.array([df,SS,MS,F,p]).T
        
        res = pandas.DataFrame(data,index=index,columns=cols)
        return res
    
    def plot(self,nbins=100,range=None):
        plt.plot([self.F[0],self.F[0]],[0,100],'--r',lw=2)
        h = hist(self.F,nbins,range)
        plt.xlabel('F-value')
        plt.ylabel('Count')
        plt.grid()
        
#######################################################################
class PermutationTestTwoWay(BaseEstimator):
    def __init__(self,n_perms = 100,sep_index = SeparabilityIndexTwoFactor(),random_state=42):

        self.n_perms = n_perms
        self.random_state = random_state
        self.SepIndex = sep_index
    

    def test(self,X,factor1,factor2,names=None):
        numpy.random.seed(self.random_state)
        self.SepIndex.fit(X)
        self.names = names
        
        self.F = numpy.zeros((self.n_perms+1,3))
        for i in range(self.n_perms):
            self.F[i+1,:] = self.SepIndex.score(numpy.random.permutation(factor1),numpy.random.permutation(factor2))
        
        self.F[0,:] = self.SepIndex.score(factor1,factor2)
        
        self.p_value = (self.F[0,:]<=self.F).sum(axis=0)/numpy.float(self.n_perms)

        return self.p_value,self.F
    
    def summary(self):
        sep = self.SepIndex
        Ntx = sep.Ntx
        
        df = [sep.a1-1,sep.a2-1,(sep.a1-1)*(sep.a2-1),Ntx-sep.a1*sep.a2,Ntx-1]
        SS = [sep._SS1,sep._SS2,sep._SS12,sep._SSR,sep._SST]
        MS = numpy.array(SS)/numpy.array(df)
        F  = [self.F[0,0],self.F[0,1],self.F[0,2],numpy.nan,numpy.nan]
        p  = [self.p_value[0],self.p_value[1],self.p_value[2],numpy.nan,numpy.nan]
        
        cols = ['df','sum_sq','mean_sq','F','PR(>F)']
        if self.names is not None:
            index = [self.names[0],self.names[1],self.names[0] + ':' + self.names[1],'Residual','Total']
        else:
            index = ['Fact1','Fact2','Fact1:Fact2','Residual','Total']
        
        data = numpy.array([df,SS,MS,F,p]).T
        
        res = pandas.DataFrame(data,index=index,columns=cols)
        return res
    
    def plot(self):
        for i in range(3):
            plt.subplot(3,1,i+1)
            plt.plot([self.F[0,i],self.F[0,i]],[0,100],'--r',lw=2)
            h = hist(self.F[:,i],100)
            plt.xlabel('F-value')
            plt.ylabel('Count')
            plt.grid()        
        
        
        
        
        
        