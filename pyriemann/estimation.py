import numpy

from .spatialfilters import Xdawn
from sklearn.base  import BaseEstimator, TransformerMixin

from sklearn.covariance import oas,ledoit_wolf,fast_mcd,empirical_covariance

### Mapping different estimator on the sklearn toolbox
def _lwf(X):
    C,_ = ledoit_wolf(X.T)
    return C

def _oas(X):
    C,_ = oas(X.T)
    return C

def _scm(X):
    return empirical_covariance(X.T)

def _mcd(X):
    _,C,_,_ = fast_mcd(X.T)
    return C

def _check_est(est):
    # Check estimator exist and return the correct function
    estimators = {
         'cov' : numpy.cov,
         'scm' : _scm,
         'lwf' : _lwf,
         'oas' : _oas,
         'mcd' : _mcd,
         'corr' : numpy.corrcoef
    }
    
    if callable(est):
        # All good (cross your fingers)
        pass
    elif est in estimators.keys():
        # Map the corresponding estimator
        est = estimators[est]
    else:
        # raise an error
        raise ValueError('%s is not an valid estimator ! Valid estimators are : %s or a callable function' % (est,(' , ').join(estimators.keys())))
    return est

def covariances(X,est='cov'):
    
    est = _check_est(est)
    Nt,Ne,Ns = X.shape
    covmats = numpy.zeros((Nt,Ne,Ne))
    for i in range(Nt):
        covmats[i,:,:] = est(X[i,:,:])
    return covmats

def covariances_EP(X,P,est = 'cov'):
    est = _check_est(est)
    Nt,Ne,Ns = X.shape
    Np,Ns = P.shape
    covmats = numpy.zeros((Nt,Ne+Np,Ne+Np))
    for i in range(Nt):
        covmats[i,:,:] = est(numpy.concatenate((P,X[i,:,:]),axis=0))
    return covmats
    

def eegtocov(sig,window=128,overlapp=0.5,padding = True):
    X = []
    if padding:
        padd = numpy.zeros((int(window/2),sig.shape[1]))
        sig = numpy.concatenate((padd,sig,padd),axis=0)
        
    Ns,Ne = sig.shape
    jump = int(window*overlapp)
    ix = 0
    while (ix+window<Ns):
        X.append(numpy.cov(sig[ix:ix+window,:].T))
        ix = ix+jump
        
    return numpy.array(X)

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def cospectrum(X,window=128,overlap=0.75,fmin=None,fmax=None,fs = None,phase_correction=False):
    
    Ne,Ns = X.shape
    number_freqs = int(window / 2)
    
    step = int((1.0-overlap)*window)
    step = max(1,step)
    
    
    number_windows = (Ns-window)/step + 1
    # pre-allocation of memory 
    fdata = numpy.zeros((number_windows,Ne,number_freqs),dtype=complex)
    win = numpy.hanning(window)
    
    ## Loop on all frequencies
    for window_ix in range(int(number_windows)):
    
        
        # time markers to select the data
        t1 = int(window_ix*step)  # marker of the beginning of the time window
        t2 = int(t1 + window)                           # marker of the end of the time window
        # select current window and apodize it   
        cdata = X[:,t1:t2] * win

        # FFT calculation
        fdata[window_ix,:,:] = numpy.fft.fft(cdata,n=window,axis=1)[:,0:number_freqs] 
        
        #if(phase_correction):
        #    fdata = fdata.*(exp(-sqrt(-1)*t1*( numpy.range(window) ).T/window*2*pi)*numpy.ones((1,Ne))
    
    # Adjust Frequency range to specified range (in case it is a parameter)
    if fmin is not None:        
        f = numpy.arange(0,1,1.0/number_freqs)*(fs/2.0)
        Fix = (f>=fmin) & (f<=fmax)
        fdata = fdata[:,:,Fix]
    
    #fdata = fdata.real
    Nf = fdata.shape[2]
    S = numpy.zeros((Ne,Ne,Nf),dtype=complex)
    
    for i in range(Nf):
        S[:,:,i] = numpy.dot(fdata[:,:,i].conj().T,fdata[:,:,i])/number_windows
    
    return S

###############################################################################
class Covariances(BaseEstimator,TransformerMixin):
    """ 
    compute the covariances matrices

    """    
    def __init__(self,estimator = 'scm'):
        self.estimator = estimator
        
    def fit(self,X,y=None):
        pass
    
    def transform(self,X):
            
        covmats = covariances(X,est=self.estimator)
        return covmats
    
    def fit_transform(self,X,y=None):
        return self.transform(X)

###############################################################################
class ERPCovariances(BaseEstimator,TransformerMixin):
    """ 
    Compute special form ERP cov mat

    """    
    def __init__(self,classes=None,estimator = 'scm'):
        self.classes = classes
        self.estimator = estimator
        
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
            
        covmats = covariances_EP(X,self.P,est=self.estimator)
        return covmats
    
    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)

###############################################################################
class XdawnCovariances(BaseEstimator,TransformerMixin):
    """ 
    Compute xdawn, project the signal and compute the covariances

    """    
    def __init__(self,nfilter=4,applyfilters=True,classes=None,estimator = 'scm'):
        self.Xd = Xdawn(nfilter=nfilter,classes=classes)
        self.applyfilters=applyfilters
        self.estimator = estimator
        
    def fit(self,X,y):
        self.Xd.fit(X,y)
        
    def transform(self,X):
        if self.applyfilters:
             X = self.Xd.transform(X)
            
        covmats = covariances_EP(X,self.Xd.P,est=self.estimator)
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
        
    def fit(self,X,y=None):
        pass
    
    def transform(self,X):
        
        Nt,Ne,_ = X.shape
        out = []
        
        for i in range(Nt):
            S = cospectrum(X[i],window=self._window,overlap=self._overlap,fmin=self._fmin,fmax=self._fmax,fs=self._fs)
            out.append(S.real)
        
        return numpy.array(out)
    
    def fit_transform(self,X,y=None):
        return self.transform(X)