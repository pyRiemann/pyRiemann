"""Estimation of covariance matrices."""
import numpy

from .spatialfilters import Xdawn
from .utils.covariance import (covariances, covariances_EP, cospectrum,
                               coherence)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import shrunk_covariance


def _nextpow2(i):
    """Find next power of 2."""
    n = 1
    while n < i:
        n *= 2
    return n


class Covariances(BaseEstimator, TransformerMixin):
    """Estimation of covariance matrix.

    Perform a simple covariance matrix estimation for each given trial.

    Parameters
    ----------
    estimator : string (default: 'scm')
        covariance matrix estimator. For regularization consider 'lwf' or 'oas'
        For a complete list of estimator, see `utils.covariance`.

    See Also
    --------
    ERPCovariances
    XdawnCovariances
    CospCovariances
    HankelCovariances
    """

    def __init__(self, estimator='scm'):
        """Init."""
        self.estimator = estimator

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.

        Returns
        -------
        self : Covariances instance
            The Covariances instance.
        """
        return self

    def transform(self, X):
        """Estimate covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.

        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of covariance matrices for each trials.
        """
        covmats = covariances(X, estimator=self.estimator)
        return covmats


class ERPCovariances(BaseEstimator, TransformerMixin):
    """Estimate special form covariance matrix for ERP.

    Estimation of special form covariance matrix dedicated to ERP processing.
    For each class, a prototyped response is obtained by average across trial :

    .. math::
        \mathbf{P} = \\frac{1}{N} \sum_i^N \mathbf{X}_i

    and a super trial is build using the concatenation of P and the trial X :

    .. math::
        \mathbf{\\tilde{X}}_i =  \left[
                                 \\begin{array}{c}
                                 \mathbf{P} \\\\
                                 \mathbf{X}_i
                                 \end{array}
                                 \\right]

    This super trial :math:`\mathbf{\\tilde{X}}_i` will be used for covariance
    estimation.
    This allows to take into account the spatial structure of the signal, as
    described in [1].

    Parameters
    ----------
    classes : list of int | None (default None)
        list of classes to take into account for prototype estimation.
        If None (default), all classes will be accounted.
    estimator : string (default: 'scm')
        covariance matrix estimator. For regularization consider 'lwf' or 'oas'
        For a complete list of estimator, see `utils.covariance`.
    svd : int | None (default None)
        if not none, the prototype responses will be reduce using a svd using
        the number of components passed in svd.

    See Also
    --------
    Covariances
    XdawnCovariances
    CospCovariances
    HankelCovariances

    References
    ----------
    [1] A. Barachant, M. Congedo ,"A Plug&Play P300 BCI Using Information
    Geometry", arXiv:1409.0107, 2014.

    [2] M. Congedo, A. Barachant, A. Andreev ,"A New generation of
    Brain-Computer Interface Based on Riemannian Geometry", arXiv: 1310.8115.
    2013.

    [3] A. Barachant, M. Congedo, G. Van Veen, C. Jutten, "Classification de
    potentiels evoques P300 par geometrie riemannienne pour les interfaces
    cerveau-machine EEG", 24eme colloque GRETSI, 2013.
    """

    def __init__(self, classes=None, estimator='scm', svd=None):
        """Init."""
        self.classes = classes
        self.estimator = estimator
        self.svd = svd

        if svd is not None:
            if not isinstance(svd, int):
                raise TypeError('svd must be None or int')

    def fit(self, X, y):
        """Fit.

        Estimate the Prototyped response for each classes.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial.

        Returns
        -------
        self : ERPCovariances instance
            The ERPCovariances instance.
        """
        if self.classes is not None:
            classes = self.classes
        else:
            classes = numpy.unique(y)

        self.P_ = []
        for c in classes:
            # Prototyped responce for each class
            P = numpy.mean(X[y == c, :, :], axis=0)

            # Apply svd if requested
            if self.svd is not None:
                U, s, V = numpy.linalg.svd(P)
                P = numpy.dot(U[:, 0:self.svd].T, P)

            self.P_.append(P)

        self.P_ = numpy.concatenate(self.P_, axis=0)
        return self

    def transform(self, X):
        """Estimate special form covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.

        Returns
        -------
        covmats : ndarray, shape (n_trials, n_c, n_c)
            ndarray of covariance matrices for each trials, with n_c the size
            of covmats equal to n_channels * (n_classes + 1) in case svd is
            None and equal to n_channels + n_classes * svd otherwise.
        """
        covmats = covariances_EP(X, self.P_, estimator=self.estimator)
        return covmats


class XdawnCovariances(BaseEstimator, TransformerMixin):
    """Estimate special form covariance matrix for ERP combined with Xdawn.

    Estimation of special form covariance matrix dedicated to ERP processing
    combined with Xdawn spatial filtering. This is similar to `ERPCovariances`
    but data are spatially filtered with `Xdawn`. A complete descrition of the
    method is available in [1].

    The advantage of this estimation is to reduce dimensionality of the
    covariance matrices efficiently.

    Parameters
    ----------
    nfilter: int (default 4)
        number of Xdawn filter per classes.
    applyfilters: bool (default True)
        if set to true, spatial filter are applied to the prototypes and the
        signals. When set to False, filters are applied only to the ERP prototypes
        allowing for a better generalization across subject and session at the
        expense of dimensionality increase. In that case, the estimation is
        similar to ERPCovariances with `svd=nfilter` but with more compact
        prototype reduction.
    classes : list of int | None (default None)
        list of classes to take into account for prototype estimation.
        If None (default), all classes will be accounted.
    estimator : string (default: 'scm')
        covariance matrix estimator. For regularization consider 'lwf' or 'oas'
        For a complete list of estimator, see `utils.covariance`.
    xdawn_estimator : string (default: 'scm')
        covariance matrix estimator for xdawn spatial filtering.
    baseline_cov : baseline_cov : array, shape(n_chan, n_chan) | None (default)
        baseline_covariance for xdawn. see `Xdawn`.

    See Also
    --------
    ERPCovariances
    Xdawn

    References
    ----------
    [1] Barachant, A. "MEG decoding using Riemannian Geometry and Unsupervised
        classification."
    """

    def __init__(self,
                 nfilter=4,
                 applyfilters=True,
                 classes=None,
                 estimator='scm',
                 xdawn_estimator='scm',
                 baseline_cov=None):
        """Init."""
        self.applyfilters = applyfilters
        self.estimator = estimator
        self.xdawn_estimator = xdawn_estimator
        self.classes = classes
        self.nfilter = nfilter
        self.baseline_cov = baseline_cov

    def fit(self, X, y):
        """Fit.

        Estimate spatial filters and prototyped response for each classes.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial.

        Returns
        -------
        self : XdawnCovariances instance
            The XdawnCovariances instance.
        """
        self.Xd_ = Xdawn(
            nfilter=self.nfilter,
            classes=self.classes,
            estimator=self.xdawn_estimator,
            baseline_cov=self.baseline_cov)
        self.Xd_.fit(X, y)
        self.P_ = self.Xd_.evokeds_
        return self

    def transform(self, X):
        """Estimate xdawn covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.

        Returns
        -------
        covmats : ndarray, shape (n_trials, n_c, n_c)
            ndarray of covariance matrices for each trials.
        """
        if self.applyfilters:
            X = self.Xd_.transform(X)

        covmats = covariances_EP(X, self.P_, estimator=self.estimator)
        return covmats


###############################################################################


class CospCovariances(BaseEstimator, TransformerMixin):
    """Estimation of cospectral covariance matrix.

    Covariance estimation in the frequency domain. this method will return a
    4-d array with a covariance matrice estimation for each trial and in each
    frequency bin of the FFT.

    Parameters
    ----------
    window : int (default 128)
        The lengt of the FFT window used for spectral estimation.
    overlap : float (default 0.75)
        The percentage of overlap between window.
    fmin : float | None , (default None)
        the minimal frequency to be returned.
    fmax : float | None , (default None)
        The maximal frequency to be returned.
    fs : float | None, (default None)
        The sampling frequency of the signal.

    See Also
    --------
    Covariances
    HankelCovariances
    Coherences
    """

    def __init__(self, window=128, overlap=0.75, fmin=None, fmax=None,
                 fs=None):
        """Init."""
        self.window = _nextpow2(window)
        self.overlap = overlap
        self.fmin = fmin
        self.fmax = fmax
        self.fs = fs

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.

        Returns
        -------
        self : CospCovariances instance
            The CospCovariances instance.
        """
        return self

    def transform(self, X):
        """Estimate the cospectral covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.

        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
            ndarray of covariance matrices for each trials and for each
            frequency bin.
        """
        Nt, Ne, _ = X.shape
        out = []

        for i in range(Nt):
            S = cospectrum(
                X[i],
                window=self.window,
                overlap=self.overlap,
                fmin=self.fmin,
                fmax=self.fmax,
                fs=self.fs)
            out.append(S)

        return numpy.array(out)


class Coherences(CospCovariances):
    """Estimation of coherences matrix.

    Coherence matrix estimation. this method will return a
    4-d array with a coherence matrice estimation for each trial and in each
    frequency bin of the FFT.

    The estimation of coherence matrix is done with matplotlib cohere function.

    Parameters
    ----------
    window : int (default 128)
        The lengt of the FFT window used for spectral estimation.
    overlap : float (default 0.75)
        The percentage of overlap between window.
    fmin : float | None , (default None)
        the minimal frequency to be returned.
    fmax : float | None , (default None)
        The maximal frequency to be returned.
    fs : float | None, (default None)
        The sampling frequency of the signal.

    See Also
    --------
    Covariances
    HankelCovariances
    CospCovariances
    """

    def transform(self, X):
        """Estimate the coherences matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.

        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
            ndarray of coherence matrices for each trials and for each
            frequency bin.
        """
        Nt, Ne, _ = X.shape
        out = []

        for i in range(Nt):
            S = coherence(
                X[i],
                window=self.window,
                overlap=self.overlap,
                fmin=self.fmin,
                fmax=self.fmax,
                fs=self.fs)
            out.append(S)

        return numpy.array(out)


class HankelCovariances(BaseEstimator, TransformerMixin):
    """Estimation of covariance matrix with time delayed hankel matrices.

    This estimation is usefull to catch spectral dynamics of the signal,
    similarly to the CSSP method. It is done by concatenating time delayed
    version of the signal before covariance estimation.

    Parameters
    ----------
    delays: int, list of int (default, 2)
        the delays to apply for the hankel matrices. if Int, it use a range of
        delays up to the given value. A list of int can be given.
    estimator : string (default: 'scm')
        covariance matrix estimator. For regularization consider 'lwf' or 'oas'
        For a complete list of estimator, see `utils.covariance`.

    See Also
    --------
    Covariances
    ERPCovariances
    XdawnCovariances
    CospCovariances
    """

    def __init__(self, delays=4, estimator='scm'):
        """Init."""
        self.delays = delays
        self.estimator = estimator

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.

        Returns
        -------
        self : Covariances instance
            The Covariances instance.
        """
        return self

    def transform(self, X):
        """Estimate the hankel covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.

        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of covariance matrices for each trials.
        """

        if isinstance(self.delays, int):
            delays = range(1, self.delays)
        else:
            delays = self.delays

        X2 = []

        for x in X:
            tmp = x
            for d in delays:
                tmp = numpy.r_[tmp, numpy.roll(x, d, axis=-1)]
            X2.append(tmp)
        X2 = numpy.array(X2)
        covmats = covariances(X2, estimator=self.estimator)
        return covmats


class Shrinkage(BaseEstimator, TransformerMixin):
    """Regularization of covariance matrices by shrinkage

    This transformer apply a shrinkage regularization to any covariance matrix.
    It directly use the `shrunk_covariance` function from scikit learn, applied
    on each trial.

    Parameters
    ----------
    shrinkage: float, (default, 0.1)
        Coefficient in the convex combination used for the computation of the
        shrunk estimate. must be between 0 and 1

    Notes
    -----
    .. versionadded:: 0.2.5
    """

    def __init__(self, shrinkage=0.1):
        """Init."""
        self.shrinkage = shrinkage

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of Target data.
        y : ndarray shape (n_trials,)
            Labels corresponding to each trial, not used.

        Returns
        -------
        self : Shrinkage instance
            The Shrinkage instance.
        """
        return self

    def transform(self, X):
        """Shrink and return the covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of covariances matrices

        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of covariance matrices for each trials.
        """

        covmats = numpy.zeros_like(X)

        for ii, x in enumerate(X):
            covmats[ii] = shrunk_covariance(x, self.shrinkage)

        return covmats
