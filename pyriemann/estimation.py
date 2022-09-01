"""Estimation of covariance matrices."""
import numpy as np

from .spatialfilters import Xdawn
from .utils.covariance import (covariances, covariances_EP, cospectrum,
                               coherence, block_covariances)
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

    Perform a simple covariance matrix estimation for each given input.

    Parameters
    ----------
    estimator : string, default=scm'
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.

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
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : None
            Not used, here for compatibility with sklearn API.

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
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        covmats : ndarray, shape (n_matrices, n_channels, n_channels)
            Covariance matrices.
        """
        covmats = covariances(X, estimator=self.estimator)
        return covmats


class ERPCovariances(BaseEstimator, TransformerMixin):
    r"""Estimate special form covariance matrix for ERP.

    Estimation of special form covariance matrix dedicated to ERP processing.
    For each class, a prototyped response is obtained by average across trial:

    .. math::
        \mathbf{P} = \frac{1}{N} \sum_i^N \mathbf{X}_i

    and a super trial is built using the concatenation of P and the trial X:

    .. math::
        \mathbf{\tilde{X}}_i =  \left[
                                \begin{array}{c}
                                \mathbf{P} \\
                                \mathbf{X}_i
                                \end{array}
                                \right]

    This super trial :math:`\mathbf{\tilde{X}}_i` will be used for covariance
    estimation.
    This allows to take into account the spatial structure of the signal, as
    described in [1]_.

    Parameters
    ----------
    classes : list of int | None, default=None
        List of classes to take into account for prototype estimation.
        If None, all classes will be accounted.
    estimator : string, default='scm'
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.
    svd : int | None, default=None
        If not None, the prototype responses will be reduce using a SVD using
        the number of components passed in SVD.

    See Also
    --------
    Covariances
    XdawnCovariances
    CospCovariances
    HankelCovariances

    References
    ----------
    .. [1] A. Barachant, M. Congedo ,"A Plug&Play P300 BCI Using Information
        Geometry", arXiv:1409.0107, 2014.

    .. [2] M. Congedo, A. Barachant, A. Andreev ,"A New generation of
        Brain-Computer Interface Based on Riemannian Geometry",
        arXiv:1310.8115, 2013.

    .. [3] A. Barachant, M. Congedo, G. Van Veen, C. Jutten, "Classification de
        potentiels evoques P300 par geometrie riemannienne pour les interfaces
        cerveau-machine EEG", 24eme colloque GRETSI, 2013.
    """

    def __init__(self, classes=None, estimator='scm', svd=None):
        """Init."""
        self.classes = classes
        self.estimator = estimator
        self.svd = svd

    def fit(self, X, y):
        """Fit.

        Estimate the Prototyped response for each classes.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.

        Returns
        -------
        self : ERPCovariances instance
            The ERPCovariances instance.
        """
        if self.svd is not None:
            if not isinstance(self.svd, int):
                raise TypeError('svd must be None or int')
        if self.classes is not None:
            classes = self.classes
        else:
            classes = np.unique(y)

        self.P_ = []
        for c in classes:
            # Prototyped responce for each class
            P = np.mean(X[y == c, :, :], axis=0)

            # Apply svd if requested
            if self.svd is not None:
                U, s, V = np.linalg.svd(P)
                P = np.dot(U[:, 0:self.svd].T, P)

            self.P_.append(P)

        self.P_ = np.concatenate(self.P_, axis=0)
        return self

    def transform(self, X):
        """Estimate special form covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        covmats : ndarray, shape (n_matrices, n_c, n_c)
            Covariance matrices for each input, with n_c the size
            of covmats equal to n_channels * (n_classes + 1) in case SVD is
            None and equal to n_channels + n_classes * svd otherwise.
        """
        covmats = covariances_EP(X, self.P_, estimator=self.estimator)
        return covmats


class XdawnCovariances(BaseEstimator, TransformerMixin):
    """Estimate special form covariance matrix for ERP combined with Xdawn.

    Estimation of special form covariance matrix dedicated to ERP processing
    combined with `Xdawn` spatial filtering.
    This is similar to :class:`pyriemann.estimation.ERPCovariances` but data
    are spatially filtered with :class:`pyriemann.spatialfilters.Xdawn`.
    A complete description of the method is available in [1]_.

    The advantage of this estimation is to reduce dimensionality of the
    covariance matrices efficiently.

    Parameters
    ----------
    nfilter: int, default=4
        Number of Xdawn filters per class.
    applyfilters: bool, default=True
        If set to true, spatial filter are applied to the prototypes and the
        signals. When set to False, filters are applied only to the ERP
        prototypes allowing for a better generalization across subject and
        session at the expense of dimensionality increase. In that case, the
        estimation is similar to :class:`pyriemann.estimation.ERPCovariances`
        with `svd=nfilter` but with more compact prototype reduction.
    classes : list of int | None, default=None
        list of classes to take into account for prototype estimation.
        If None, all classes will be accounted.
    estimator : string, default='scm'
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.
    xdawn_estimator : string, default='scm'
        Covariance matrix estimator for `Xdawn` spatial filtering.
        Should be regularized using 'lwf' or 'oas', see
        :func:`pyriemann.utils.covariance.covariances`.
    baseline_cov : array, shape (n_chan, n_chan) | None, default=None
        Baseline covariance for `Xdawn` spatial filtering,
        see :class:`pyriemann.spatialfilters.Xdawn`.

    See Also
    --------
    ERPCovariances
    Xdawn

    References
    ----------
    .. [1] Barachant, A. "MEG decoding using Riemannian Geometry and
        Unsupervised classification", 2014
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
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.

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
        """Estimate Xdawn covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        covmats : ndarray, shape (n_matrices, n_c, n_c)
            Covariance matrices.
        """
        if self.applyfilters:
            X = self.Xd_.transform(X)

        covmats = covariances_EP(X, self.P_, estimator=self.estimator)
        return covmats


class BlockCovariances(BaseEstimator, TransformerMixin):
    """Estimation of block covariance matrix.

    Perform a block covariance estimation for each given matrix. The
    resulting matrices are block diagonal matrices.

    The blocks on the diagonal are calculated as individual covariance
    matrices for a subset of channels using the given the estimator.
    Varying block sized possible by passing a list to allow incorporation
    of different modalities with different number of channels (e.g. EEG,
    ECoG, LFP, EMG) with their own respective covariance matrices.

    Parameters
    ----------
    block_size : int | list of int
        Sizes of individual blocks given as int for same-size block, or list
        for varying block sizes.
    estimator : string, default='scm'
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    Covariances
    """

    def __init__(self, block_size, estimator='scm'):
        """Init."""
        self.estimator = estimator
        self.block_size = block_size

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : BlockCovariances instance
            The BlockCovariances instance.
        """
        return self

    def transform(self, X):
        """Estimate block covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        covmats : ndarray, shape (n_matrices, n_channels, n_channels)
            Covariance matrices.
        """
        n_matrices, n_channels, n_times = X.shape

        if isinstance(self.block_size, int):
            n_blocks = n_channels // self.block_size
            blocks = [self.block_size for b in range(n_blocks)]

        elif isinstance(self.block_size, (list, np.ndarray)):
            blocks = self.block_size

        else:
            raise ValueError("Parameter block_size must be int or list.")

        return block_covariances(X, blocks, self.estimator)


###############################################################################


class CospCovariances(BaseEstimator, TransformerMixin):
    """Estimation of cospectral covariance matrix.

    Co-spectral matrices are the real part of complex cross-spectral matrices
    (see :func:`pyriemann.utils.covariance.cross_spectrum`), estimated as the
    spectrum covariance in the frequency domain. This method returns a 4-d
    array with a cospectral covariance matrix for each input and in each
    frequency bin of the FFT.

    Parameters
    ----------
    window : int, default=128
        The length of the FFT window used for spectral estimation.
    overlap : float, default=0.75
        The percentage of overlap between window.
    fmin : float | None, default=None
        The minimal frequency to be returned.
    fmax : float | None, default=None
        The maximal frequency to be returned.
    fs : float | None, default=None
        The sampling frequency of the signal.

    Attributes
    ----------
    freqs_ : ndarray, shape (n_freqs,)
        If transformed, the frequencies associated to cospectra.
        None if ``fs`` is None.

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
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : None
            Not used, here for compatibility with sklearn API.

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
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        covmats : ndarray, shape (n_matrices, n_channels, n_channels, n_freqs)
            Covariance matrices for each input and for each frequency bin.
        """
        out = []

        for i in range(len(X)):
            S, freqs = cospectrum(
                X[i],
                window=self.window,
                overlap=self.overlap,
                fmin=self.fmin,
                fmax=self.fmax,
                fs=self.fs)
            out.append(S)
        self.freqs_ = freqs

        return np.array(out)


class Coherences(CospCovariances):
    """Estimation of squared coherence matrices.

    Squared coherence matrices estimation [1]_. This method will return a 4-d
    array with a squared coherence matrix estimation for each input and in
    each frequency bin of the FFT.

    Parameters
    ----------
    window : int, default=128
        The length of the FFT window used for spectral estimation.
    overlap : float, default=0.75
        The percentage of overlap between window.
    fmin : float | None, default=None
        the minimal frequency to be returned.
    fmax : float | None, default=None
        The maximal frequency to be returned.
    fs : float | None, default=None
        The sampling frequency of the signal.
    coh : {'ordinary', 'instantaneous', 'lagged', 'imaginary'}, \
            default='ordinary'
        The coherence type:

        * 'ordinary' for the ordinary coherence, defined in Eq.(22) of [1]_;
          this normalization of cross-spectral matrices captures both in-phase
          and out-of-phase correlations. However it is inflated by the
          artificial in-phase (zero-lag) correlation engendered by volume
          conduction.
        * 'instantaneous' for the instantaneous coherence, Eq.(26) of [1]_,
          capturing only in-phase correlation.
        * 'lagged' for the lagged-coherence, Eq.(28) of [1]_, capturing only
          out-of-phase correlation (not defined for DC and Nyquist bins).
        * 'imaginary' for the imaginary coherence [2]_, Eq.(0.16) of [3]_,
          capturing out-of-phase correlation but still affected by in-phase
          correlation.

    Attributes
    ----------
    freqs_ : ndarray, shape (n_freqs,)
        If transformed, the frequencies associated to cospectra.
        None if ``fs`` is None.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    Covariances
    HankelCovariances
    CospCovariances

    References
    ----------
    .. [1] R. Pascual-Marqui, "Instantaneous and lagged measurements of linear
        and nonlinear dependence between groups of multivariate time series:
        frequency decomposition", arXiv, 2007.
        https://arxiv.org/ftp/arxiv/papers/0711/0711.1455.pdf

    .. [2] G. Nolte, O. Bai, L. Wheaton, Z. Mari, S. Vorbach, M. Hallett,
        "Identifying true brain interaction from EEG data using the imaginary
        part of coherency", Clin Neurophysiol, 2004.
        https://doi.org/10.1016/j.clinph.2004.04.029

    .. [3] Congedo, M. "Non-Parametric Synchronization Measures used in EEG
        and MEG", TechReport, 2018.
        https://hal.archives-ouvertes.fr/hal-01868538v2/document
    """

    def __init__(self, window=128, overlap=0.75, fmin=None, fmax=None,
                 fs=None, coh='ordinary'):
        """Init."""
        self.window = _nextpow2(window)
        self.overlap = overlap
        self.fmin = fmin
        self.fmax = fmax
        self.fs = fs
        self.coh = coh

    def transform(self, X):
        """Estimate the squared coherences matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        covmats : ndarray, shape (n_matrices, n_channels, n_channels, n_freqs)
            Squared coherence matrices for each input and for each frequency
            bin.
        """
        out = []

        for i in range(len(X)):
            S, freqs = coherence(
                X[i],
                window=self.window,
                overlap=self.overlap,
                fmin=self.fmin,
                fmax=self.fmax,
                fs=self.fs,
                coh=self.coh)
            out.append(S)
        self.freqs_ = freqs

        return np.array(out)


class HankelCovariances(BaseEstimator, TransformerMixin):
    """Estimation of covariance matrix with time delayed Hankel matrices.

    This estimation is usefull to catch spectral dynamics of the signal,
    similarly to the CSSP method. It is done by concatenating time delayed
    version of the signal before covariance estimation.

    Parameters
    ----------
    delays: int | list of int, default=4
        The delays to apply for the Hankel matrices. If `int`, it use a range
        of delays up to the given value. A list of int can be given.
    estimator : string, default='scm'
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.

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
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : Covariances instance
            The Covariances instance.
        """
        return self

    def transform(self, X):
        """Estimate the Hankel covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        covmats : ndarray, shape (n_matrices, n_channels, n_channels)
            Covariance matrices.
        """

        if isinstance(self.delays, int):
            delays = range(1, self.delays)
        else:
            delays = self.delays

        X2 = []

        for x in X:
            tmp = x
            for d in delays:
                tmp = np.r_[tmp, np.roll(x, d, axis=-1)]
            X2.append(tmp)
        X2 = np.array(X2)
        covmats = covariances(X2, estimator=self.estimator)
        return covmats


class Shrinkage(BaseEstimator, TransformerMixin):
    """Regularization of SPD matrices by shrinkage.

    This transformer applies a shrinkage regularization to any SPD matrix.
    It directly uses the `shrunk_covariance` function from scikit-learn [1]_,
    applied on each input.

    Parameters
    ----------
    shrinkage : float, default=0.1
        Coefficient in the convex combination used for the computation of the
        shrunk estimate. Must be between 0 and 1.

    Notes
    -----
    .. versionadded:: 0.2.5

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.covariance.shrunk_covariance.html
    """  # noqa

    def __init__(self, shrinkage=0.1):
        """Init."""
        self.shrinkage = shrinkage

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : Shrinkage instance
            The Shrinkage instance.
        """
        return self

    def transform(self, X):
        """Shrink and return the SPD matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        covmats : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of shrunk SPD matrices.
        """

        covmats = np.zeros_like(X)

        for ii, x in enumerate(X):
            covmats[ii] = shrunk_covariance(x, self.shrinkage)

        return covmats
