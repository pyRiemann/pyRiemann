"""Estimation of SPD matrices."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import shrunk_covariance
from sklearn.metrics.pairwise import pairwise_kernels

from .spatialfilters import Xdawn
from .utils.covariance import (covariances, covariances_EP, cross_spectrum,
                               coherence, block_covariances)


def _nextpow2(i):
    """Find next power of 2."""
    n = 1
    while n < i:
        n *= 2
    return n


class Covariances(TransformerMixin, BaseEstimator):
    """Estimation of covariance matrices.

    Perform a simple covariance matrix estimation for each given input.

    Parameters
    ----------
    estimator : string, default="scm"
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.
    **kwds : dict
        Any further parameters are passed directly to the covariance estimator.

    See Also
    --------
    ERPCovariances
    XdawnCovariances
    """

    def __init__(self, estimator="scm", **kwds):
        """Init."""
        self.estimator = estimator
        self.kwds = kwds

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
        X_new : ndarray, shape (n_matrices, n_channels, n_channels)
            Covariance matrices.
        """
        covmats = covariances(X, estimator=self.estimator, **self.kwds)
        return covmats

    def fit_transform(self, X, y=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels, n_channels)
            Covariance matrices.
        """
        return self.fit(X, y).transform(X)


class ERPCovariances(TransformerMixin, BaseEstimator):
    r"""Estimate special form covariance matrices for ERP.

    Estimation of special form covariance matrix dedicated to event-related
    potentials (ERP) processing.
    For each class, a prototyped response is obtained by average across trials:

    .. math::
        \mathbf{P} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{X}_i

    and a super trial is built using the concatenation of :math:`\mathbf{P}`
    and the trial :math:`\mathbf{X}_i`:

    .. math::
        \mathbf{\tilde{X}}_i = \left[ \begin{array}{c} \mathbf{P} \\
        \mathbf{X}_i \end{array} \right]

    This super trial :math:`\mathbf{\tilde{X}}_i` will be used for covariance
    estimation.
    This allows to take into account the spatial structure of the signal, as
    described in [1]_.

    Parameters
    ----------
    classes : list of int | None, default=None
        List of classes to take into account for prototype estimation.
        If None, all classes will be accounted.
    estimator : string, default="scm"
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.
    svd : int | None, default=None
        If not None, number of components of SVD used to reduce prototype
        responses.
    **kwds : dict
        Any further parameters are passed directly to the covariance estimator.

    Attributes
    ----------
    P_ : ndarray, shape (n_components, n_times)
        If fit, prototyped responses for each class, where n_components is
        equal to n_classes x n_channels if ``svd`` is None,
        and to n_classes x min(``svd``, n_channels) otherwise.

    See Also
    --------
    Covariances
    XdawnCovariances

    References
    ----------
    .. [1] `A Plug and Play P300 BCI Using Information Geometry
        <https://arxiv.org/abs/1409.0107>`_
        A. Barachant, M. Congedo. Research report, 2014.
    .. [2] `A New generation of Brain-Computer Interface Based on Riemannian
        Geometry
        <https://hal.archives-ouvertes.fr/hal-00879050>`_
        M. Congedo, A. Barachant, A. Andreev. Research report, 2013.
    .. [3] `Classification de potentiels evoques P300 par geometrie
        riemannienne pour les interfaces cerveau-machine EEG
        <https://hal.archives-ouvertes.fr/hal-00877447>`_
        A. Barachant, M. Congedo, G. van Veen, and C. Jutten, 24eme colloque
        GRETSI, 2013.
    """

    def __init__(self, classes=None, estimator="scm", svd=None, **kwds):
        """Init."""
        self.classes = classes
        self.estimator = estimator
        self.svd = svd
        self.kwds = kwds

    def fit(self, X, y):
        """Fit.

        Estimate the prototyped responses for each class.

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
                raise TypeError("svd must be None or int")
        if self.classes is not None:
            classes = self.classes
        else:
            classes = np.unique(y)

        self.P_ = []
        for c in classes:
            # Prototyped response for each class
            P = np.mean(X[y == c], axis=0)

            # Apply svd if requested
            if self.svd is not None:
                U, _, _ = np.linalg.svd(P)
                P = U[:, 0:self.svd].T @ P

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
        X_new : ndarray, shape (n_matrices, n_components, n_components)
            Covariance matrices for ERP, where the size of matrices
            n_components is equal to (1 + n_classes) x n_channels if ``svd``
            is None, and to n_channels + n_classes x min(``svd``, n_channels)
            otherwise.
        """
        covmats = covariances_EP(
            X,
            self.P_,
            estimator=self.estimator,
            **self.kwds
        )
        return covmats

    def fit_transform(self, X, y):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_components, n_components)
            Covariance matrices for ERP, where the size of matrices
            n_components is equal to (1 + n_classes) x n_channels if ``svd``
            is None, and to n_channels + n_classes x min(``svd``, n_channels)
            otherwise.
        """
        return self.fit(X, y).transform(X)


class XdawnCovariances(TransformerMixin, BaseEstimator):
    """Estimate special form covariance matrices for ERP combined with Xdawn.

    Estimation of special form covariance matrix dedicated to ERP processing
    combined with Xdawn spatial filtering [1]_.
    This is similar to :class:`pyriemann.estimation.ERPCovariances` but data
    are spatially filtered with :class:`pyriemann.spatialfilters.Xdawn`.

    The advantage of this estimation is to reduce dimensionality of the
    covariance matrices supervisely.

    Parameters
    ----------
    nfilter : int, default=4
        Number of Xdawn filters per class.
    applyfilters : bool, default=True
        If set to true, spatial filter are applied to the prototypes and the
        signals. When set to False, filters are applied only to the ERP
        prototypes allowing for a better generalization across subject and
        session at the expense of dimensionality increase. In that case, the
        estimation is similar to :class:`pyriemann.estimation.ERPCovariances`
        with ``svd=nfilter`` but with more compact prototype reduction.
    classes : list of int | None, default=None
        list of classes to take into account for prototype estimation.
        If None, all classes will be accounted.
    estimator : string, default="scm"
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.
    xdawn_estimator : string, default="scm"
        Covariance matrix estimator for :class:`pyriemann.spatialfilters.Xdawn`
        spatial filtering.
        Should be regularized using "lwf" or "oas", see
        :func:`pyriemann.utils.covariance.covariances`.
    baseline_cov : ndarray, shape (n_channels, n_channels) | None, default=None
        Baseline covariance for :class:`pyriemann.spatialfilters.Xdawn`
        spatial filtering.
    **kwds : dict
        Any further parameters are passed directly to the covariance estimator.

    Attributes
    ----------
    P_ : ndarray, shape (n_classes x min(n_channels, n_filters), n_times)
        If fit, the evoked response for each event type, concatenated.

    See Also
    --------
    ERPCovariances
    Xdawn

    References
    ----------
    .. [1] `MEG decoding using Riemannian Geometry and
        Unsupervised classification
        <https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.713.5131>`_
        A. Barachant. Technical report with the solution of the DecMeg 2014
        challenge.
    """

    def __init__(
        self,
        nfilter=4,
        applyfilters=True,
        classes=None,
        estimator="scm",
        xdawn_estimator="scm",
        baseline_cov=None,
        **kwds
    ):
        """Init."""
        self.applyfilters = applyfilters
        self.estimator = estimator
        self.xdawn_estimator = xdawn_estimator
        self.classes = classes
        self.nfilter = nfilter
        self.baseline_cov = baseline_cov
        self.kwds = kwds

    def fit(self, X, y):
        """Fit.

        Estimate spatial filters and prototyped response for each class.

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
            baseline_cov=self.baseline_cov,
        )
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
        X_new : ndarray, shape (n_matrices, n_components, n_components)
            Covariance matrices filtered by Xdawn, where n_components is equal
            to 2 x n_classes x min(n_channels, ``nfilter``)
            if ``applyfilters`` is True, and
            to n_channels + n_classes x min(n_channels, ``nfilter``) otherwise.
        """
        if self.applyfilters:
            X = self.Xd_.transform(X)

        covmats = covariances_EP(
            X,
            self.P_,
            estimator=self.estimator,
            **self.kwds
        )
        return covmats

    def fit_transform(self, X, y):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_components, n_components)
            Covariance matrices filtered by Xdawn, where n_components is equal
            to 2 x n_classes x min(n_channels, ``nfilter``)
            if ``applyfilters`` is True, and
            to n_channels + n_classes x min(n_channels, ``nfilter``) otherwise.
        """
        return self.fit(X, y).transform(X)


class BlockCovariances(Covariances):
    """Estimation of block covariance matrices.

    Perform a block covariance estimation for each given matrix. The
    resulting matrices are block diagonal matrices.

    The blocks on the diagonal are calculated as individual covariance
    matrices for a subset of channels using the given the estimator.
    Varying block sized possible by passing a list to allow incorporation
    of different modalities with different number of channels (e.g. EEG,
    ECoG, LFP, EMG) with their own respective covariance matrices.

    Parameters
    ----------
    block_size : int | array-like of int
        Sizes of individual blocks given as int for same-size block, or list
        for varying block sizes.
    estimator : string, default="scm"
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.
    **kwds : dict
        Any further parameters are passed directly to the covariance estimator.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    Covariances
    """

    def __init__(self, block_size, estimator="scm", **kwds):
        """Init."""
        self.estimator = estimator
        self.block_size = block_size
        self.kwds = kwds

    def transform(self, X):
        """Estimate block covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels, n_channels)
            Covariance matrices.
        """
        n_matrices, n_channels, n_times = X.shape

        blocks = self._check_block_size(self.block_size, n_channels)

        return block_covariances(X, blocks, self.estimator, **self.kwds)

    @staticmethod
    def _check_block_size(block_size, n_channels):
        """Check validity of parameter block_size"""
        if isinstance(block_size, int):
            if n_channels % block_size != 0:
                raise ValueError(
                    f"Number of channels ({n_channels}) must be "
                    f"divisible by block size ({block_size})"
                )
            n_blocks = n_channels // block_size
            blocks = [block_size] * n_blocks

        elif isinstance(block_size, (list, np.ndarray)):
            if np.sum(block_size) != n_channels:
                raise ValueError(
                    "Sum of individual block sizes must match "
                    f"number of channels ({n_channels})"
                )
            blocks = block_size

        else:
            raise ValueError("Parameter block_size must be int or list.")

        return blocks


###############################################################################


class CrossSpectra(TransformerMixin, BaseEstimator):
    """Estimation of cross-spectral matrices.

    Complex cross-spectral matrices are HPD matrices estimated as the spectrum
    covariance in the frequency domain [1]_. It returns a 4-d array with a
    cross-spectral matrix for each input and in each frequency bin of the
    Fourier transform.

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
        If transformed, the frequencies associated to cross-spectra.
        None if ``fs`` is None.

    Notes
    -----
    .. versionadded:: 0.6

    See Also
    --------
    CoSpectra
    Coherences

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Cross-spectrum
    """

    def __init__(
        self,
        window=128,
        overlap=0.75,
        fmin=None,
        fmax=None,
        fs=None,
    ):
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
        self : CrossSpectra instance
            The CrossSpectra instance.
        """
        return self

    def transform(self, X):
        """Estimate cross-spectral matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels, n_channels, n_freqs)
            Cross-spectral matrices for each input and for each frequency bin.
        """
        X_new = []

        for x in X:
            x_new, freqs = cross_spectrum(
                x,
                window=self.window,
                overlap=self.overlap,
                fmin=self.fmin,
                fmax=self.fmax,
                fs=self.fs,
            )
            X_new.append(x_new)
        self.freqs_ = freqs

        return np.array(X_new)

    def fit_transform(self, X, y=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels, n_channels, n_freqs)
            Cross-spectral matrices for each input and for each frequency bin.
        """
        return self.fit(X, y).transform(X)


class CoSpectra(CrossSpectra):
    """Estimation of co-spectral matrices.

    Co-spectral matrices are SPD matrices estimated as the real part of the
    :class:`pyriemann.estimation.CrossSpectra`. It returns a 4-d array with a
    co-spectral matrix for each input and in each frequency bin of the
    Fourier transform.

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
    CrossSpectra
    Coherences
    """

    def transform(self, X):
        """Estimate co-spectral matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels, n_channels, n_freqs)
            Co-spectral matrices for each input and for each frequency bin.
        """
        X_new = super().transform(X)
        return X_new.real


class Coherences(CoSpectra):
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
    coh : {"ordinary", "instantaneous", "lagged", "imaginary"}, \
            default="ordinary"
        The coherence type:

        * "ordinary" for the ordinary coherence, defined in Eq.(22) of [1]_;
          this normalization of cross-spectral matrices captures both in-phase
          and out-of-phase correlations. However it is inflated by the
          artificial in-phase (zero-lag) correlation engendered by volume
          conduction.
        * "instantaneous" for the instantaneous coherence, Eq.(26) of [1]_,
          capturing only in-phase correlation.
        * "lagged" for the lagged-coherence, Eq.(28) of [1]_, capturing only
          out-of-phase correlation (not defined for DC and Nyquist bins).
        * "imaginary" for the imaginary coherence [2]_, Eq.(0.16) of [3]_,
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
    CrossSpectra
    TimeDelayCovariances

    References
    ----------
    .. [1] `Instantaneous and lagged measurements of linear
        and nonlinear dependence between groups of multivariate time series:
        frequency decomposition
        <https://arxiv.org/ftp/arxiv/papers/0711/0711.1455.pdf>`_
        R. Pascual-Marqui. Technical report, 2007.
    .. [2] `Identifying true brain interaction from EEG data using the
        imaginary part of coherency
        <https://doi.org/10.1016/j.clinph.2004.04.029>`_
        G. Nolte, O. Bai, L. Wheaton, Z. Mari, S. Vorbach, M. Hallett.
        Clinical Neurophysioly, Volume 115, Issue 10, October 2004,
        Pages 2292-2307
    .. [3] `Non-Parametric Synchronization Measures used in EEG
        and MEG
        <https://hal.archives-ouvertes.fr/hal-01868538v2>`_
        M. Congedo. Technical Report, 2018.
    """

    def __init__(
        self,
        window=128,
        overlap=0.75,
        fmin=None,
        fmax=None,
        fs=None,
        coh="ordinary",
    ):
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
        X_new : ndarray, shape (n_matrices, n_channels, n_channels, n_freqs)
            Squared coherence matrices for each input and for each frequency
            bin.
        """
        X_new = []

        for x in X:
            x_new, freqs = coherence(
                x,
                window=self.window,
                overlap=self.overlap,
                fmin=self.fmin,
                fmax=self.fmax,
                fs=self.fs,
                coh=self.coh,
            )
            X_new.append(x_new)
        self.freqs_ = freqs

        return np.array(X_new)


class TimeDelayCovariances(TransformerMixin, BaseEstimator):
    """Estimation of covariance matrices with time delay matrices.

    Time delay covariance matrices are useful to catch spectral dynamics of
    the signal, similarly to the CSSP method [1]_. It is done by concatenating
    time delayed version of the signal before covariance estimation.

    Parameters
    ----------
    delays : int | list of int, default=4
        The delays to apply for the Hankel matrices. If int, it use a range
        of delays up to the given value. A list of int can be given.
    estimator : string, default="scm"
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.
    **kwds : dict
        Any further parameters are passed directly to the covariance estimator.

    Attributes
    ----------
    Xtd_ : ndarray, shape (n_matrices, n_channels x n_delays, n_times)
        Time delay multi-channel time-series, where n_delays is equal to:
        ``delays`` when it is a int, and 1 + len(``delays``) when it is a list.

    See Also
    --------
    Covariances
    ERPCovariances
    CoSpectra

    References
    ----------
    .. [1] `Spatio-spectral filters for improving the classification of single
        trial EEG
        <http://doc.ml.tu-berlin.de/bbci/publications/LemBlaCurMue05.pdf>`_
        S. Lemm, B. Blankertz, B. Curio, K-R. Muller. IEEE Transactions on
        Biomedical Engineering 52(9), 1541-1548, 2005.
    """

    def __init__(self, delays=4, estimator="scm", **kwds):
        """Init."""
        self.delays = delays
        self.estimator = estimator
        self.kwds = kwds

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
        self : TimeDelayCovariances instance
            The TimeDelayCovariances instance.
        """
        return self

    def transform(self, X):
        """Estimate the time delay covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels x n_delays, \
                n_channels x n_delays)
            Time delay covariance matrices, where n_delays is equal to:
            ``delays`` when it is a int, and 1 + len(``delays``) when it is a
            list.
        """

        if isinstance(self.delays, int):
            delays = range(1, self.delays)
        elif isinstance(self.delays, list):
            delays = self.delays
        else:
            raise ValueError("delays must be an integer or a list")

        Xtd = [X]
        for d in delays:
            Xtd.append(np.roll(X, d, axis=-1))
        self.Xtd_ = np.concatenate(Xtd, axis=-2)

        covmats = covariances(self.Xtd_, estimator=self.estimator, **self.kwds)
        return covmats

    def fit_transform(self, X, y=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels x n_delays, \
                n_channels x n_delays)
            Time delay covariance matrices, where n_delays is equal to:
            ``delays`` when it is a int, and 1 + len(``delays``) when it is a
            list.
        """
        return self.fit(X, y).transform(X)


###############################################################################


ker_est_functions = [
    "linear", "poly", "polynomial", "rbf", "laplacian", "cosine"
]


class Kernels(TransformerMixin, BaseEstimator):
    r"""Estimation of kernel matrices between channels of time series.

    Perform a kernel matrix estimation for each given time series, evaluating a
    kernel function between each pair of channels (rather than between pairs of
    time samples) and allowing to extract nonlinear channel relationship [1]_.

    For an input time series :math:`X \in \mathbb{R}^{c \times t}`, composed of
    :math:`c` channels and :math:`t` time samples, kernel function
    :math:`\kappa()` is computed between channels :math:`i` and :math:`j`:

    .. math::
        K_{i,j} = \kappa \left( X[i], X[j] \right)

    Linear kernel is related to :class:`pyriemann.estimation.Covariances` [1]_,
    but this class allows to generalize to nonlinear relationships.

    Parameters
    ----------
    metric : string, default="linear"
        Metric to use when computing kernel function between channels [2]_:
        "linear", "poly", "polynomial", "rbf", "laplacian", "cosine".
    n_jobs : int, default=None
        Number of jobs to use for the computation [2]_. This works by
        breaking down the pairwise matrix into n_jobs even slices and computing
        them in parallel.
    **kwds : dict
        Any further parameters are passed directly to the kernel function [2]_.

    See Also
    --------
    Covariances

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] `Beyond Covariance: Feature Representation with Nonlinear Kernel
        Matrices
        <https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Wang_Beyond_Covariance_Feature_ICCV_2015_paper.pdf>`_
        L. Wang, J. Zhang, L. Zhou, C. Tang, W Li. ICCV, 2015.
    .. [2]
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
    """  # noqa

    def __init__(self, metric="linear", n_jobs=None, **kwds):
        """Init."""
        self.metric = metric
        self.n_jobs = n_jobs
        self.kwds = kwds

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
        self : Kernels instance
            The Kernels instance.
        """
        return self

    def transform(self, X):
        """Estimate kernel matrices from time series.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels, n_channels)
            Kernel matrices.
        """
        if self.metric not in ker_est_functions:
            raise TypeError("Unsupported metric for kernel estimation.")

        K = [
            pairwise_kernels(
                x,
                None,
                metric=self.metric,
                n_jobs=self.n_jobs,
                **self.kwds
            ) for x in X
        ]

        return np.asarray(K)

    def fit_transform(self, X, y=None):
        """Fit and transform in a single function.

        Estimate kernel matrices from time series.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels, n_channels)
            Kernel matrices.
        """
        return self.fit(X, y).transform(X)


###############################################################################


class Shrinkage(TransformerMixin, BaseEstimator):
    """Regularization of SPD/HPD matrices by shrinkage.

    This transformer applies a shrinkage regularization to SPD/HPD matrices.
    It directly uses the ``shrunk_covariance`` function from scikit-learn [1]_.

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
            Set of SPD/HPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : Shrinkage instance
            The Shrinkage instance.
        """
        return self

    def transform(self, X):
        """Shrink the SPD/HPD matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of shrunk SPD/HPD matrices.
        """
        Xnew = X.copy()

        for i, x in enumerate(X):
            Xnew[i].real = shrunk_covariance(x.real, self.shrinkage)

        return Xnew

    def fit_transform(self, X, y=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of shrunk SPD/HPD matrices.
        """
        return self.fit(X, y).transform(X)
