"""
=================================
 Frequency Band Selection Helpers
=================================

This file contains helper functions for the frequency band selection example
"""

import numpy as np
from mne import Epochs, events_from_annotations

from pyriemann.estimation import Covariances
from pyriemann.classification import class_distinctiveness


def freq_selection_class_dis(raw, freq_band=(5., 35.), sub_band_width=4,
                             sub_band_step=2, alpha=0.4,
                             tmin=0.5, tmax=2.5,
                             picks=None, event_id=None,
                             cv=None,
                             return_class_dis=False, verbose=None):
    r"""Select optimal frequency band based on class distinctiveness measure.

    Optimal frequency band is selected by combining a filter bank with
    a heuristic based on class distinctiveness on the manifold [1]_:

    1. Filter training raw EEG data for each sub-band of a filter bank.
    2. Estimate covariance matrices of filtered EEG for each sub-band.
    3. Measure the class distinctiveness of each sub-band using the
      classDis metric.
    4. Find the optimal frequency band by starting from the sub-band
      with the largest classDis and expanding the selected frequency band
      as long as the classDis exceeds the threshold :math:`th`:

    .. math::
      th = max(classDis) - (max(classDis)−min(classDis)) × alpha


    Parameters
    ----------
    raw : Raw object
        An instance of Raw from MNE.
    freq_band : list, default=(5., 35.)
        Frequency band for band-pass filtering.
    sub_band_width : float, default=4.
        Frequency bandwidth of filter bank.
    sub_band_step : float, default=2.
        Step length of each filter bank.
    alpha : float, default=0.4
        Parameter to define an appropriate threshold for each individual.
    tmin, tmax : float, default=0.5, 2.5
        Start and end time of the epochs in seconds, relative to
        the time-locked event.
    picks : str | array_like | slice,  default=None
        Channels to include. Slices and lists of integers will be
        interpreted as channel indices.
        If None (default), all channels will pick.
    event_id : int | list of int | dict, default=None
        Id of the events to consider.

        - If dict, the keys can later be used to access associated
          events.
        - If int, a dict will be created with the id as string.
        - If a list, all events with the IDs specified in the list
          are used.
        - If None, all events will be used and a dict is created
          with string integer names corresponding to the event id integers.
    cv : cross-validation generator, default=None
        An instance of a cross validation iterator from sklearn.
    return_class_dis : bool, default=False
        Whether to return all_cv_class_dis value.
    verbose : bool, str, int, default=None
        Control verbosity of the logging output of filtering and .
        If None, use the default verbosity level.

    Returns
    -------
    all_cv_best_freq : list
        List of the selected frequency band for each hold of
        cross validation.
    all_cv_class_dis : list, optional
        List of class_dis value of each hold of cross validation.

    Notes
    -----
    .. versionadded:: 0.3.1

    References
    ----------
    .. [1] `Class-distinctiveness-based frequency band selection on the
       Riemannian manifold for oscillatory activity-based BCIs: preliminary
       results
       <https://hal.archives-ouvertes.fr/hal-03641137/>`_
       M. S. Yamamoto, F. Lotte, F. Yger, and S. Chevallier.
       44th Annual International Conference of the IEEE Engineering
       in Medicine & Biology Society (EMBC2022), 2022.
    """

    subband_fmin = list(np.arange(freq_band[0],
                                  freq_band[1] - sub_band_width + 1.,
                                  sub_band_step))
    subband_fmax = list(np.arange(freq_band[0] + sub_band_width,
                                  freq_band[1] + 1., sub_band_step))
    n_subband = len(subband_fmin)

    all_sub_band_cov = []

    for fmin, fmax in zip(subband_fmin, subband_fmax):
        cov_data, labels = _get_filtered_cov(raw, picks,
                                             event_id,
                                             fmin,
                                             fmax,
                                             tmin, tmax, verbose)
        all_sub_band_cov.append(cov_data)

    all_cv_best_freq = []
    all_cv_class_dis = []
    for i, (train_ind, test_ind) in enumerate(cv.split(all_sub_band_cov[0],
                                                       labels)):

        all_class_dis = []
        for ii in range(n_subband):
            class_dis = class_distinctiveness(
                all_sub_band_cov[ii][train_ind], labels[train_ind],
                exponent=1, metric='riemann', return_num_denom=False)
            all_class_dis.append(class_dis)
        all_cv_class_dis.append(all_class_dis)

        best_freq = _get_best_freq_band(all_class_dis, n_subband,
                                        subband_fmin, subband_fmax, alpha)

        all_cv_best_freq.append(best_freq)

    if return_class_dis:
        return all_cv_best_freq, all_cv_class_dis
    else:
        return all_cv_best_freq


def _get_filtered_cov(raw, picks, event_id, fmin, fmax, tmin, tmax, verbose):
    """Private function to apply band-pass filter and estimate
    covariance matrix."""

    best_raw_filter = raw.copy().filter(fmin, fmax, method='iir', picks=picks,
                                        verbose=verbose)

    events, _ = events_from_annotations(best_raw_filter, event_id=event_id,
                                        verbose=verbose)

    epochs = Epochs(
        best_raw_filter,
        events,
        event_id,
        tmin,
        tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
        verbose=verbose)
    labels = epochs.events[:, -1] - 2

    epochs_data = epochs.get_data(units="uV")

    cov_data = Covariances().transform(epochs_data)

    return cov_data, labels


def _get_best_freq_band(all_class_dis, n_subband, subband_fmin, subband_fmax,
                        alpha):
    """Private function to select frequency bands whose class dis value are
     higher than the user-specific threshold."""

    fmaxstart = np.argmax(all_class_dis)

    fmin = np.min(all_class_dis)
    fmax = np.max(all_class_dis)
    threshold_freq = fmax - (fmax - fmin) * alpha

    f0 = fmaxstart
    f1 = fmaxstart
    while f0 >= 1 and (all_class_dis[f0 - 1] >= threshold_freq):
        f0 = f0 - 1

    while f1 < n_subband - 1 and (all_class_dis[f1 + 1] >= threshold_freq):
        f1 = f1 + 1

    best_freq_f0 = subband_fmin[f0]
    best_freq_f1 = subband_fmax[f1]
    best_freq = [best_freq_f0, best_freq_f1]

    return best_freq
