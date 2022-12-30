"""
====================================================================
Motor imagery classification with frequency band selection on the manifold
====================================================================

Find optimal frequency band using class distinctiveness measure on
the manifold and compare classification performance for Motor imagery
data to the baseline with no frequency band selection.


"""
# Authors: Maria Sayu Yamamoto <maria-sayu.yamamoto@universite-paris-saclay.fr>
#
# License: BSD (3-clause)


import numpy as np
from matplotlib import pyplot as plt

from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci

from sklearn.model_selection import cross_val_score, StratifiedKFold

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.frequencybandselection import freq_selection_class_dis

###############################################################################
# Set basic parameters and read data
# -------------------------------

tmin, tmax = 0.5, 2.5
event_id = dict(T1=2, T2=3)
subject = 1
runs = [4, 8, 12]  # motor imagery: left hand vs right hand

raw_files = [
    read_raw_edf(f, preload=True) for f in eegbci.load_data(subject, runs)
]
raw = concatenate_raws(raw_files)
picks = pick_types(
    raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
# subsample elecs
picks = picks[::2]

# cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

###############################################################################
# Baseline pipline without frequency band selection
# -------------------------------
#
# Apply band-pass filter using a wide frequency band
raw_filter = raw.copy().filter(5., 35., method='iir', picks=picks)

events, _ = events_from_annotations(raw_filter, event_id)

# Read epochs (train will be done only between 0.5 and 2.5 s)
epochs = Epochs(
    raw_filter,
    events,
    event_id,
    tmin,
    tmax,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
    verbose=False)
labels = epochs.events[:, -1] - 2

# Get epochs
epochs_data_baseline = 1e6 * epochs.get_data()

# Compute covariance matrices
cov_data_baseline = Covariances().transform(epochs_data_baseline)

# Set classifier
mdm = MDM(metric=dict(mean='riemann', distance='riemann'))

# Use scikit-learn Pipeline with cross_val_score function
scores = cross_val_score(mdm, cov_data_baseline, labels, cv=cv, n_jobs=1)
ave_baseline = np.mean(scores)

###############################################################################
# Pipline with a frequency band selection based on the class distinctiveness
# -------------------------------
#
# Define parameters of sub frequency bands

freq_band = [5., 35.]
sub_band_width = 4.
sub_band_step = 2.

# Select frequency band using training data for each hold of cv
all_cv_best_freq, all_cv_class_dis = \
    freq_selection_class_dis(raw, cv, freq_band, sub_band_width,
                             sub_band_step, tmin, tmax, picks,
                             event_id, return_class_dis=True)

all_cv_acc = []
for i, (train_ind, test_ind) in enumerate(cv.split(cov_data_baseline, labels)):
    # apply band-pass filter using the best frequency band
    best_raw_filter = raw.copy().filter(all_cv_best_freq[i][0],
                                        all_cv_best_freq[i][1],
                                        method='iir', picks=picks)

    events, _ = events_from_annotations(best_raw_filter, event_id)

    # read epochs (train will be done only between 0.5 and 2.5s)
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
        verbose=False)

    # get epochs
    epochs_data_train = 1e6 * epochs.get_data()

    # estimate covariance matrices
    cov_data = Covariances().transform(epochs_data_train)

    # classification with Minimum distance to mean
    mdm.fit(cov_data[train_ind], labels[train_ind])
    acc = mdm.score(cov_data[test_ind], labels[test_ind])
    all_cv_acc.append(acc)

ave_acc = np.array(all_cv_acc).mean()

print("Classification accuracy without frequency band selection: %f"
      % (ave_baseline))
print("Classification accuracy with frequency band selection: %f"
      % (ave_acc))

###############################################################################
# Plot result
# -------------------------------
#
# Plot the results

subband_fmin = list(np.arange(freq_band[0],
                              freq_band[1] - sub_band_width + 1.,
                              sub_band_step))
subband_fmax = list(np.arange(freq_band[0] + sub_band_width,
                              freq_band[1] + 1., sub_band_step))
nb_subband = len(subband_fmin)
all_cv_class_dis = np.array(all_cv_class_dis)
x = list(range(0, nb_subband, 1))
fig = plt.figure(figsize=(28, 8))
for cv in range(len(all_cv_class_dis)):
    freq_start = subband_fmin.index(all_cv_best_freq[cv][0])
    freq_end = subband_fmax.index(all_cv_best_freq[cv][1])

    plt.subplot(2, 3, cv + 1)
    plt.grid()
    plt.plot(x, all_cv_class_dis[cv], marker='o')
    plt.xticks(list(range(0, 14, 1)),
               ["[5, 9]", "[7, 11]", "[9, 13]", "[11, 15]", "[13, 17]",
                "[15, 19]", "[17, 21]", "[19, 23]", "[21, 25]",
                "[23, 27]", "[25, 29]", "[27, 31]", "[29, 33]", "[31, 35]"])
    plt.ylim(0.145, 0.180)

    plt.axvspan(freq_start, freq_end, color="orange", alpha=0.3,
                label='Selected frequency band')
    plt.ylabel('Class distinctiveness')
    plt.xlabel('Filter bank')
    plt.title('CV{:01d}'.format(cv + 1))
    plt.legend(loc='upper right')

fig.tight_layout()
plt.show()

###############################################################################
# References
# ----------
# .. [1] `Class-distinctiveness-based frequency band selection on the
#    Riemannian manifold for oscillatory activity-based BCIs: preliminary
#    results
#    <https://hal.archives-ouvertes.fr/hal-03641137/>`_
#    M. S. Yamamoto, F. Lotte, F. Yger, and S. Chevallier.
#    44th Annual International Conference of the IEEE Engineering
#    in Medicine & Biology Society (EMBC2022), 2022.
