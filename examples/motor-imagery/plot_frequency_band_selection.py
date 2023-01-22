"""
=========================================================================
Frequency band selection on the manifold for motor imagery classification
=========================================================================

Find optimal frequency band using class distinctiveness measure on
the manifold and compare classification performance for motor imagery
data to the baseline with no frequency band selection [1]_.


"""
# Authors: Maria Sayu Yamamoto <maria-sayu.yamamoto@universite-paris-saclay.fr>
#
# License: BSD (3-clause)


import numpy as np
from time import time
from matplotlib import pyplot as plt

from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci

from sklearn.model_selection import train_test_split

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from helpers.frequencybandselection_helpers import freq_selection_class_dis

###############################################################################
# Set basic parameters and read data
# ------------------------------------

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

###############################################################################
# Baseline pipeline without frequency band selection
# ---------------------------------------------------
#
# Apply band-pass filter using a wide frequency band, 5-35 Hz.
# Train and evaluate classifier.
t0 = time()
raw_filter = raw.copy().filter(5., 35., method='iir', picks=picks,
                               verbose='warning')

events, _ = events_from_annotations(raw_filter, event_id,
                                    verbose='warning')

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

# Split data into training and test set
indices = np.array(range(labels.shape[0]))
Cov_train, Cov_test, y_train, y_test, train_ind, test_ind = \
    train_test_split(cov_data_baseline, labels, indices, test_size=0.2,
                     random_state=42)

# Set classifier
model = MDM(metric=dict(mean='riemann', distance='riemann'))

# Classification with minimum distance to mean
model.fit(Cov_train, y_train)
acc_baseline = model.score(Cov_test, y_test)
t1 = time() - t0

###############################################################################
# Pipeline with a frequency band selection based on the class distinctiveness
# ----------------------------------------------------------------------------
#
# Step1: Select frequency band maximizing class distinctiveness on
# training set.
#
# Define parameters for frequency band selection
t2 = time()
freq_band = [5., 35.]
sub_band_width = 4.
sub_band_step = 2.
alpha = 0.4

# Select frequency band using training set
best_freq, all_class_dis = \
    freq_selection_class_dis(raw, freq_band, sub_band_width,
                             sub_band_step, alpha,
                             tmin, tmax,
                             picks, event_id,
                             train_ind=train_ind, method='train_test_split',
                             return_class_dis=True, verbose=False)

print('Selected frequency band : ' + str(best_freq[0])
      + '-' + str(best_freq[1]) + ' Hz')

###############################################################################
# Step2: Train classifier using selected frequency band and evaluate
# performance using test set

# Apply band-pass filter using the best frequency band
best_raw_filter = raw.copy().filter(best_freq[0], best_freq[1],
                                    method='iir', picks=picks,
                                    verbose=False)

events, _ = events_from_annotations(best_raw_filter, event_id,
                                    verbose=False)

# Read epochs (train will be done only between 0.5 and 2.5s)
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

# Get epochs
epochs_data_train = 1e6 * epochs.get_data()

# Estimate covariance matrices
cov_data = Covariances().transform(epochs_data_train)

# Classification with minimum distance to mean
model.fit(cov_data[train_ind], labels[train_ind])
acc = model.score(cov_data[test_ind], labels[test_ind])

t3 = time() - t2

###############################################################################
# Compare pipelines: accuracies and training times
# ------------------------------------------------

print("Classification accuracy without frequency band selection: "
      + f"{acc_baseline:.02f}")
print("Total computational time without frequency band selection: "
      + f"{t1:.5f} s")
print("Classification accuracy with frequency band selection: "
      + f"{acc:.02f}")
print("Total computational time with frequency band selection: "
      + f"{t3:.5f} s")

###############################################################################
# Plot selected frequency bands
# ----------------------------------
#
# Plot the class distinctiveness values for each sub_band,
# along with the highlight of the finally selected frequency band.

subband_fmin = list(np.arange(freq_band[0],
                              freq_band[1] - sub_band_width + 1.,
                              sub_band_step))
subband_fmax = list(np.arange(freq_band[0] + sub_band_width,
                              freq_band[1] + 1., sub_band_step))
nb_subband = len(subband_fmin)

x = list(range(0, nb_subband, 1))
fig = plt.figure(figsize=(10, 5))

freq_start = subband_fmin.index(best_freq[0])
freq_end = subband_fmax.index(best_freq[1])

plt.subplot(1, 1, 1)
plt.grid()
plt.plot(x, all_class_dis, marker='o')
plt.xticks(list(range(0, 14, 1)),
           ["[5, 9]", "[7, 11]", "[9, 13]", "[11, 15]", "[13, 17]",
            "[15, 19]", "[17, 21]", "[19, 23]", "[21, 25]",
            "[23, 27]", "[25, 29]", "[27, 31]", "[29, 33]", "[31, 35]"])

plt.axvspan(freq_start, freq_end, color="orange", alpha=0.3,
            label='Selected frequency band')
plt.ylabel('Class distinctiveness')
plt.xlabel('Filter bank')
plt.title('Class distinctiveness value of each subband')
plt.legend(loc='upper right')

fig.tight_layout()
plt.show()

print('Optimal frequency band for this subject is ' + str(best_freq[0])
      + '-' + str(best_freq[1]) + ' Hz')

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
