"""
====================================================================
Offline SSVEP-based BCI Multiclass Prediction
====================================================================

Building extended covariance matrices for SSVEP-based BCI. The
obtained matrices are shown. A Minimum Distance to Mean classifier
is trained to predict a 4-class problem for an offline setup.
"""
# Authors: Sylvain Chevallier <sylvain.chevallier@uvsq.fr> ,
# Emmanuel Kalunga , Quentin Barthélemy
#
# License: BSD (3-clause)

# generic import
import os
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
import matplotlib.pyplot as plt

# mne import
from mne import get_config, set_config, find_events, create_info
from mne.datasets.utils import _get_path
from mne.utils import _fetch_file, _url_to_local_path
from mne.io import Raw, RawArray

# pyriemann import
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.classification import MDM

# scikit-learn import
from sklearn.model_selection import cross_val_score, RepeatedKFold


###############################################################################
# Loading EEG data
# ----------------
#
# The data are loaded through a MNE loader

SSVEPEXO_URL = 'https://zenodo.org/record/2392979/files/'
subject, run = 12, 1
url = '{:s}subject{:02d}_run{:d}_raw.fif'.format(SSVEPEXO_URL,
                                                 subject, run + 1)
sign = 'SSVEPEXO'
key, key_dest = 'MNE_DATASETS_SSVEPEXO_PATH', 'MNE-ssvepexo-data'

# Use MNE _fetch_file to download EEG file
if get_config(key) is None:
    set_config(key, os.path.join(os.path.expanduser("~"), "mne_data"))
path = _get_path(None, key, sign)
destination = _url_to_local_path(url, os.path.join(path, key_dest))
os.makedirs(os.path.dirname(destination), exist_ok=True)
if not os.path.exists(destination):
    _fetch_file(url, destination, print_destination=False)

# Read data in MNE Raw and numpy format
raw = Raw(destination, preload=True, verbose='ERROR')
events = find_events(raw, shortest_event=0, verbose=False)
raw = raw.pick_types(eeg=True)
event_id = {'13 Hz': 2, '17 Hz': 4, '21 Hz': 3, 'resting-state': 1}
sfreq = int(raw.info['sfreq'])

eeg_data = raw.get_data()

###############################################################################
# Visualization of raw EEG data
# -----------------------------
#
# Plot few seconds of signal from the `Oz` electrode using matplotlib

n_seconds = 2
time = np.linspace(0, n_seconds, n_seconds*sfreq).reshape((1, n_seconds*sfreq))
plt.figure(figsize=(10, 4))
plt.plot(time.T, eeg_data[np.array(raw.ch_names) == 'Oz', :n_seconds*sfreq].T,
         color='C0', lw=0.5)
plt.xlabel("Time (s)")
_ = plt.ylabel(r"Oz ($\mu$V)")

###############################################################################
# And a fewer seconds of all electrodes:

time = np.linspace(0, n_seconds, n_seconds*sfreq).reshape((1, n_seconds*sfreq))
plt.figure(figsize=(10, 4))
for ch_idx, ch_name in enumerate(raw.ch_names):
    plt.plot(time.T, eeg_data[ch_idx, :n_seconds*sfreq].T, lw=0.5, label=ch_name)
plt.xlabel("Time (s)")
plt.ylabel(r"EEG ($\mu$V)")
plt.legend(loc='upper right')

###############################################################################
# With MNE, it is much easier to visualize the data

raw.plot(duration=n_seconds, start=0, n_channels=8, scalings={'eeg': 4e-2}, 
         color={'eeg': 'steelblue'})

###############################################################################
# Extended signals for spatial covariance
# ---------------------------------------
# Using the approach proposed by [1]_, the SSVEP signal is extended to include
# the filtered signals for each stimulation frequency. We define a function to
# filter the signal:

def _butter_bandpass(signal, lowcut, highcut, fs, order=4):
    """ Bilateral Butterworth filter for offline filtering """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal, axis=-1)
    return filtered


# we stack the filtered signals to build an extended signal
frequencies = [13., 17., 21.]
freq_band = 0.1
ext_signal = np.empty_like(eeg_data[0, :])
for f in frequencies:
    ext_signal = np.vstack((ext_signal,
                            _butter_bandpass(eeg_data,
                                             lowcut=f-freq_band,
                                             highcut=f+freq_band,
                                             fs=sfreq)))
ext_signal = ext_signal[1:, :]

# and we build an array with the signal for each trial
ext_trials = list()
for t in events[:, 0]:
    start = t + 2 * sfreq
    stop = t + 5 * sfreq
    ext_trials.append(ext_signal[:, start:stop])
ext_trials = np.array(ext_trials)
n, c, t = ext_trials.shape

###############################################################################
# We plot 3 seconds of the signal from electrode Oz for a trial

n_seconds = 3
time = np.linspace(0, n_seconds, n_seconds*sfreq).reshape((1, n_seconds*sfreq))

plt.figure(figsize=(7, 5))
plt.plot(time.T, ext_trials[5, 0, :].T, label=str(int(frequencies[0]))+' Hz')
plt.plot(time.T, ext_trials[5, 8, :].T, label=str(int(frequencies[1]))+' Hz')
plt.plot(time.T, ext_trials[5, 16, :].T, label=str(int(frequencies[2]))+' Hz')
plt.xlabel("Time (s)")
plt.ylabel(r"Oz after filtering ($\mu$V)")
plt.legend(loc='upper right')

###############################################################################
# Creating an MNE Raw object from the extended signal and plot it

info = create_info(
    ch_names=sum(list(map(lambda s: [ch+s for ch in raw.ch_names],
                          ["-13Hz", "-17Hz", "-21Hz"])), []),
    ch_types=['eeg'] * 24,
    sfreq=sfreq)

raw_ext = RawArray(ext_signal, info)
raw_ext.plot(duration=n_seconds, start=14, n_channels=24,
             scalings={'eeg': 5e-4}, color={'eeg': 'steelblue'})

###############################################################################
# As it can be seen on this example, the subject is watching the 13Hz
# stimulation and the EEG activity is showing an increase activity in this
# frequency band while other frequency have a lower amplitude.
#
# Spatial covariance for SSVEP
# ----------------------------
#
# The covariance matrices will be estimated using the Ledoit-Wolf shrinkage
# estimator on the extended signal.

cov_ext_trials = Covariances(estimator='lwf').transform(ext_trials)

# This plot shows an example of a covariance matrix observed for each class:

plt.figure(figsize=(7, 7))
for i, l in enumerate(event_id):
    ax = plt.subplot(2, 2, i+1)
    plt.imshow(cov_ext_trials[events[:, 2] == event_id[l]][0],
               cmap=plt.get_cmap('RdBu_r'),
               interpolation='nearest')
    plt.title('Cov for class: '+l)
    plt.xticks([])
    if i == 0 or i == 2:
        plt.yticks(np.arange(len(info['ch_names'])),info['ch_names'])
        ax.tick_params(axis='both', which='major', labelsize=7)
    else:
        plt.yticks([])

###############################################################################
# It appears clearly that each class yields a different structure of the
# covariance matrix. Each stimulation (13, 17 and 21 Hz) generating higher
# covariance values for EEG signal filtered at the proper bandwith and no
# activation at all for the other bandwith. The resting state, where the
# subject focus on the center of the display and far from all blinking
# stimulus, shows an activity with higher correlation in the 13Hz frequency
# and lower but still visible activity in the other bandwith.
#
# Classify with MDM
# -----------------
# Plotting mean of each class

cov_centers = np.empty((len(event_id), 24, 24))
for i, l in enumerate(event_id):
    cov_centers[i] = mean_riemann(cov_ext_trials[events[:, 2] == event_id[l]])

plt.figure(figsize=(7, 7))
for i, l in enumerate(event_id):
    ax = plt.subplot(2, 2, i+1)
    plt.imshow(cov_centers[i],
               cmap=plt.get_cmap('RdBu_r'),
               interpolation='nearest')
    plt.title('Cov mean for class: '+l)
    plt.xticks([])
    if i == 0 or i == 2:
        plt.yticks(np.arange(len(info['ch_names'])),info['ch_names'])
        ax.tick_params(axis='both', which='major', labelsize=7)
    else:
        plt.yticks([])

###############################################################################
# Minimum distance to mean is a simple and robust algorithm for BCI decoding

cv = RepeatedKFold(n_splits=2, n_repeats=10, random_state=42)
mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
scores = cross_val_score(mdm, cov_ext_trials, events[:, 2], cv=cv, n_jobs=1)
print("MDM accuracy: {:.2f}% +/- {:.2f}".format(np.mean(scores)*100, 
                                                np.std(scores)*100))

###############################################################################
# References
# ----------
# [1] M. Congedo, A. Barachant, A. Andreev ,"A New generation of Brain-Computer 
# Interface Based on Riemannian Geometry", arXiv: 1310.8115, 2013.
# [2] E. K. Kalunga, S. Chevallier, Q. Barthélemy, K. Djouani, E. Monacelli, 
# Y. Hamam, "Online SSVEP-based BCI using Riemannian geometry", Neurocomputing, 
# vol. 191, p. 55-68, 2016.
