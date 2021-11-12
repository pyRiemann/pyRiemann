"""
====================================================================
Offline SSVEP-based BCI Multiclass Prediction
====================================================================

Building extended covariance matrices for SSVEP-based BCI. The
obtained matrices are shown. A Minimum Distance to Mean classifier
is trained to predict a 4-class problem for an offline setup.
"""
# Authors: Sylvain Chevallier <sylvain.chevallier@uvsq.fr>,
# Emmanuel Kalunga, Quentin Barthélemy, David Ojeda
#
# License: BSD (3-clause)

# generic import
import os
import numpy as np
import matplotlib.pyplot as plt

# mne import
from mne import find_events, create_info, Epochs
from mne.io import Raw, RawArray
from mne.datasets import fetch_dataset

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

def download_sample_data(dataset="ssvep", subject=1, session=1):
    """Download BCI data for example purpose

    Parameters
    ----------
    dataset : str
        type of the dataset, could be "ssvep", "p300" or "imagery"
        Default is "ssvep", as other are not implemented
    subject : int
        Subject id, dataset specific (default: 1)
    session : int, default 1
        Session number%load , dataset specific (default: 1)

    Returns
    -------
    destination : str
        Path to downloaded data
    """
    if dataset == "ssvep":
        DATASET_URL = 'https://zenodo.org/record/2392979/files/'
        url = '{:s}subject{:02d}_run{:d}_raw.fif'.format(DATASET_URL,
                                                         subject, session + 1)
        config_key = 'MNE_DATASETS_SSVEPEXO_PATH'
        folder_name = 'MNE-ssvepexo-data'
    elif dataset == "p300" or dataset == "imagery":
        raise NotImplementedError("Not yet implemented")

    # Use MNE fetch_dataset to download EEG file
    archive_name = url.split('/')[-1]
    dataset_params = {
        'dataset_name': dataset,
        'archive_name': archive_name,
        'hash': 'md5:ff7f0361a2d41f8df3fb53b9a9bc1220',
        'url': url,
        'folder_name': folder_name,
        'config_key': config_key
    }

    data_path = fetch_dataset(
        dataset_params
    )

    return os.path.join(data_path, archive_name)


# Download data
destination = download_sample_data(dataset="ssvep", subject=12, session=1)
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
time = np.linspace(0, n_seconds, n_seconds * sfreq,
                   endpoint=False)[np.newaxis, :]
plt.figure(figsize=(10, 4))
plt.plot(time.T, eeg_data[np.array(raw.ch_names) == 'Oz', :n_seconds*sfreq].T,
         color='C0', lw=0.5)
plt.xlabel("Time (s)")
plt.ylabel(r"Oz ($\mu$V)")
plt.show()

###############################################################################
# And of all electrodes:

plt.figure(figsize=(10, 4))
for ch_idx, ch_name in enumerate(raw.ch_names):
    plt.plot(time.T, eeg_data[ch_idx, :n_seconds*sfreq].T, lw=0.5,
             label=ch_name)
plt.xlabel("Time (s)")
plt.ylabel(r"EEG ($\mu$V)")
plt.legend(loc='upper right')
plt.show()

###############################################################################
# With MNE, it is much easier to visualize the data

raw.plot(duration=n_seconds, start=0, n_channels=8, scalings={'eeg': 4e-2},
         color={'eeg': 'steelblue'})

###############################################################################
# Extended signals for spatial covariance
# ---------------------------------------
# Using the approach proposed by [1], the SSVEP signal is extended to include
# the filtered signals for each stimulation frequency. We stack the filtered
# signals to build an extended signal.


def _bandpass_filter(signal, lowcut, highcut):
    """ Bandpass filter using MNE """
    return signal.copy().filter(l_freq=lowcut, h_freq=highcut,
                                method="iir").get_data()


# We stack the filtered signals to build an extended signal
frequencies = [13., 17., 21.]
freq_band = 0.1
ext_signal = np.vstack([_bandpass_filter(raw,
                                         lowcut=f-freq_band,
                                         highcut=f+freq_band)
                        for f in frequencies])

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
# Building Epochs and plotting 3 s of the signal from electrode Oz for a trial

epochs = Epochs(raw_ext, events, event_id, tmin=2, tmax=5, baseline=None)

n_seconds = 3
time = np.linspace(0, n_seconds, n_seconds * sfreq,
                   endpoint=False)[np.newaxis, :]
channels = range(0, len(raw_ext.ch_names), len(raw.ch_names))
plt.figure(figsize=(7, 5))
for f, c in zip(frequencies, channels):
    plt.plot(epochs.get_data()[5, c, :].T, label=str(int(f))+' Hz')
plt.xlabel("Time (s)")
plt.ylabel(r"Oz after filtering ($\mu$V)")
plt.legend(loc='upper right')
plt.show()

###############################################################################
# As it can be seen on this example, the subject is watching the 13Hz
# stimulation and the EEG activity is showing an increase activity in this
# frequency band while other frequencies have lower amplitudes.
#
# Spatial covariance for SSVEP
# ----------------------------
#
# The covariance matrices will be estimated using the Ledoit-Wolf shrinkage
# estimator on the extended signal.

cov_ext_trials = Covariances(estimator='lwf').transform(epochs.get_data())

# This plot shows an example of a covariance matrix observed for each class:

plt.figure(figsize=(7, 7))
for i, l in enumerate(event_id):
    ax = plt.subplot(2, 2, i+1)
    plt.imshow(cov_ext_trials[events[:, 2] == event_id[l]][0],
               cmap=plt.get_cmap('RdBu_r'))
    plt.title('Cov for class: '+l)
    plt.xticks([])
    if i == 0 or i == 2:
        plt.yticks(np.arange(len(info['ch_names'])), info['ch_names'])
        ax.tick_params(axis='both', which='major', labelsize=7)
    else:
        plt.yticks([])
plt.show()

###############################################################################
# It appears clearly that each class yields a different structure of the
# covariance matrix. Each stimulation (13, 17 and 21 Hz) generating higher
# covariance values for EEG signal filtered at the proper bandwith and no
# activation at all for the other bandwiths. The resting state, where the
# subject focus on the center of the display and far from all blinking
# stimulus, shows an activity with higher correlation in the 13Hz frequency
# and lower but still visible activity in the other bandwiths.
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
    plt.imshow(cov_centers[i], cmap=plt.get_cmap('RdBu_r'))
    plt.title('Cov mean for class: '+l)
    plt.xticks([])
    if i == 0 or i == 2:
        plt.yticks(np.arange(len(info['ch_names'])), info['ch_names'])
        ax.tick_params(axis='both', which='major', labelsize=7)
    else:
        plt.yticks([])
plt.show()

###############################################################################
# Minimum distance to mean is a simple and robust algorithm for BCI decoding.
# It reproduces results of [2] for the first session of subject 12.

cv = RepeatedKFold(n_splits=2, n_repeats=10, random_state=42)
mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
scores = cross_val_score(mdm, cov_ext_trials, events[:, 2], cv=cv, n_jobs=1)
print("MDM accuracy: {:.2f}% +/- {:.2f}".format(np.mean(scores)*100,
                                                np.std(scores)*100))
# The obtained results are 80.62% +/- 16.29 for this session, with a repeated
# k-fold validation.

###############################################################################
# References
# ----------
# [1] M. Congedo, A. Barachant, A. Andreev ,"A New generation of Brain-Computer
# Interface Based on Riemannian Geometry", arXiv: 1310.8115, 2013.
#
# [2] E. K. Kalunga, S. Chevallier, Q. Barthélemy, E. Monacelli,
# "Review of Riemannian distances and divergences, applied to SSVEP-based BCI",
# Neuroinformatics, 2020.
