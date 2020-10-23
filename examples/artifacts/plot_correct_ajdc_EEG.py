"""
=========================================================
Artifact Correction by AJDC-based Blind Source Separation
=========================================================

Blind source separation (BSS) based on approximate joint diagonalization of 
Fourier cospectra (AJDC), applied to artifact correction of EEG [1].
"""
# Authors: Quentin Barthélemy & David Ojeda
#
# License: BSD (3-clause)

import numpy as np
from mne import create_info
from mne.io import RawArray
from mne.viz import plot_topomap
from mne.time_frequency import psd_welch
from mne.preprocessing import ICA
from pyriemann.spatialfilters import AJDC
from matplotlib import pyplot as plt

import warnings
from mne import __version__ as v
if v != '0.18.2':
    warnings.warn('This example has only been tested with mne version 0.18.2',
                  UserWarning)


###############################################################################


def read_header(fname):
    with open(fname, "r") as f:
        line = f.readline()
        content = line.split()
        return content[:-1], int(content[-1])


def plot_cospectra(cosp, title, ch_names=None):
    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(title)
    for f in range(cosp.shape[0]):
        ax = plt.subplot(4, 8, f+1)
        plt.imshow(cosp[f,: , :], cmap=plt.get_cmap('YlGnBu'))
        plt.title('{} Hz'.format(f+1))
        plt.xticks([])
        if ch_names and f == 0:
            plt.yticks(np.arange(len(ch_names)), ch_names)
            ax.tick_params(axis='both', which='major', labelsize=7)
        else:
            plt.yticks([])
    plt.show()


###############################################################################
# Load EEG data
# -------------

fname = '../sample/blinks.txt'
signal_raw = np.loadtxt(fname, skiprows=1).T
ch_names, sfreq = read_header(fname)
ch_count = len(ch_names)
duration = signal_raw.shape[1] / sfreq


###############################################################################
# Signal space
# ------------

# Plot signal
info = create_info(ch_names=ch_names, ch_types=['eeg'] * ch_count, sfreq=sfreq,
                   montage='standard_1020')
signal = RawArray(signal_raw, info, verbose=False)
signal.plot(duration=duration, start=0, n_channels=ch_count,
            scalings={'eeg': 1e2}, color={'eeg': 'steelblue'},
            title='Original EEG signal')


###############################################################################
# SOS-based BSS, diagonalizing cospectra
# --------------------------------------

# Compute and diagonalize cospectra between 1 and 32 Hz
window, overlap = sfreq, 0.5
fmin, fmax = 1, 32
ajdc = AJDC(window=window, overlap=overlap, fmin=fmin, fmax=fmax, fs=sfreq)
ajdc.fit(signal_raw[np.newaxis, ...])

# Plot cospectra
plot_cospectra(ajdc._cosp, 'Raw cospectra, in signal space',
               ch_names=info['ch_names'])

# Plot diagonalized reduced cospectra
plot_cospectra(ajdc._diag_cosp,
               'Diagonalized reduced cospectra, in source space')


###############################################################################
# Source space
# ------------

# Estimate sources applying forward filters
source_raw = ajdc.transform(signal_raw)

# Plot sources
sr_count = source_raw.shape[0]
sr_info = create_info(ch_names=['S'+str(s) for s in range(sr_count)],
                      ch_types=['eeg'] * sr_count, sfreq=sfreq)
source = RawArray(source_raw, sr_info, verbose=False)
source.plot(duration=duration, start=0, n_channels=sr_count,
            scalings={'eeg': 5e-1}, color={'eeg': 'steelblue'},
            title='EEG sources estimated by AJDC')


###############################################################################
# Artifact correction
# -------------------

# Identify artifact: blinks are well separated in source S0
blink_idx = 0

# Plot topographic map and spectrum of the blink source
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
axs[0].set_title('Topographic map of the blink source estimated by AJDC')
axs[1].set(title='Spectrum of the blink source estimated by AJDC',
           xlabel='Frequency (Hz)', ylabel='Spectrum power (mV²)')
spectrum, freqs = psd_welch(source, n_fft=window, n_overlap=window * overlap,
                            fmin=fmin, fmax=fmax, picks='S' + str(blink_idx), 
                            verbose=False)
axs[1].plot(freqs, spectrum[0])
plot_topomap(ajdc.backward_filters_[:, blink_idx], pos=info, axes=axs[0])
plt.show()

# Suppress blink source and apply backward filters
denoised_signal_raw = ajdc.transform_back(source_raw, idx=[blink_idx])

# Plot denoised signal
denoised_signal = RawArray(denoised_signal_raw, info, verbose=False)
denoised_signal.plot(duration=duration, start=0, n_channels=ch_count,
                     scalings={'eeg': 1e2}, color={'eeg': 'steelblue'},
                     title='Denoised EEG signal by AJDC')


###############################################################################
# Comparison with ICA
# -------------------

# Infomax-based ICA is a HOS-based BSS, minimizing mutual information
ica = ICA(n_components=ajdc.n_sources_, method='infomax', random_state=42)
ica.fit(signal, picks='eeg')

# Can you find the blink source?
ica.plot_sources(signal, title='EEG sources estimated by ICA')
ica.plot_components(title='Topographic maps of EEG sources estimated by ICA')


###############################################################################
# References
# ----------
# [1] Q. Barthélemy, L. Mayaud, Y. Renard, D. Kim, S.-W. Kang, J. Gunkelman and 
# M. Congedo, "Online denoising of eye-blinks in electroencephalography",
# Neurophysiol Clin, 2017
