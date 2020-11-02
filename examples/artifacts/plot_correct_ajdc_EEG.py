"""
=========================================================
Artifact Correction by AJDC-based Blind Source Separation
=========================================================

Blind source separation (BSS) based on approximate joint diagonalization of
Fourier cospectra (AJDC), applied to artifact correction of EEG [1].
"""
# Authors: Quentin Barthélemy & David Ojeda.
# EEG signal kindly shared by Marco Congedo.
#
# License: BSD (3-clause)

import numpy as np
from scipy.signal import welch
from mne import create_info              # tested with mne 0.21
from mne.io import RawArray
from mne.viz import plot_topomap
from mne.preprocessing import ICA
from pyriemann.spatialfilters import AJDC
from matplotlib import pyplot as plt


###############################################################################


def read_header(fname):
    """Read the header of blink.txt"""
    with open(fname, "r") as f:
        line = f.readline()
        content = line.split()
        return content[:-1], int(content[-1])


def plot_cospectra(cosp, freqs, ylabels=None, title=None):
    """Plot cospectra, ndarray of shape (n_freqs, n, n)"""
    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(title)
    n_freqs = cosp.shape[0]
    for f in range(n_freqs):
        ax = plt.subplot((n_freqs - 1)//8 + 1, 8, f+1)
        plt.imshow(cosp[f], cmap=plt.get_cmap('Reds'))
        plt.title('{} Hz'.format(freqs[f]))
        plt.xticks([])
        if ylabels and f == 0:
            plt.yticks(np.arange(0, len(ylabels), 2), ylabels[::2])
            ax.tick_params(axis='both', which='major', labelsize=7)
        elif ylabels and f == 8:
            plt.yticks(np.arange(1, len(ylabels), 2), ylabels[1::2])
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
# Channel space
# -------------

# Plot signal
ch_info = create_info(ch_names=ch_names, ch_types=['eeg'] * ch_count,
                      sfreq=sfreq)
ch_info.set_montage('standard_1020')
signal = RawArray(signal_raw, ch_info, verbose=False)
signal.plot(duration=duration, start=0, n_channels=ch_count,
            scalings={'eeg': 3e1}, color={'eeg': 'steelblue'},
            title='Original EEG signal', show_scalebars=False)


###############################################################################
# SOS-based BSS, diagonalizing cospectra
# --------------------------------------

# Compute and diagonalize cospectra between 1 and 32 Hz
window, overlap = sfreq, 0.5
fmin, fmax = 1, 32
ajdc = AJDC(window=window, overlap=overlap, fmin=fmin, fmax=fmax, fs=sfreq,
            expl_var=0.99)
ajdc.fit(signal_raw[np.newaxis, ...])
freqs = ajdc.freqs_

# Plot cospectra in channel space, after trace-normalization by frequency
plot_cospectra(ajdc._cosp_channels, freqs, ylabels=ch_names,
               title='Cospectra, in channel space')

# Plot diagonalized cospectra in source space
sr_count = ajdc.n_sources_
sr_names = ['S' + str(s).zfill(2) for s in range(sr_count)]
plot_cospectra(ajdc._cosp_sources, freqs, ylabels=sr_names,
               title='Diagonalized cospectra, in source space')


###############################################################################
# Source space
# ------------

# Estimate sources applying forward filters
source_raw = ajdc.transform(signal_raw)

# Plot sources
sr_info = create_info(ch_names=sr_names, ch_types=['misc'] * sr_count,
                      sfreq=sfreq)
source = RawArray(source_raw, sr_info, verbose=False)
source.plot(duration=duration, start=0, n_channels=sr_count,
            scalings={'misc': 2e2}, title='EEG sources estimated by AJDC',
            show_scalebars=False)


###############################################################################
# Artifact correction
# -------------------

# Identify artifact: blinks are well separated in source S0
blink_idx = 0

# Get normal spectrum, ie power spectrum after trace-normalization
blink_spectrum_norm = ajdc._cosp_sources[:, blink_idx, blink_idx]
blink_spectrum_norm = blink_spectrum_norm / np.linalg.norm(blink_spectrum_norm)

# Get absolute spectrum, ie raw power spectrum of the source
f, spectrum = welch(source.get_data(picks=[blink_idx]), fs=sfreq,
                    nperseg=window, noverlap=int(window * overlap))
blink_spectrum_abs = spectrum[0, (f >= fmin) & (f <= fmax)]
blink_spectrum_abs = blink_spectrum_abs / np.linalg.norm(blink_spectrum_abs)

# Get topographic map
blink_filter = ajdc.backward_filters_[:, blink_idx]

# Plot spectrum and topographic map of the blink source
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
axs[0].set(title='Power spectrum of the blink source estimated by AJDC',
           xlabel='Frequency (Hz)', ylabel='Power spectral density')
axs[0].plot(freqs, blink_spectrum_abs, label='Absolute power')
axs[0].plot(freqs, blink_spectrum_norm, label='Normal power')
axs[0].legend()
axs[1].set_title('Topographic map of the blink source estimated by AJDC')
plot_topomap(blink_filter, pos=ch_info, axes=axs[1], extrapolate='box')
plt.show()

# BSS denoising: suppress blink source and apply backward filters
denoised_signal_raw = ajdc.transform_back(source_raw, suppress=[blink_idx])

# Plot denoised signal
denoised_signal = RawArray(denoised_signal_raw, ch_info, verbose=False)
denoised_signal.plot(duration=duration, start=0, n_channels=ch_count,
                     scalings={'eeg': 3e1}, color={'eeg': 'steelblue'},
                     title='Denoised EEG signal by AJDC', show_scalebars=False)


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
