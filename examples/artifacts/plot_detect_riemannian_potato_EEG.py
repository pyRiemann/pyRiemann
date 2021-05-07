"""
===============================================================================
Artifact Detection with Riemannian Potato
===============================================================================

Example of Riemannian Potato [1] applied on EEG time-series to detect artifacts
in online processing. It is computed only for two channels to display intuitive
visualizations.
"""
# Authors: Quentin Barthélemy & David Ojeda
#
# License: BSD (3-clause)

from functools import partial

import os
import numpy as np
from mne.datasets import sample                    # tested with mne 0.21
from mne.io import read_raw_fif
from mne import Epochs, make_fixed_length_events
from pyriemann.estimation import Covariances
from pyriemann.clustering import Potato
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


###############################################################################


@partial(np.vectorize, excluded=['potato'])
def get_zscores(cov_00, cov_01, cov_11, potato):
    cov = np.array([[cov_00, cov_01], [cov_01, cov_11]])
    return potato.transform(cov[np.newaxis, ...])


def plot_potato_2D(ax, cax, X, Y, p_zscores, p_center, covs, p_colors, clabel):
    qcs = ax.contourf(X, Y, p_zscores, levels=20, vmin=p_zscores.min(),
                      vmax=p_zscores.max(), cmap='RdYlBu_r', alpha=0.5)
    ax.contour(X, Y, p_zscores, levels=[z_th], colors='k')
    sc = ax.scatter(covs[:, 0, 0], covs[:, 1, 1], c=p_colors)
    ax.scatter(p_center[0, 0], p_center[1, 1], c='k', s=100)
    if cax:
        cbar = fig.colorbar(qcs, cax=cax)
    else:
        cbar = fig.colorbar(qcs, ax=ax)
    cbar.ax.set_ylabel(clabel)
    return sc


def plot_sig(ax, time, sig):
    ax.axis((time[0], time[-1], -15, 15))
    pl, = ax.plot(time, sig, lw=0.75)
    ax.axhspan(
        -14, 14, edgecolor='k', facecolor='none',
        xmin=-test_time_start / (test_time_end - test_time_start),
        xmax=(duration - test_time_start) / (test_time_end - test_time_start))
    return pl


def add_alpha(p_cols, alphas):
    cols = [to_rgb(c) for c in p_cols]
    return [(c[0], c[1], c[2], a) for c, a in zip(cols, alphas[-len(p_cols):])]


###############################################################################
# Load EEG data
# -------------

raw_fname = os.path.join(sample.data_path(), 'MEG', 'sample',
                         'sample_audvis_filt-0-40_raw.fif')
raw = read_raw_fif(raw_fname, preload=True, verbose=False)
sfreq = int(raw.info['sfreq'])


###############################################################################
# Offline processing of EEG data
# ------------------------------

# Apply common average reference on EEG channels
raw.pick_types(meg=False, eeg=True).apply_proj()

# Select two EEG channels for the example, preferably without artifact at the
# beginning to have a reliable calibration
ch_names = ['EEG 010', 'EEG 015']

# Apply band-pass filter between 1 and 35 Hz
raw.filter(1., 35., method='iir', picks=ch_names)

# Epoch time-series with a sliding window
duration = 2.5    # duration of epochs
interval = 0.2    # interval between epochs
epochs = Epochs(raw, make_fixed_length_events(raw, id=1, duration=interval),
                tmin=0., tmax=duration, baseline=None, verbose=False)
epochs_data = 5e5 * epochs.get_data(picks=ch_names)

# Estimate spatial covariance matrices
covs = Covariances(estimator='lwf').transform(epochs_data)


###############################################################################
# Calibration of Potato
# ---------------------

z_th = 2.5       # z-score threshold
t = 40           # nb of matrices to train the potato

# Calibrate potato by unsupervised training on first matrices: compute a
# reference matrix, mean and standard deviation of distances to this reference.
train_set = range(t)
rp = Potato(metric='riemann', threshold=z_th).fit(covs[train_set])
rp_center = rp._mdm.covmeans_[0]
ep = Potato(metric='euclid', threshold=z_th).fit(covs[train_set])
ep_center = ep._mdm.covmeans_[0]

rp_labels = rp.predict(covs[train_set])
rp_colors = ['b' if l==1 else 'r' for l in rp_labels.tolist()]
ep_labels = ep.predict(covs[train_set])
ep_colors = ['b' if l==1 else 'r' for l in ep_labels.tolist()]

# 2D projection of the z-score map of the Riemannian potato, for 2x2 covariance
# matrices (in blue if clean, in red if artifacted) and their reference matrix
# (in black). The colormap defines the z-score and a chosen isocontour defines
# the potato. It reproduces Fig 1 of reference [2].

# Zscores in the horizontal 2D plane going through the reference
X, Y = np.meshgrid(np.linspace(1, 31, 100), np.linspace(1, 31, 100))
rp_zscores = get_zscores(X, np.full_like(X, rp_center[0, 1]), Y, potato=rp)
rp_mzscores = np.ma.masked_where(~np.isfinite(rp_zscores), rp_zscores)
ep_zscores = get_zscores(X, np.full_like(X, ep_center[0, 1]), Y, potato=ep)

# Plot calibration
xlabel = 'Cov({},{})'.format(ch_names[0], ch_names[0])
ylabel = 'Cov({},{})'.format(ch_names[1], ch_names[1])

fig, axs = plt.subplots(figsize=(12, 5), nrows=1, ncols=2)
fig.suptitle('Offline calibration of potatoes', fontsize=16)
axs[0].set(xlabel=xlabel, ylabel=ylabel,
           title='2D projection of Riemannian potato')
plot_potato_2D(axs[0], None, X, Y, rp_mzscores, rp_center, covs[train_set],
               rp_colors, 'Z-score of Riemannian distance to reference')
axs[1].set(xlabel=xlabel, ylabel=ylabel,
           title='2D projection of Euclidean potato')
plot_potato_2D(axs[1], None, X, Y, ep_zscores, ep_center, covs[train_set],
               ep_colors, 'Z-score of Euclidean distance to reference')
plt.show()


###############################################################################
# Online Artifact Detection with Potato
# -------------------------------------

# Detect artifacts/outliers on test set, with an animation to imitate an online
# acquisition, processing and artifact detection of EEG time-series.
# The potato is static: it is not updated when EEG is not artifacted, damaging
# its efficiency over time.

test_covs_max = 300     # nb of matrices to visualize in this example
test_covs_visu = 30     # nb of matrices to display simultaneously
test_time_start = -2    # start time to display signal
test_time_end = 5       # end time to display signal

time_start = t * interval + test_time_start
time_end = t * interval + test_time_end
time = np.linspace(time_start, time_end, int((time_end - time_start) * sfreq),
                   endpoint=False)
eeg_data = 3e5 * raw.get_data(picks=ch_names)
sig = eeg_data[:, int(time_start * sfreq):int(time_end * sfreq)]
covs_visu = np.empty([0, 2, 2])
rp_colors, ep_colors = [], []
alphas = np.linspace(0, 1, test_covs_visu)

# Prepare animation for online detection
fig = plt.figure(figsize=(12, 10), constrained_layout=False)
fig.suptitle('Online artifact detection by potatoes', fontsize=16)
gs = fig.add_gridspec(nrows=4, ncols=40, top=0.90, hspace=0.3, wspace=1.0)
ax_sig0 = fig.add_subplot(gs[0, :], xlabel='Time (s)', ylabel=ch_names[0])
pl_sig0 = plot_sig(ax_sig0, time, sig[0])
ax_sig1 = fig.add_subplot(gs[1, :], ylabel=ch_names[1])
pl_sig1 = plot_sig(ax_sig1, time, sig[1])
ax_sig1.set_xticks([])
ax_rp = fig.add_subplot(gs[2:4, 0:15], xlabel=xlabel, ylabel=ylabel,
                        title='2D projection of Riemannian potato')
cax_rp = fig.add_subplot(gs[2:4, 15])
p_rp = plot_potato_2D(ax_rp, cax_rp, X, Y, rp_mzscores, rp_center, covs_visu,
                      rp_colors, 'Z-score of Riemannian distance to reference')
ax_ep = fig.add_subplot(gs[2:4, 21:36], xlabel=xlabel, ylabel=ylabel,
                        title='2D projection of Euclidean potato')
cax_ep = fig.add_subplot(gs[2:4, 36])
p_ep = plot_potato_2D(ax_ep, cax_ep, X, Y, ep_zscores, ep_center, covs_visu,
                      ep_colors, 'Z-score of Euclidean distance to reference')

# Plot online detection (an interactive display is required)
def online_update(self):
    global t, time, sig, covs_visu

    # Online artifact detection
    rp_label = rp.predict(covs[t][np.newaxis, ...])
    ep_label = ep.predict(covs[t][np.newaxis, ...])

    # Update data
    time_start = t * interval + test_time_end
    time_end = (t + 1) * interval + test_time_end
    time_ = np.linspace(time_start, time_end, int(interval * sfreq),
                        endpoint=False)
    time = np.r_[time[int(interval * sfreq):], time_]
    sig = np.hstack((sig[:, int(interval*sfreq):],
                     eeg_data[:, int(time_start*sfreq):int(time_end*sfreq)]))
    covs_visu = np.vstack((covs_visu, covs[t][np.newaxis, ...]))
    rp_colors.append('b' if rp_label==1 else 'r')
    ep_colors.append('b' if ep_label==1 else 'r')
    if len(covs_visu) > test_covs_visu:
        covs_visu = covs_visu[1:]
        rp_colors.pop(0)
        ep_colors.pop(0)
    rp_colors_ = add_alpha(rp_colors, alphas)
    ep_colors_ = add_alpha(ep_colors, alphas)
    t += 1

    # Update plot
    pl_sig0.set_data(time, sig[0])
    pl_sig0.axes.set_xlim(time[0], time[-1])
    pl_sig1.set_data(time, sig[1])
    pl_sig1.axes.set_xlim(time[0], time[-1])
    p_rp.set_offsets(np.c_[covs_visu[:, 0, 0], covs_visu[:, 1, 1]])
    p_rp.set_color(rp_colors_)
    p_ep.set_offsets(np.c_[covs_visu[:, 0, 0], covs_visu[:, 1, 1]])
    p_ep.set_color(ep_colors_)
    return pl_sig0, pl_sig1, p_rp, p_ep

# For a correct display, change the parameter interval_display
interval_display = 1.0
potato = FuncAnimation(fig, online_update, frames=test_covs_max,
                       interval=interval_display, blit=False, repeat=False)
plt.show()


###############################################################################
# References
# ----------
# [1] A. Barachant, A. Andreev, M. Congedo, "The Riemannian Potato: an
# automatic and adaptive artifact detection method for online experiments using
# Riemannian geometry", Proc. TOBI Workshop IV, 2013.
#
# [2] Q. Barthélemy, L. Mayaud, D. Ojeda, M. Congedo, "The Riemannian potato
# field: a tool for online signal quality index of EEG", IEEE TNSRE, 2019.
