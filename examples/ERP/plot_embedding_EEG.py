"""
====================================================================
Embedding ERP EEG data in 2D Euclidean space with Laplacian Eigenmaps
====================================================================

Spectral embedding via Laplacian Eigenmaps of a set of ERP data.

"""
# Authors: Pedro Rodrigues <pedro.rodrigues01@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from pyriemann.estimation import XdawnCovariances
from pyriemann.embedding import Embedding

import mne
from mne import io
from mne.datasets import sample

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import seaborn as sns

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0., 1
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True, verbose=False)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False)

X = epochs.get_data()
y = epochs.events[:, -1]

###############################################################################
# Embedding the Xdawn covariance matrices with Laplacian Eigenmaps

nfilter = 4
xdwn = XdawnCovariances(estimator='scm', nfilter=nfilter)
split = train_test_split(X, y, train_size=0.25, random_state=42)
Xtrain, Xtest, ytrain, ytest = split
covs = xdwn.fit(Xtrain, ytrain).transform(Xtest)

lapl = Embedding(metric='riemann', n_components=2)
embd = lapl.fit_transform(covs)

###############################################################################
# Plot the three first components of the embedded points

fig, ax = plt.subplots(figsize=(7, 8), facecolor='white')

for cond, label in event_id.items():
    idx = (ytest == label)
    ax.scatter(embd[idx, 0], embd[idx, 1], s=36, label=cond)

ax.set_xlabel(r'$\varphi_1$', fontsize=16)
ax.set_ylabel(r'$\varphi_2$', fontsize=16)
ax.set_title('Spectral embedding of ERP recordings', fontsize=16)
ax.set_xticks([-1.0, -0.5, 0.0, +0.5, 1.0])
ax.set_yticks([-1.0, -0.5, 0.0, +0.5, 1.0])
ax.grid(False)
ax.legend()
plt.show()
