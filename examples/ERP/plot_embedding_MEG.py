"""
=====================================================================
Embedding ERP MEG data in 2D Euclidean space
============================================

Embeddings via Laplacian Eigenmaps and Riemannian locally linear
embedding of a set of ERP data.

"""
# Authors:  Pedro Rodrigues <pedro.rodrigues01@gmail.com>,
#           Gabriel Wagner vom Berg <gabriel@bccn-berlin.de>
# License: BSD (3-clause)

from pyriemann.estimation import XdawnCovariances
from pyriemann.utils.viz import plot_embedding

import mne
from mne import io
from mne.datasets import sample

from sklearn.model_selection import train_test_split


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
# Embedding of Xdawn covariance matrices

nfilter = 4
xdwn = XdawnCovariances(estimator='scm', nfilter=nfilter)
split = train_test_split(X, y, train_size=0.25, random_state=42)
Xtrain, Xtest, ytrain, ytest = split
covs = xdwn.fit(Xtrain, ytrain).transform(Xtest)

###############################################################################
# Spectral Embedding (SE)
# -----------------------

plot_embedding(covs, metric='riemann', embd_type='Spectral', normalize=True)


###############################################################################
# Riemannian Locally Linear Embedding (RLLE)
# ------------------------------------------

plot_embedding(covs, metric='riemann', embd_type='LocallyLinear',
               normalize=True)
