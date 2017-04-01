"""
====================================================================
Embedding ERP EEG data in 2D Euclidean space with Diffusion maps
====================================================================

"""
# Authors: Pedro Rodrigues <pedro.rodrigues01@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.embedding import Embedding

import mne
from mne import io
from mne.datasets import sample

from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0., 1
event_id = dict(vis_l=3, vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True, verbose=False)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False)

X = epochs.get_data()
y = epochs.events[:, -1]

###############################################################################
# Embedding the Xdawn covariance matrices with Diffusion maps

n_components = 4  # pick some components 
covs = XdawnCovariances(n_components).fit_transform(X, y)
u,l = Embedding(metric='riemann').fit_transform(covs)

###############################################################################
# Plot the three first components of the embedded points

fig = plt.figure(figsize=(6.54, 5.66), facecolor='white')
ax  = fig.add_subplot(111, projection='3d')
ax.grid(False)
      
for label in np.unique(y):
    idx = (y==label)
    ax.scatter(u[idx,1], u[idx,2], u[idx,3], s=36)      
    
ax.set_xlabel(r'$\varphi_1$', fontsize=18)        
ax.set_ylabel(r'$\varphi_2$', fontsize=18)        
ax.set_zlabel(r'$\varphi_3$', fontsize=18)        
plt.title('3D embedding via Diffusion Maps', fontsize=16, position=(0.5, 1.10)) 







 
