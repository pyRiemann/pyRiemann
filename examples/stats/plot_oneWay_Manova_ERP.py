"""
====================================================================
Manova for ERP data
====================================================================

"""
# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import seaborn as sns

from time import time
from matplotlib import pyplot as plt

import mne
from mne import io
from mne.datasets import sample

from pyriemann.stats import PermutationDistance, PermutationModel
from pyriemann.estimation import XdawnCovariances
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0., 1
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False)

labels = epochs.events[::10, -1]

# get epochs
epochs_data = epochs.get_data()[::10]

fig, axes = plt.subplots(2, 2, figsize=[12, 6], sharey=True)

###############################################################################
# Pairwise distance based permutation test
###############################################################################
print(epochs_data.shape)
t_init = time()
p_test = PermutationDistance(50, metric='riemann', mode='pairwise',
                             estimator=XdawnCovariances(2))
p, F = p_test.test(epochs_data, labels)
duration = time() - t_init

p_test.plot(axes=axes[0, 0], nbins=20)
axes[0, 0].set_title('Pairwise distance - %.2f sec.' % duration)
print('p-value: %.3f' % p)

###############################################################################
# t-test distance based permutation test
###############################################################################

t_init = time()
p_test = PermutationDistance(50, metric='riemann', mode='ttest',
                             estimator=XdawnCovariances(2))
p, F = p_test.test(epochs_data, labels)
duration = time() - t_init

p_test.plot(axes=axes[0, 1], nbins=20)
axes[0, 1].set_title('t-test distance - %.2f sec.' % duration)
print('p-value: %.3f' % p)

###############################################################################
# F-test distance based permutation test
###############################################################################

t_init = time()
p_test = PermutationDistance(50, metric='riemann', mode='ftest',
                             estimator=XdawnCovariances(2))
p, F = p_test.test(epochs_data, labels)
duration = time() - t_init

p_test.plot(axes=axes[1, 0], nbins=20)
axes[1, 0].set_title('F-test distance - %.2f sec.' % duration)
print('p-value: %.3f' % p)

sns.despine()
plt.tight_layout()
plt.show()
