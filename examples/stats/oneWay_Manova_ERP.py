"""
====================================================================
Manova for ERP data
====================================================================

"""
# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

from time import time
from matplotlib import pyplot as plt
import seaborn as sns

import mne
from mne import io
from mne.datasets import sample
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from pyriemann.stats import PermutationDistance, PermutationModel
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

print(__doc__)
sns.set_style('whitegrid')
data_path = sample.data_path()

###############################################################################
# Set parameters and read data
###############################################################################
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

labels = epochs.events[::5, -1]

# get epochs
epochs_data = epochs.get_data()[::5]

n_perms = 100
###############################################################################
# Pairwise distance based permutation test
###############################################################################
t_init = time()
p_test = PermutationDistance(n_perms, metric='riemann', mode='pairwise',
                             estimator=XdawnCovariances(2))
p, F = p_test.test(epochs_data, labels)
duration = time() - t_init

fig, axes = plt.subplots(1, 1, figsize=[6, 3], sharey=True)
p_test.plot(nbins=10, axes=axes)
plt.title('Pairwise distance - %.2f sec.' % duration)
print('p-value: %.3f' % p)
sns.despine()
plt.tight_layout()
plt.show()

###############################################################################
# t-test distance based permutation test
###############################################################################

t_init = time()
p_test = PermutationDistance(n_perms, metric='riemann', mode='ttest',
                             estimator=XdawnCovariances(2))
p, F = p_test.test(epochs_data, labels)
duration = time() - t_init

fig, axes = plt.subplots(1, 1, figsize=[6, 3], sharey=True)
p_test.plot(nbins=10, axes=axes)
plt.title('Pairwise distance - %.2f sec.' % duration)
print('p-value: %.3f' % p)
sns.despine()
plt.tight_layout()
plt.show()

###############################################################################
# F-test distance based permutation test
###############################################################################

t_init = time()
p_test = PermutationDistance(n_perms, metric='riemann', mode='ftest',
                             estimator=XdawnCovariances(2))
p, F = p_test.test(epochs_data, labels)
duration = time() - t_init

fig, axes = plt.subplots(1, 1, figsize=[6, 3], sharey=True)
p_test.plot(nbins=10, axes=axes)
plt.title('Pairwise distance - %.2f sec.' % duration)
print('p-value: %.3f' % p)
sns.despine()
plt.tight_layout()
plt.show()

###############################################################################
# Classification based permutation test
###############################################################################

clf = make_pipeline(XdawnCovariances(2), TangentSpace('logeuclid'),
                    LogisticRegression())

t_init = time()
p_test = PermutationModel(n_perms, model=clf, cv=3)
p, F = p_test.test(epochs_data, labels)
duration = time() - t_init

fig, axes = plt.subplots(1, 1, figsize=[6, 3], sharey=True)
p_test.plot(nbins=10, axes=axes)
plt.title('Classification - %.2f sec.' % duration)
print('p-value: %.3f' % p)
sns.despine()
plt.tight_layout()
plt.show()
