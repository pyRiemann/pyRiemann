"""
====================================================================
One Way manova
====================================================================

One way manova to compare Left vs Right.
"""
import seaborn as sns

from time import time
from matplotlib import pyplot as plt

from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci

from pyriemann.stats import PermutationDistance, PermutationModel
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

sns.set_style('whitegrid')
###############################################################################
# Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = 1., 3.
event_id = dict(hands=2, feet=3)
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_files = [
    read_raw_edf(f, preload=True, verbose=False)
    for f in eegbci.load_data(subject, runs)
]
raw = concatenate_raws(raw_files)

# Apply band-pass filter
raw.filter(7., 35., method='iir')

events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

picks = pick_types(
    raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
picks = picks[::4]

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
    verbose=False)
labels = epochs.events[:, -1] - 2

# get epochs
epochs_data = epochs.get_data()

# compute covariance matrices
covmats = Covariances().fit_transform(epochs_data)

n_perms = 500
###############################################################################
# Pairwise distance based permutation test
###############################################################################

t_init = time()
p_test = PermutationDistance(n_perms, metric='riemann', mode='pairwise')
p, F = p_test.test(covmats, labels)
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
p_test = PermutationDistance(n_perms, metric='riemann', mode='ttest')
p, F = p_test.test(covmats, labels)
duration = time() - t_init

fig, axes = plt.subplots(1, 1, figsize=[6, 3], sharey=True)
p_test.plot(nbins=10, axes=axes)
plt.title('t-test distance - %.2f sec.' % duration)
print('p-value: %.3f' % p)
sns.despine()
plt.tight_layout()
plt.show()

###############################################################################
# F-test distance based permutation test
###############################################################################

t_init = time()
p_test = PermutationDistance(n_perms, metric='riemann', mode='ftest')
p, F = p_test.test(covmats, labels)
duration = time() - t_init

fig, axes = plt.subplots(1, 1, figsize=[6, 3], sharey=True)
p_test.plot(nbins=10, axes=axes)
plt.title('F-test distance - %.2f sec.' % duration)
print('p-value: %.3f' % p)
sns.despine()
plt.tight_layout()
plt.show()

###############################################################################
# Classification based permutation test
###############################################################################

clf = make_pipeline(CSP(4), LogisticRegression())

t_init = time()
p_test = PermutationModel(n_perms, model=clf, cv=3, scoring='roc_auc')
p, F = p_test.test(covmats, labels)
duration = time() - t_init

fig, axes = plt.subplots(1, 1, figsize=[6, 3], sharey=True)
p_test.plot(nbins=10, axes=axes)
plt.title('Classification - %.2f sec.' % duration)
print('p-value: %.3f' % p)
sns.despine()
plt.tight_layout()
plt.show()
