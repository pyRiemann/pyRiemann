"""
=============================================================================
Mutlitclass Decoding in sensor space data using Riemannian Geometry and XDAWN
=============================================================================

Decoding applied to EEG data in sensor space decomposed using Xdawn.
After spatial filtering, covariances matrices are estimated and
classified by the MDM algorithm (Nearest centroid) or the more accurate
version FgMDM (MDM + geodesic filtering)

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Romain Trachel <romain.trachel@inria.fr>
#          Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM,FgMDM

import mne
from mne import io
from mne.datasets import sample

from sklearn.pipeline import Pipeline  # noqa
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0., 1
event_id = dict(aud_l=1,aud_r=2, vis_l=3,vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True)

labels = epochs.events[:, -1]
evoked = epochs.average()

###############################################################################
# Decoding in sensor space using a linear SVM


n_components = 3  # pick some components

# Define a monte-carlo cross-validation generator (reduce variance):
cv = KFold(len(labels), 10,shuffle=True, random_state=42)
pr = np.zeros(len(labels))
epochs_data = epochs.get_data()



print('Multiclass classification with XDAWN + MDM')
clf = Pipeline([('COV',XdawnCovariances(n_components)),('MDM',MDM())])

for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    clf.fit(epochs_data[train_idx], y_train)
    pr[test_idx] = clf.predict(epochs_data[test_idx])

print classification_report(labels,pr)
print confusion_matrix(labels,pr)

print('Multiclass classification with XDAWN + FgMDM')
clf = Pipeline([('COV',XdawnCovariances(n_components)),('MDM',FgMDM())])

for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    clf.fit(epochs_data[train_idx], y_train)
    pr[test_idx] = clf.predict(epochs_data[test_idx])

print classification_report(labels,pr)
print confusion_matrix(labels,pr)