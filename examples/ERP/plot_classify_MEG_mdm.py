"""
====================================================================
Multiclass MEG ERP Decoding
====================================================================

Decoding applied to MEG data in sensor space decomposed using Xdawn.
After spatial filtering, covariances matrices are estimated and
classified by the MDM algorithm (Nearest centroid).

4 Xdawn spatial patterns (1 for each class) are displayed, as per the
for mean-covariance matrices used by the classification algorithm.

"""
# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from matplotlib import pyplot as plt
from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM

import mne
from mne import io
from mne.datasets import sample

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + "/MEG/sample/sample_audvis_filt-0-40_raw.fif"
event_fname = data_path + "/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif"
tmin, tmax = -0.0, 1
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True)
raw.filter(2, None, method="iir")  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info["bads"] = ["MEG 2443"]  # set bad channels
picks = mne.pick_types(
    raw.info, meg="grad", eeg=False, stim=False, eog=False, exclude="bads"
)

# Read epochs
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    proj=False,
    picks=picks,
    baseline=None,
    preload=True,
    verbose=False,
)

labels = epochs.events[:, -1]
evoked = epochs.average()

###############################################################################
# Decoding with Xdawn + MDM

n_components = 3  # pick some components

# Define a monte-carlo cross-validation generator (reduce variance):
cv = KFold(n_splits=10, shuffle=True, random_state=42)
pr = np.zeros(len(labels))
epochs_data = epochs.get_data()

print("Multiclass classification with XDAWN + MDM")

clf = make_pipeline(XdawnCovariances(n_components), MDM())

for train_idx, test_idx in cv.split(epochs_data):
    y_train, y_test = labels[train_idx], labels[test_idx]

    clf.fit(epochs_data[train_idx], y_train)
    pr[test_idx] = clf.predict(epochs_data[test_idx])

print(classification_report(labels, pr))

###############################################################################
# plot the spatial patterns
xd = XdawnCovariances(n_components)
xd.fit(epochs_data, labels)

evoked.data = xd.Xd_.patterns_.T
evoked.times = np.arange(evoked.data.shape[0])
evoked.plot_topomap(
    times=[0, n_components, 2 * n_components, 3 * n_components],
    ch_type="grad",
    colorbar=False,
    size=1.5,
)

###############################################################################
# plot the confusion matrix
names = ["audio left", "audio right", "vis left", "vis right"]
cm = confusion_matrix(labels, pr)
ConfusionMatrixDisplay(cm, display_labels=names).plot()
plt.show()
