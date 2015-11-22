"""
====================================================================
Motor imagery classification
====================================================================

Classify Motor imagery data with Riemannian Geometry.
"""
# generic import
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# mne import
from mne import Epochs, pick_types
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.event import find_events
from mne.decoding import CSP

# pyriemann import
from pyriemann.classification import MDM, TSclassifier
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import covariances

# sklearn imports
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA

###############################################################################
# Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = 1., 2.
event_id = dict(hands=2, feet=3)
subject = 7
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_files = [read_raw_edf(f, preload=True) for f in eegbci.load_data(subject,
             runs)]
raw = concatenate_raws(raw_files)

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')
# subsample elecs
picks = picks[::2]

# Apply band-pass filter
raw.filter(7., 35., method='iir', picks=picks)

events = find_events(raw, shortest_event=0, stim_channel='STI 014')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True, add_eeg_ref=False, verbose=False)
labels = epochs.events[:, -1] - 2


# cross validation
cv = KFold(len(labels), 10, shuffle=True, random_state=42)
# get epochs
epochs_data_train = epochs.get_data()

# compute covariance matrices
cov_data_train = covariances(epochs_data_train)

###############################################################################
# Classification with Minimum distance to mean
mdm = MDM()

# Use scikit-learn Pipeline with cross_val_score function
scores = cross_val_score(mdm, cov_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("MDM Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                              class_balance))

###############################################################################
# Classification with Tangent Space Logistic Regression
clf = TSclassifier()
# Use scikit-learn Pipeline with cross_val_score function
scores = cross_val_score(clf, cov_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Tangent space Classification accuracy: %f / Chance level: %f" %
      (np.mean(scores), class_balance))
###############################################################################
# Classification with CSP + linear discrimant analysis

# Assemble a classifier
lda = LDA()
csp = CSP(n_components=4, reg='ledoit_wolf', log=True)


clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("CSP + LDA Classification accuracy: %f / Chance level: %f" %
      (np.mean(scores), class_balance))

###############################################################################
# Display MDM centroid

mdm = MDM()
mdm.fit(cov_data_train, labels)

fig, axes = plt.subplots(1, 2, figsize=[8, 4])
ch_names = [ch.replace('.', '') for ch in epochs.ch_names]

df = pd.DataFrame(data=mdm.covmeans[0], index=ch_names, columns=ch_names)
g = sns.heatmap(df, ax=axes[0], square=True, cbar=False, xticklabels=2,
                yticklabels=2)
g.set_title('Mean covariance - hands')

df = pd.DataFrame(data=mdm.covmeans[1], index=ch_names, columns=ch_names)
g = sns.heatmap(df, ax=axes[1], square=True, cbar=False, xticklabels=2,
                yticklabels=2)
plt.xticks(rotation='vertical')
plt.yticks(rotation='horizontal')
g.set_title('Mean covariance - feets')

# dirty fix
plt.sca(axes[0])
plt.xticks(rotation='vertical')
plt.yticks(rotation='horizontal')
plt.show()
