"""
====================================================================
Motor imagery classification
====================================================================

Classify motor imagery data with Riemannian geometry [1]_.
"""
# generic import
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# mne import
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

# pyriemann import
from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances

# sklearn imports
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pyriemann.preprocessing import AugmentedDataset
from sklearn.svm import SVC
from pyriemann.tangentspace import TangentSpace

###############################################################################
# Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = 1., 2.
event_id = dict(hands=2, feet=3)
subject = 7
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_files = [
    read_raw_edf(f, preload=True) for f in eegbci.load_data(subject, runs)
]
raw = concatenate_raws(raw_files)

picks = pick_types(
    raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
# subsample elecs
picks = picks[::2]

# Apply band-pass filter
raw.filter(7., 35., method='iir', picks=picks)

events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

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

# get epochs
X = 1e6 * epochs.get_data()
y = epochs.events[:, -1] - 2

###########################################################################################
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

inner_cv = StratifiedKFold(3, shuffle=True, random_state=42)

#################################################################
pipelines = {}
pipelines["TGSP+SVM"] = Pipeline(steps=[
    ("Covariances", Covariances("oas")),
    ("Tangent_Space", TangentSpace(metric="riemann")),
    ("SVM", SVC(kernel="linear"))
])

###############################################################################
# Use scikit-learn Pipeline with cross_val_score function
scores = cross_val_score(pipelines["TGSP+SVM"], X, y, cv=outer_cv, n_jobs=-1)

# Printing the results
class_balance = np.mean(y == y[0])
class_balance = max(class_balance, 1. - class_balance)
print("TGSP+SVM Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                              class_balance))

#################################################################
pipelines = {}
pipelines["ACM+TGSP+SVM(Grid)"] = Pipeline(steps=[
    ("augmenteddataset", AugmentedDataset()),
    ("Covariances", Covariances("oas")),
    ("Tangent_Space", TangentSpace(metric="riemann")),
    ("SVM", SVC(kernel="linear"))
])

param_grid = {}
param_grid["ACM+TGSP+SVM(Grid)"] = {
    'augmenteddataset__order': [1, 2, 3, 4, 5, 6, 7],
    'augmenteddataset__lag': [1, 2, 3, 4, 5, 6, 7],
}

search = GridSearchCV(
    pipelines["ACM+TGSP+SVM(Grid)"],
    param_grid["ACM+TGSP+SVM(Grid)"],
    refit=True,
    cv=inner_cv,
    n_jobs=-1,
    scoring="accuracy",
    return_train_score=True,
)


###############################################################################
# Use scikit-learn Pipeline with cross_val_score function
scores = cross_val_score(search, X, y, cv=outer_cv, n_jobs=-1)

# Printing the results
class_balance = np.mean(y == y[0])
class_balance = max(class_balance, 1. - class_balance)
print("ACM+TGSP+SVM(Grid) Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                              class_balance))


###############################################################################
# References
# ----------
# .. [1] Carrara, I., & Papadopoulo, T. (2023). Classification of BCI-EEG based on augmented covariance matrix.
#        arXiv preprint arXiv:2302.04508.
#        https://doi.org/10.48550/arXiv.2302.04508
