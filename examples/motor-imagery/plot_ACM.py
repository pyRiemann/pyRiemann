"""
====================================================================
Augmented Covariance Method (ACM)
====================================================================

This example shows how to use the ACM classifier [1]_.
"""
# Authors: Igor Carrara <igor.carrara@inria.fr>
#
# License: BSD (3-clause)

# generic import
import numpy as np
import matplotlib.pyplot as plt

# mne import
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci

# pyriemann import
from pyriemann.estimation import Covariances

# sklearn imports
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from pyriemann.classification import MDM
from pyriemann.classification import FgMDM
from pyriemann.tangentspace import TangentSpace
import pandas as pd
from sklearn.base import clone
import seaborn as sns

###############################################################################
# Define the Augmented Dataset
# -------------------------------
class AugmentedDataset(BaseEstimator, TransformerMixin):
    """This transformation allow to create an embedding version of the current dataset.
    The implementation and the application is described in [1]_.
    """

    def __init__(self, order=1, lag=1):
        self.order = order
        self.lag = lag

    def fit(self, X, y):
        return self

    def transform(self, X):
        if self.order == 1:
            X_fin = X
        else:
            X_fin = []

            for i in np.arange(X.shape[0]):
                X_p = X[i][:, : -self.order * self.lag]
                X_p = np.concatenate(
                    [X_p]
                    + [
                        X[i][:, p * self.lag : -(self.order - p) * self.lag]
                        for p in range(1, self.order)
                    ],
                    axis=0,
                )
                X_fin.append(X_p)
            X_fin = np.array(X_fin)

        return X_fin

###############################################################################
# Load EEG data
# -------------

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
# Defining Cross Validation Scheme
# -------------------
#
# Define the inner and outer CV scheme for implemented the Nested Cross Validation for hyper-parameter search

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

inner_cv = StratifiedKFold(3, shuffle=True, random_state=42)

#################################################################
# Defining pipelines
# -------------------
#
# Compare TGSP+SVM, MDM and ACM+TGSP+SVM ACM+MDM

# Define the standard pipeline TGSP+SVM as baseline with standard covariance
pipelines = {}
pipelines["TGSP+SVM"] = Pipeline(steps=[
    ("Covariances", Covariances("oas")),
    ("Tangent_Space", TangentSpace(metric="riemann")),
    ("SVM", SVC(kernel="linear"))
])

pipelines["MDM"] = Pipeline(steps=[
    ("Covariances", Covariances("cov")),
    ("MDM", MDM(metric=dict(mean='riemann', distance='riemann')))
])

#################################################################
# Define the ACM pipeline, the approach is based on the expansion of the current EEG signal using theory of
# phase space reconstruction
pipelines_ = {}
pipelines_["ACM+TGSP+SVM(Grid)"] = Pipeline(steps=[
    ("augmenteddataset", AugmentedDataset()),
    ("Covariances", Covariances("oas")),
    ("Tangent_Space", TangentSpace(metric="riemann")),
    ("SVM", SVC(kernel="linear"))
])

pipelines_["ACM+MDM(Grid)"] = Pipeline(steps=[
    ("augmenteddataset", AugmentedDataset()),
    ("Covariances", Covariances("oas")),
    ("MDM", MDM(metric=dict(mean='riemann', distance='riemann')))
])

param_grid = {}
# Define the parameter to test in the Nested Cross Validation
param_grid["ACM+TGSP+SVM(Grid)"] = {
    'augmenteddataset__order': [1, 2, 3, 4, 5, 6, 7],
    'augmenteddataset__lag': [1, 2, 3, 4, 5, 6, 7],
}
param_grid["ACM+MDM(Grid)"] = {
    'augmenteddataset__order': [1, 2, 3, 4, 5, 6, 7],
    'augmenteddataset__lag': [1, 2, 3, 4, 5, 6, 7],
}


pipelines["ACM+TGSP+SVM(Grid)"] = GridSearchCV(
    pipelines_["ACM+TGSP+SVM(Grid)"],
    param_grid["ACM+TGSP+SVM(Grid)"],
    refit=True,
    cv=inner_cv,
    n_jobs=-1,
    scoring="accuracy",
)

pipelines["ACM+MDM(Grid)"] = GridSearchCV(
    pipelines_["ACM+MDM(Grid)"],
    param_grid["ACM+MDM(Grid)"],
    refit=True,
    cv=inner_cv,
    n_jobs=-1,
    scoring="accuracy",
)

###############################################################################
# Evaluation
# ----------
#
results = []
for ppn, ppl in pipelines.items():
    cvclf = clone(ppl)
    score = cross_val_score(cvclf, X, y, cv=outer_cv, n_jobs=-1)
    res = {
        "score": score,
        "pipeline": ppn,
    }
    results.append(res)

results = pd.DataFrame(results)

# Flatten the 'score' lists into individual rows
flattened_results = results.explode('score')
flattened_results['score'] = pd.to_numeric(flattened_results['score'])

# Calculate the mean score and standard deviation for each pipeline
pipeline_stats = flattened_results.groupby('pipeline')['score'].agg(['mean', 'std']).sort_values(by='mean', ascending=False)

for pipeline, stats in pipeline_stats.iterrows():
    print(f"Pipeline: {pipeline}, Mean Score: {stats['mean']:.4f} +/- {stats['std']:.4f}")

###############################################################################
# Plot
# ----
order = ["ACM+TGSP+SVM(Grid)", "TGSP+SVM", "ACM+MDM(Grid)", "MDM"]

g = sns.catplot(
    data=flattened_results,
    x="pipeline",
    y="score",
    order=order,
    kind="bar",
    height=7,
    aspect=2,
)
plt.show()


###############################################################################
# References
# ----------
# .. [1] Carrara, I., & Papadopoulo, T. (2023). Classification of BCI-EEG based on augmented covariance matrix.
#        arXiv preprint arXiv:2302.04508.
#        https://doi.org/10.48550/arXiv.2302.04508
