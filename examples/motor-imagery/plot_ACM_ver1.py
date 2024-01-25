"""
====================================================================
Augmented Covariance Method (ACM)
====================================================================

This example shows how to use the ACM classifier [1]_.
"""
# Authors: Igor Carrara <igor.carrara@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci

from sklearn.base import clone
from sklearn.model_selection import (StratifiedKFold,
                                     GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


###############################################################################
# Define the Augmented Dataset
# ----------------------------

class AugmentedDataset(BaseEstimator, TransformerMixin):
    """This transformation creates an embedding version of the current dataset.
    The implementation and the application is described in [1]_.
    """

    def __init__(self, order=1, lag=1):
        self.order = order
        self.lag = lag

    def fit(self, X, y):
        return self

    def transform(self, X):
        if self.order == 1:
            return X

        X_fin = []

        for i in np.arange(X.shape[0]):
            X_p = X[i][:, : -self.order * self.lag]
            X_p = np.concatenate(
                [X_p]
                + [
                    X[i][:, p * self.lag: -(self.order - p) * self.lag]
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


###############################################################################
# Defining cross-validation schemes
# ---------------------------------
#
# Define the inner CV scheme for implemented the cross validation for hyper-parameter search

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True, stratify=y)

###############################################################################
# Define the ACM pipeline, the approach is based on the expansion of the
# current EEG signal using theory of phase space reconstruction

pipelines = {}
pipelines["ACM+TGSP+SVM(Grid)"] = Pipeline(steps=[
    ("augmenteddataset", AugmentedDataset()),
    ("Covariances", Covariances("oas")),
    ("Tangent_Space", TangentSpace(metric="riemann")),
    ("SVM", SVC(kernel="linear"))
])

pipelines["ACM+MDM(Grid)"] = Pipeline(steps=[
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

pipelines_grid = {}
pipelines_grid["ACM+TGSP+SVM(Grid)"] = GridSearchCV(
    pipelines["ACM+TGSP+SVM(Grid)"],
    param_grid["ACM+TGSP+SVM(Grid)"],
    refit=True,
    cv=inner_cv,
    n_jobs=-1,
    scoring="accuracy",
)

pipelines_grid["ACM+MDM(Grid)"] = GridSearchCV(
    pipelines["ACM+MDM(Grid)"],
    param_grid["ACM+MDM(Grid)"],
    refit=True,
    cv=inner_cv,
    n_jobs=-1,
    scoring="accuracy",
)

# Run the inner CV for getting the hyper-parameter
results_grid = []
for ppn, ppl in pipelines_grid.items():
    cvclf = clone(ppl)
    score = cvclf.fit(X_train, y_train)
    res = {
        "pipeline": ppn,
        "order": score.best_params_["augmenteddataset__order"],
        "lag": score.best_params_["augmenteddataset__lag"]
    }
    results_grid.append(res)

results_grid = pd.DataFrame(results_grid)
print(results_grid)

# Update the pipeline with the best hyper-parameter
pipelines["ACM+TGSP+SVM(Grid)"].steps[0][1].order = results_grid.loc[results_grid['pipeline'] == "ACM+TGSP+SVM(Grid)", "order"].values[0]
pipelines["ACM+TGSP+SVM(Grid)"].steps[0][1].lag = results_grid.loc[results_grid['pipeline'] == "ACM+TGSP+SVM(Grid)", "lag"].values[0]

pipelines["ACM+MDM(Grid)"].steps[0][1].order = results_grid.loc[results_grid['pipeline'] == "ACM+MDM(Grid)", "order"].values[0]
pipelines["ACM+MDM(Grid)"].steps[0][1].lag = results_grid.loc[results_grid['pipeline'] == "ACM+MDM(Grid)", "lag"].values[0]

###############################################################################
# Defining pipelines
# ------------------
#
# Compare TGSP+SVM, MDM versus ACM+TGSP+SVM(Grid) and ACM+MDM(Grid)

# Define the standard pipeline TGSP+SVM as baseline with standard covariance
pipelines["TGSP+SVM"] = Pipeline(steps=[
    ("Covariances", Covariances("oas")),
    ("Tangent_Space", TangentSpace(metric="riemann")),
    ("SVM", SVC(kernel="linear"))
])

pipelines["MDM"] = Pipeline(steps=[
    ("Covariances", Covariances("cov")),
    ("MDM", MDM(metric=dict(mean='riemann', distance='riemann')))
])

###############################################################################
# Evaluation
# ----------

results = []
for ppn, ppl in pipelines.items():
    cvclf = clone(ppl)
    cvclf.fit(X_train, y_train)
    y_pred = cvclf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    res = {
        "score": score,
        "pipeline": ppn,
    }
    results.append(res)

results = pd.DataFrame(results)

# Calculate the mean score for each pipeline
pipeline_stats = results.groupby('pipeline')['score'].agg(
    ['mean']).sort_values(by='mean', ascending=False)

for pipeline, stats in pipeline_stats.iterrows():
    print(f"{pipeline}, score: {stats['mean']:.4f}")


###############################################################################
# Plot
# ----

order = ["ACM+TGSP+SVM(Grid)", "TGSP+SVM", "ACM+MDM(Grid)", "MDM"]

g = sns.catplot(
    data=results,
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
# .. [1] `Classification of BCI-EEG based on augmented covariance matrix
#    <https://doi.org/10.48550/arXiv.2302.04508>`_
#    Carrara, I., & Papadopoulo, T., arXiv, 2023
