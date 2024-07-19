"""
====================================================================
Augmented Covariance Matrix
====================================================================

This example compares classification pipelines based on covariance maxtrix (CM)
versus augmented covariance matrix (ACM) [1]_.
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
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


###############################################################################
# Define the Augmented Covariance
# -------------------------------

class AugmentedDataset(BaseEstimator, TransformerMixin):
    """This transformation creates an embedding version of the current dataset.

    The implementation and the application is described in [1]_.
    """
    def __init__(self, order=1, lag=1):
        self.order = order
        self.lag = lag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.order == 1:
            return X

        X_new = np.concatenate(
            [
                X[:, :, p * self.lag: -(self.order - p) * self.lag]
                for p in range(0, self.order)
            ],
            axis=1,
        )

        return X_new


###############################################################################
# Load EEG data
# -------------

# Avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = 1., 2.
event_id = dict(hands=2, feet=3)
subject = 7
runs = [6, 10]  # motor imagery: hands vs feet

raw_files = [
    read_raw_edf(f, preload=True) for f in eegbci.load_data(
                                            subject,
                                            runs,
                                            update_path=True
                                           )
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
X = 1e6 * epochs.get_data(copy=False)
y = epochs.events[:, -1] - 2

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    shuffle=True,
                                                    stratify=y)


###############################################################################
# Defining pipelines
# ------------------
#
# Define pipelines based on augmented covariance matrix (ACM),
# which is an expansion of the current EEG signal using theory of phase space
# reconstruction [1]_

pipelines = {}
pipelines["ACM(Grid)+TGSP+SVM"] = Pipeline(steps=[
    ("augmenteddataset", AugmentedDataset()),
    ("Covariances", Covariances("oas")),
    ("Tangent_Space", TangentSpace(metric="riemann")),
    ("SVM", SVC(kernel="linear"))
])

pipelines["ACM(Grid)+MDM"] = Pipeline(steps=[
    ("augmenteddataset", AugmentedDataset()),
    ("Covariances", Covariances("oas")),
    ("MDM", MDM(metric=dict(mean='riemann', distance='riemann')))
])

# Define the inner CV scheme for the nested cross-validation of
# hyper-parameter search
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Define the hyper-parameters to test in the nested cross-validation
param_grid = {
    'augmenteddataset__order': [1, 2, 3, 4, 5, 6, 7],
    'augmenteddataset__lag': [1, 2, 3, 4, 5, 6, 7],
}

pipelines_grid = {}
pipelines_grid["ACM(Grid)+TGSP+SVM"] = GridSearchCV(
    pipelines["ACM(Grid)+TGSP+SVM"],
    param_grid,
    refit=True,
    cv=inner_cv,
    n_jobs=-1,
    scoring="accuracy",
)

pipelines_grid["ACM(Grid)+MDM"] = GridSearchCV(
    pipelines["ACM(Grid)+MDM"],
    param_grid,
    refit=True,
    cv=inner_cv,
    n_jobs=-1,
    scoring="accuracy",
)


###############################################################################
# Run the inner CV for getting the hyper-parameter
results_grid = []
best_estimator = []
for ppn, ppl in pipelines_grid.items():
    cvclf = clone(ppl)
    score = cvclf.fit(X_train, y_train)
    res = {
        "pipeline": ppn,
        "order": score.best_params_["augmenteddataset__order"],
        "lag": score.best_params_["augmenteddataset__lag"]
    }

    res_best = {
        "pipeline": ppn,
        "best_estimator": score.best_estimator_
    }
    best_estimator.append(res_best)
    results_grid.append(res)

results_grid = pd.DataFrame(results_grid)
best_estimator = pd.DataFrame(best_estimator)
print(results_grid)

# Update the pipeline with the best pipeline obtained with GridSearch process
pipelines["ACM(Grid)+TGSP+SVM"] = best_estimator.loc[
    best_estimator['pipeline'] == "ACM(Grid)+TGSP+SVM",
    "best_estimator"].values[0]

pipelines["ACM(Grid)+MDM"] = best_estimator.loc[
    best_estimator['pipeline'] == "ACM(Grid)+MDM",
    "best_estimator"].values[0]


###############################################################################
# Define pipelines with usual covariance matrix (CM)
pipelines["CM+TGSP+SVM"] = Pipeline(steps=[
    ("Covariances", Covariances("oas")),
    ("Tangent_Space", TangentSpace(metric="riemann")),
    ("SVM", SVC(kernel="linear"))
])

pipelines["CM+MDM"] = Pipeline(steps=[
    ("Covariances", Covariances("cov")),
    ("MDM", MDM(metric=dict(mean='riemann', distance='riemann')))
])


###############################################################################
# Evaluation
# ----------
#
# Compare classification pipelines:
#
# - CM+TGSP+SVM versus ACM(Grid)+TGSP+SVM
# - CM+MDM versus ACM(Grid)+MDM

results = []
for ppn, ppl in pipelines.items():
    cvclf = clone(ppl)
    cvclf.fit(X_train, y_train)
    y_pred = cvclf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    res = {
        "score": score,
        "pipeline": ppn,
        "Features": ppn.split('+')[0],
        "Classifiers": ppn.split('+', 1)[1]
    }
    results.append(res)
results = pd.DataFrame(results)

for _, row in results.sort_values(by='score', ascending=False).iterrows():
    print(f"{row.pipeline}, score: {row.score:.4f}")


###############################################################################
# Plot
# ----

sns.pointplot(
    data=results,
    x="Features",
    y="score",
    hue="Classifiers",
    order=["CM", "ACM(Grid)"]
)
plt.show()


###############################################################################
# References
# ----------
# .. [1] `Classification of BCI-EEG based on augmented covariance matrix
#    <https://doi.org/10.48550/arXiv.2302.04508>`_
#    Carrara, I., & Papadopoulo, T., arXiv, 2023
