"""
====================================================================
Ensemble learning on functional connectivity
====================================================================

This example shows how to compute SPD matrices from functional
connectivity estimators and how to combine classification with
ensemble learning [1]_.
"""
# Authors: Sylvain Chevallier <sylvain.chevallier@universite-paris-saclay.fr>,
#          Marie-Constance Corsi <marie.constance.corsi@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from pyriemann.classification import FgMDM
from pyriemann.estimation import Coherences, Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from helpers.coherence_helpers import NearestSPD, get_results


###############################################################################
# Define connectivity transformer
# -------------------------------
#
# This estimator computes the functional connectivity from input signal using
# `pyriemann.estimation.Coherences`


class Connectivities(TransformerMixin, BaseEstimator):
    """Getting connectivity features from epoch"""

    def __init__(self, method="ordinary", fmin=8, fmax=35, fs=None):
        self.method = method
        self.fmin = fmin
        self.fmax = fmax
        self.fs = fs

    def fit(self, X, y=None):
        self._coh = Coherences(
            coh=self.method,
            fmin=self.fmin,
            fmax=self.fmax,
            fs=self.fs,
        )
        return self

    def transform(self, X):
        X_coh = self._coh.fit_transform(X)
        X_con = np.mean(X_coh, axis=-1, keepdims=False)
        return X_con


###############################################################################
# Load EEG data
# -------------

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = 1.0, 2.0
event_id = dict(hands=2, feet=3)
subject = 7
runs = [4, 8]  # motor imagery: left vs right hand

raw_files = [
    read_raw_edf(f, preload=True) for f in eegbci.load_data(subject, runs)
]
raw = concatenate_raws(raw_files)

picks = pick_types(
    raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
)
# subsample elecs
picks = picks[::2]

# Apply band-pass filter
raw.filter(7.0, 35.0, method="iir", picks=picks)

events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

# Read epochs (train will be done only between 1 and 2s)
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
    verbose=False,
)
labels = epochs.events[:, -1] - 2
fs = epochs.info["sfreq"]
X = 1e6 * epochs.get_data()


###############################################################################
# Defining pipelines
# -------------------
#
# Compare CSP+SVM, FgMDM on covariance, tangent space logistic regression with
# covariance, lag coherence, and instantaneous coherence, along with ensemble
# method

ppl_baseline, ppl_fc, ppl_ens = {}, {}, {}

###############################################################################
# Baseline algorithms are CSP with optimal SVM and FgMDM based on covariances

param_svm = {"kernel": ("linear", "rbf"), "C": [0.1, 1, 10]}
step_csp = [
    ("cov", Covariances(estimator="lwf")),
    ("csp", CSP(nfilter=6)),
    ("optsvm", GridSearchCV(SVC(), param_svm, cv=3)),
]
ppl_baseline["CSP+optSVM"] = Pipeline(steps=step_csp)

step_mdm = [
    ("cov", Covariances(estimator="lwf")),
    ("fgmdm", FgMDM(metric="riemann", tsupdate=False)),
]
ppl_baseline["FgMDM"] = Pipeline(steps=step_mdm)

###############################################################################
# Functional connectivity pipelines use logistic regression in tangent space.
# They will be estimated from covariance, lagged coherence and instantaneous
# coherence.

spectral_met = ["cov", "lagged", "instantaneous"]
fmin, fmax = 8, 35
param_lr = {
    "penalty": "elasticnet",
    "l1_ratio": 0.15,
    "intercept_scaling": 1000.0,
    "solver": "saga",
}
param_ft = {"fmin": fmin, "fmax": fmax, "fs": fs}
step_fc = [
    ("spd", NearestSPD()),
    ("tg", TangentSpace(metric="riemann")),
    ("LogistReg", LogisticRegression(**param_lr)),
]
for sm in spectral_met:
    pname = sm + "+elasticnet"
    if sm == "cov":
        ppl_fc[pname] = Pipeline(
            steps=[("cov", Covariances(estimator="lwf"))] + step_fc
        )
    else:
        ft = Connectivities(**param_ft, method=sm)
        ppl_fc[pname] = Pipeline(steps=[("ft", ft)] + step_fc)

###############################################################################
# The ensemble classifier stacks a logistic regression on top of the three
# functional connectivity pipelines to make a global prediction

fc_estim = [(n, ppl_fc[n]) for n in ppl_fc]
cvkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr = LogisticRegression(**param_lr)
ppl_ens["ensemble"] = StackingClassifier(
    estimators=fc_estim,
    cv=cvkf,
    n_jobs=1,
    final_estimator=lr,
    stack_method="predict_proba",
)

###############################################################################
# Evaluation
# ----------

dataset_res = list()
all_ppl = {**ppl_baseline, **ppl_ens}

# Compute results
results = get_results(X, labels, all_ppl)
results = pd.DataFrame(results)


###############################################################################
# Plot
# ----

list_fc_ens = ["ensemble", "CSP+optSVM", "FgMDM"] + \
    [sm + "+elasticnet" for sm in spectral_met]

g = sns.catplot(
    data=results,
    x="pipeline",
    y="score",
    kind="bar",
    order=list_fc_ens,
    height=7,
    aspect=2,
)
plt.show()


###############################################################################
# References
# ----------
# .. [1] `Functional connectivity ensemble method to enhance BCI performance
#    (FUCONE)
#    <https://arxiv.org/abs/2111.03122>`_
#    Corsi, M.-C., Chevallier, S., De Vico Fallani, F. & Yger, F. IEEE TBME,
#    2022
