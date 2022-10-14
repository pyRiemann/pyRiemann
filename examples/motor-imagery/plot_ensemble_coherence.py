"""
====================================================================
Ensemble learning on functional connectivity
====================================================================

This example shows how to compute SPD matrices from functional
connectivity estimators and how to combine classification with
ensemble learning.
"""
# Authors: Sylvain Chevallier <sylvain.chevallier@uvsq.fr>,
#          Marie-Constance Corsi <marie.constance.corsi@gmail.com>
#
# License: BSD (3-clause)

from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pyriemann.classification import FgMDM
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP

from pyriemann.tangentspace import TangentSpace

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.exceptions import ConvergenceWarning
from warnings import filterwarnings

from coherence_helpers import (
    EnsureSPD,
    FunctionalTransformer,
    get_results,
)

###############################################################################
# Load EEG data
# -------------

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = 1.0, 2.0
event_id = dict(hands=2, feet=3)
subject = 7
# runs = [6, 10, 14]  # motor imagery: hands vs feet
runs = [4, 8]  # motor imagery: left vs right hand ,

raw_files = [read_raw_edf(f, preload=True)
             for f in eegbci.load_data(subject, runs)]
raw = concatenate_raws(raw_files)

picks = pick_types(raw.info, meg=False, eeg=True,
                   stim=False, eog=False, exclude="bads")
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
# nchan = len(epochs.ch_names)

# get epochs
X = 1e6 * epochs.get_data()

# Parameters
spectral_met = ["cov", "lagged", "instantaneous"]
fmin, fmax = 8, 35

###############################################################################
# Defining pipelines
# -------------------
#
# Compare CSP+SVM, fgMDM on covariance/coherence/imaginary coherence,
# and ensemble method

# Baseline evaluation
param_svm = {"kernel": ("linear", "rbf"), "C": [0.1, 1, 10]}
step_csp = [
    ("cov", Covariances(estimator="lwf")),
    ("csp", CSP(nfilter=6)),
    ("optsvm", GridSearchCV(SVC(), param_svm, cv=3)),
]

step_mdm = [("cov", Covariances(estimator="lwf")),
            ("fgmdm", FgMDM(metric="riemann", tsupdate=False))]

# Covariance-based Riemannian geometry
param_lr = {
    "penalty": "elasticnet",
    "l1_ratio": 0.15,
    "intercept_scaling": 1000.0,
    "solver": "saga",
}
step_cov = [("cov", Covariances(estimator="lwf")),
            ("tg", TangentSpace(metric="riemann")),
            ("LogReg", LogisticRegression(**param_lr))]

# Functional connectivity-based Riemannian geometry
param_ft = {"fmin": fmin, "fmax": fmax, "fs": fs}
step_fc = [("spd", EnsureSPD()),
           ("tg", TangentSpace(metric="riemann")),
           ("LogistReg", LogisticRegression(**param_lr))]
# step_fc = [("spd", EnsureSPD()),
#            ("fgmdm", FgMDM(metric="riemann", tsupdate=False))]


###############################################################################
# Evaluation
# ----------
#

filterwarnings(action="ignore", category=ConvergenceWarning)

dataset_res = list()

ppl_fc, ppl_ens, ppl_baseline = {}, {}, {}

# baseline pipeline
ppl_baseline["CSP+optSVM"] = Pipeline(steps=step_csp)
ppl_baseline["FgMDM"] = Pipeline(steps=step_mdm)

# functionnal connectivity pipeline
for sm in spectral_met:
    pname = sm + "+elasticnet"
    # pname = sm + "+fgmdm"
    if sm == "cov":
        ppl_fc[pname] = Pipeline(
            steps=[("cov", Covariances(estimator="lwf"))] + step_fc
        )
    else:
        ft = FunctionalTransformer(**param_ft, method=sm)
        ppl_fc[pname] = Pipeline(steps=[("ft", ft)] + step_fc)

# ensemble pipeline
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
all_ppl = {**ppl_baseline, **ppl_ens}

# Compute results
results = get_results(X, labels, all_ppl)
results = pd.DataFrame(results)


###############################################################################
# Plot
# ----
#

list_fc_ens = ["ensemble", "CSP+optSVM", "FgMDM"] + [
    sm + "+elasticnet" for sm in spectral_met
]

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
