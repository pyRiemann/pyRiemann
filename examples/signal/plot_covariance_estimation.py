"""
===============================================================================
Estimate covariance with different time windows
===============================================================================

Covariance estimators comparison for different EEG signal length and
impact on classification [1]_.
"""
# Author: Sylvain Chevallier
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.linalg import eigh
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance
from pyriemann.classification import MDM, TSclassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

###############################################################################
# Estimating covariance on synthetic data
# ----------------------------------------
#
# Generate synthetic data, sampled from a distribution considered as the
# groundtruth.

Nt, Nc, Ns = 10, 5, 1000
shape = (Nt, Nc, Nc)
var = 2.0 + 0.1 * np.random.randn(Nt, Nc)
A = 2 * np.random.rand(Nc, Nc) - 1
A /= np.atleast_2d(np.sqrt(np.sum(A ** 2, 1))).T
true_cov = np.empty(shape=shape)
X = np.empty(shape=shape)
for i in range(Nt):
    true_cov[i] = A @ np.diag(var[i]) @ A.T
    X[i] = np.random.multivariate_normal(np.array([0.0] * Nc), true_cov[i], size=Ns).T

###############################################################################
# Covariances() method offers several estimators: SCM, Ledoit-Wolf (LWF),
# Schaefer-Strimmer (SCH), oracle approximating shrunk covariance (OAS),
# minimum covariance determinant (MCD) and others. We will compare the
# distance of LWF, OAS and SCH estimators with the groundtruth, while
# increasing number of samples.

estimators = ["lwf", "oas", "sch"]
w_len = np.linspace(10, Ns, 100, dtype=np.int)
dist_true = np.zeros(shape=(len(estimators), len(w_len), Nt))
for i, e in enumerate(estimators):
    for j, t in enumerate(w_len):
        cov_est = Covariances(estimator=e).transform(X[:, :, :t])
        for k in range(Nt):
            dist_true[i, j, k] = distance(cov_est[k], true_cov[k], metric="riemann")

###############################################################################

fig, ax = plt.subplots(1)
fig.set_size_inches(10, 8)
for i, e in enumerate(estimators):
    ax.plot(w_len, np.median(dist_true[i, :, :], axis=1), "-", label=e.upper())
    ax.fill_between(
        w_len,
        np.percentile(dist_true[i, :, :], q=90, axis=1),
        np.percentile(dist_true[i, :, :], q=10, axis=1),
        alpha=0.5,
    )
ax.set_title("Distance to groundtruth covariance matrix")
ax.set_xlabel("Number of samples")
ax.set_ylabel(r"$\delta(\Sigma, \hat{\Sigma})$")
_ = ax.legend()

###############################################################################
# Choice of estimator for motor imagery data
# -------------------------------------------
# Loading data from PhysioNet MI dataset

event_id = dict(hands=2, feet=3)
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet
raw_files = [
    read_raw_edf(f, preload=True, stim_channel="auto")
    for f in eegbci.load_data(subject, runs)
]
raw = concatenate_raws(raw_files)
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

# subsample elecs
picks = picks[::2]
# Apply band-pass filter
raw.filter(7.0, 35.0, method="iir", picks=picks, skip_by_annotation="edge")
events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
event_ids = dict(hands=2, feet=3)
# events = find_events(raw, shortest_event=0, stim_channel=None)

###############################################################################
# Influence of shrinkage to estimate covariance
# -----------------------------------------------
# Sample covariance matrix (SCM) estimation could lead to ill-conditionned
# matrices depending to the quality and quantity of EEG sample available.
# Matrix condition number is the ratio between the highest andlowest
# eigenvalues: high values indicates ill-conditionned matrices that are not
# suitable for classification.
# A common approach to mitigate this issue is to regularize covariance matrices
# by shrinkage, like in Ledoit-Wolf, Schaefer-Strimmer or oracle estimator.

estimators = ["lwf", "oas", "scm", "sch"]
tmin = -0.2
w_len = np.linspace(0.2, 2, 10)
n_trials = 45
cond_nb = np.zeros(shape=(len(estimators), len(w_len), n_trials))

for i, e in enumerate(estimators):
    for j, t in enumerate(w_len):
        epochs = Epochs(
            raw,
            events,
            event_ids,
            tmin,
            tmin + t,
            picks=picks,
            preload=True,
            verbose=False,
        )
        cov = Covariances(estimator=e).transform(epochs.get_data())
        for k in range(len(cov)):
            w, _ = eigh(cov[k, :, :])
            cond_nb[i, j, k] = w[-1] / w[0]

###############################################################################

fig, ax = plt.subplots(1)
fig.set_size_inches(10, 8)
for i, e in enumerate(estimators):
    ax.semilogy(w_len, np.median(cond_nb[i, :, :], axis=1), "-o", label=e.upper())
    ax.fill_between(
        w_len,
        np.percentile(cond_nb[i, :, :], q=90, axis=1),
        np.percentile(cond_nb[i, :, :], q=10, axis=1),
        alpha=0.2,
    )
ax.set_title("Condition number of estimated covariance matrices")
ax.set_xlabel("Epoch length (s)")
ax.set_ylabel(r"$\lambda_{\min}$/$\lambda_{\max}$")
ax.legend()

###############################################################################
# Picking a good estimator for classification
# -----------------------------------------------
# The choice of covariance estimator have an impact on classification,
# especially when the covariances are estimated on short time windows.

estimators = ["lwf", "oas", "scm", "sch"]
tmin = -0.2
w_len = np.linspace(0.1, 1, 10)
n_trials, n_splits = 45, 10
acc = np.zeros(shape=(len(estimators), len(w_len), n_splits))

cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
for i, e in enumerate(estimators):
    clf = make_pipeline(Covariances(estimator=e), TSclassifier(metric="riemann"))
    for j, t in enumerate(w_len):
        epochs = Epochs(
            raw,
            events,
            event_ids,
            tmin,
            tmin + t,
            proj=True,
            picks=picks,
            baseline=None,
            preload=True,
            verbose=False,
        )
        X, y = epochs.get_data(), list()
        ev = epochs.events
        for k in range(len(ev)):
            if ev[k, -1] == 2:
                y.append(0)
            elif ev[k, -1] == 3:
                y.append(1)
        y = np.array(y)
        try:
            acc[i, j, :] = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        except ValueError:
            print(
                "Wrong covariance estimation with ",
                e,
                ":",
                t,
                "s not sufficent to estimate a PSD matrix",
            )
            acc[i, j, :] = [np.nan] * n_splits

###############################################################################

fig, ax = plt.subplots(1)
fig.set_size_inches(10, 8)
for i, e in enumerate(estimators):
    ax.plot(w_len, np.mean(acc[i, :, :], axis=1), "-o", label=e.upper())
#    ax.fill_between(w_len, np.percentile(acc[i, :, :], q=75, axis=1),
#                    np.percentile(acc[i, :, :], q=25, axis=1), alpha=0.2)
ax.set_title("Accuracy for different estimator and epoch length")
ax.set_xlabel("Epoch length (s)")
ax.set_ylabel(r"Accuracy (\%)")
ax.legend()

###############################################################################
# References
# ----------
# .. [1] S. Chevallier, E. Kalunga, Q. Barthélemy, F. Yger. "Riemannian
# classification for SSVEP based BCI: offline versus online implementations."
# Brain–Computer Interfaces Handbook: Technological and Theoretical Advances,
# 2018.
