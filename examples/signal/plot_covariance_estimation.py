"""
===============================================================================
Estimate covariance with different time windows
===============================================================================

Covariance estimators comparison for different EEG signal lengths and their
impact on classification [1]_.
"""
# Author: Sylvain Chevallier
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline

from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance
from pyriemann.classification import MDM

rs = np.random.RandomState(42)


###############################################################################
# Estimating covariance on synthetic data
# ----------------------------------------
#
# Generate synthetic data, sampled from a distribution considered as the
# groundtruth.

n_matrices, n_channels, n_times = 10, 5, 1000
var = 2.0 + 0.1 * rs.randn(n_matrices, n_channels)
A = 2 * rs.rand(n_channels, n_channels) - 1
A /= np.linalg.norm(A, axis=1)[:, np.newaxis]
true_cov = np.empty(shape=(n_matrices, n_channels, n_channels))
X = np.empty(shape=(n_matrices, n_channels, n_times))
for i in range(n_matrices):
    true_cov[i] = A @ np.diag(var[i]) @ A.T
    X[i] = rs.multivariate_normal(
        np.array([0.0] * n_channels), true_cov[i], size=n_times
    ).T

###############################################################################
# `Covariances()` object offers several estimators:
#
# - sample covariance matrix (SCM),
# - Ledoit-Wolf (LWF),
# - Schaefer-Strimmer (SCH),
# - oracle approximating shrunk (OAS) covariance,
# - minimum covariance determinant (MCD),
# - and others.
#
# We will compare the distance of LWF, OAS and SCH estimators with the
# groundtruth, while increasing epoch length.

estimators = ["lwf", "oas", "sch"]
w_len = np.linspace(10, n_times, 20, dtype=int)
dfd = list()
for est in estimators:
    for wl in w_len:
        cov_est = Covariances(estimator=est).transform(X[:, :, :wl])
        for k in range(n_matrices):
            dist = distance(cov_est[k], true_cov[k], metric="riemann")
            dfd.append(dict(estimator=est, wlen=wl, dist=dist))
dfd = pd.DataFrame(dfd)

###############################################################################

fig, ax = plt.subplots(figsize=(6, 4))
ax.set(xscale="log")
sns.lineplot(data=dfd, x="wlen", y="dist", hue="estimator", ax=ax)
ax.set_title("Distance to groundtruth covariance matrix")
ax.set_xlabel("Number of time samples")
ax.set_ylabel(r"$\delta(\Sigma, \hat{\Sigma})$")
plt.tight_layout()

###############################################################################
# Choice of estimator for motor imagery data
# -------------------------------------------
# Loading data from PhysioNet MI dataset, for subject 1.

event_id = dict(hands=2, feet=3)
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet
raw_files = [
    read_raw_edf(f, preload=True, stim_channel="auto")
    for f in eegbci.load_data(subject, runs)
]
raw = concatenate_raws(raw_files)
picks = pick_types(raw.info, eeg=True, exclude="bads")

# subsample elecs
picks = picks[::2]
# Apply band-pass filter
raw.filter(7.0, 35.0, method="iir", picks=picks, skip_by_annotation="edge")
events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
event_ids = dict(hands=2, feet=3)

###############################################################################
# Influence of shrinkage to estimate covariance
# -----------------------------------------------
# Sample covariance matrix (SCM) estimation could lead to ill-conditionned
# matrices depending on the quality and quantity of EEG data available.
# Matrix condition number is the ratio between the highest and lowest
# eigenvalues: high values indicates ill-conditionned matrices that are not
# suitable for classification.
# A common approach to mitigate this issue is to regularize covariance matrices
# by shrinkage, like in Ledoit-Wolf, Schaefer-Strimmer or oracle estimators.

estimators = ["lwf", "oas", "sch", "scm"]
tmin = -0.2
w_len = np.linspace(0.2, 2, 10)
n_matrices = 45
dfc = list()

for wl in w_len:
    epochs = Epochs(
        raw,
        events,
        event_ids,
        tmin,
        tmin + wl,
        picks=picks,
        preload=True,
        verbose=False,
    )
    for est in estimators:
        cov = Covariances(estimator=est).transform(epochs.get_data())
        for k in range(len(cov)):
            ev, _ = np.linalg.eigh(cov[k, :, :])
            dfc.append(dict(estimator=est, wlen=wl, cond=ev[-1] / ev[0]))
dfc = pd.DataFrame(dfc)

###############################################################################

fig, ax = plt.subplots(figsize=(6, 4))
ax.set(yscale="log")
sns.lineplot(data=dfc, x="wlen", y="cond", hue="estimator", ax=ax)
ax.set_title("Condition number of estimated covariance matrices")
ax.set_xlabel("Epoch length (s)")
ax.set_ylabel(r"$\lambda_{\max}$/$\lambda_{\min}$")
plt.tight_layout()

###############################################################################
# Picking a good estimator for classification
# -----------------------------------------------
# The choice of covariance estimator have an impact on classification,
# especially when the covariances are estimated on short time windows.

estimators = ["lwf", "oas", "sch", "scm"]
tmin = 0.0
w_len = np.linspace(0.2, 2.0, 5)
n_matrices, n_splits = 45, 3
dfa = list()
sc = "balanced_accuracy"

cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
for wl in w_len:
    epochs = Epochs(
        raw,
        events,
        event_ids,
        tmin,
        tmin + wl,
        proj=True,
        picks=picks,
        preload=True,
        baseline=None,
        verbose=False,
    )
    X = epochs.get_data()
    y = np.array([0 if ev == 2 else 1 for ev in epochs.events[:, -1]])
    for est in estimators:
        clf = make_pipeline(Covariances(estimator=est), MDM())
        try:
            score = cross_val_score(clf, X, y, cv=cv, scoring=sc)
            dfa += [dict(estimator=est, wlen=wl, accuracy=sc) for sc in score]
        except ValueError:
            print(f"{est}: {wl} is not sufficent to estimate a SPD matrix")
            dfa += [dict(estimator=est, wlen=wl, accuracy=np.nan)] * n_splits
dfa = pd.DataFrame(dfa)

###############################################################################

fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(
    data=dfa,
    x="wlen",
    y="accuracy",
    hue="estimator",
    style="estimator",
    ax=ax,
    ci=None,
    markers=True,
    dashes=False,
)
ax.set_title("Accuracy for different estimators and epoch lengths")
ax.set_xlabel("Epoch length (s)")
ax.set_ylabel("Classification accuracy")
plt.tight_layout()

###############################################################################
# References
# ----------
# .. [1] `Riemannian classification for SSVEP based BCI: offline versus online
#    implementations
#    <https://hal.archives-ouvertes.fr/hal-01739877>`_
#    S. Chevallier, E. Kalunga, Q. Barthélemy, F. Yger. Brain–Computer
#    Interfaces Handbook: Technological and Theoretical Advances, 2018.
