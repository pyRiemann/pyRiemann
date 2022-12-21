"""
===============================================================================
Compare covariance and kernel estimators with different time windows
===============================================================================

Comparison of covariance estimators for different EEG signal lengths and their
impact on classification [1]_. Kernel estimators are also compared [2]_.
"""
# Authors: Sylvain Chevallier and Quentin Barthélemy
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

from pyriemann.estimation import Covariances, Kernels
from pyriemann.utils.distance import distance
from pyriemann.classification import MDM


###############################################################################
# Estimating covariance on synthetic data
# ----------------------------------------
#
# Generate synthetic data, sampled from a distribution considered as the
# groundtruth.

rs = np.random.RandomState(42)
n_matrices, n_channels, n_times = 10, 5, 1000
var = 2.0 + 0.1 * rs.randn(n_matrices, n_channels)
A = 2 * rs.rand(n_channels, n_channels) - 1
A /= np.linalg.norm(A, axis=1)[:, np.newaxis]
true_covs = np.empty(shape=(n_matrices, n_channels, n_channels))
X = np.empty(shape=(n_matrices, n_channels, n_times))
for i in range(n_matrices):
    true_covs[i] = A @ np.diag(var[i]) @ A.T
    X[i] = rs.multivariate_normal(
        np.array([0.0] * n_channels), true_covs[i], size=n_times
    ).T

###############################################################################
# `Covariances()` class offers several estimators:
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
        est_covs = Covariances(estimator=est).transform(X[:, :, :wl])
        dists = distance(est_covs, true_covs, metric="riemann")
        dfd.extend([dict(estimator=est, wlen=wl, dist=d) for d in dists])
dfd = pd.DataFrame(dfd)

###############################################################################

fig, ax = plt.subplots(figsize=(6, 4))
ax.set(xscale="log")
sns.lineplot(data=dfd, x="wlen", y="dist", hue="estimator", ax=ax)
ax.set_title("Distance to groundtruth covariance matrix")
ax.set_xlabel("Number of time samples")
ax.set_ylabel(r"$\delta(\Sigma, \hat{\Sigma})$")
plt.tight_layout()
plt.show()

###############################################################################
# Choice of estimator for motor imagery data
# ------------------------------------------
#
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
# Influence of shrinkage to estimate matrices
# -------------------------------------------
#
# Sample covariance matrix (SCM) estimation could lead to ill-conditionned
# matrices depending on the quality and quantity of EEG data available.
# Matrix condition number is the ratio between the highest and lowest
# eigenvalues: high values indicates ill-conditionned matrices that are not
# suitable for classification.
# A common approach to mitigate this issue is to regularize covariance matrices
# by shrinkage, like in Ledoit-Wolf, Schaefer-Strimmer or oracle estimators.
#
# In addition to covariance matrices, kernel matrices are computed for three
# kernel functions:
#
# - radial basis function (RBF),
# - polynomial,
# - Laplacian.

estimators = [
    "cov-lwf", "cov-oas", "cov-sch", "cov-scm",
    "ker-rbf", "ker-polynomial", "ker-laplacian",
]
tmin = -0.2
w_len = np.linspace(0.2, 2, 10)
n_matrices = 45
dfc = list()

for wl in w_len:
    X = Epochs(
        raw,
        events,
        event_ids,
        tmin,
        tmin + wl,
        picks=picks,
        preload=True,
        verbose=False,
    ).get_data()
    for est in estimators:
        est_class, est_param = est.split('-')
        if est_class == "ker":
            covs = Kernels(metric=est_param).transform(X)
        else:
            covs = Covariances(estimator=est_param).transform(X)
        evals, _ = np.linalg.eigh(covs)
        dfc.extend([dict(estimator=est, wlen=wl, cond=e[-1] / e[0])
                    for e in evals])
dfc = pd.DataFrame(dfc)

###############################################################################

fig, ax = plt.subplots(figsize=(6, 4))
ax.set(yscale="log")
sns.lineplot(data=dfc, x="wlen", y="cond", hue="estimator", ax=ax)
ax.set_title("Condition number of estimated matrices")
ax.set_xlabel("Epoch length (s)")
ax.set_ylabel(r"$\lambda_{\max}$/$\lambda_{\min}$")
plt.tight_layout()
plt.show()

###############################################################################
# Picking a good estimator for classification
# -------------------------------------------
#
# The choice of estimator have an impact on classification,
# especially when the matrices are estimated on short time windows.

tmin = 0.0
w_len = np.linspace(0.2, 2.0, 5)
n_matrices, n_splits = 45, 5
dfa = list()
sc = "balanced_accuracy"

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
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
        est_class, est_param = est.split('-')
        if est_class == "ker":
            clf = make_pipeline(Kernels(metric=est_param), MDM())
        else:
            clf = make_pipeline(Covariances(estimator=est_param), MDM())
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
plt.show()

###############################################################################
# References
# ----------
# .. [1] `Riemannian classification for SSVEP based BCI: offline versus online
#    implementations
#    <https://hal.archives-ouvertes.fr/hal-01739877>`_
#    S. Chevallier, E. Kalunga, Q. Barthélemy, F. Yger. Brain–Computer
#    Interfaces Handbook: Technological and Theoretical Advances, 2018.
# .. [2] `Beyond Covariance: Feature Representation with Nonlinear Kernel
#    Matrices
#    <https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Wang_Beyond_Covariance_Feature_ICCV_2015_paper.pdf>`_  # noqa
#    L. Wang, J. Zhang, L. Zhou, C. Tang, W Li. ICCV, 2015.
