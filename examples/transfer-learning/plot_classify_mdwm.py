"""
=====================================================================
 Apply MDWM transfer learning to classify a set of matrices
=====================================================================

Applyin MDWM to learn from source matrices and classify target matrices.

"""
# Authors: Emmanuel Kalunga <emmanuelkalunga.k@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

from pyriemann.embedding import Embedding
from pyriemann.datasets import sample_gaussian_spd, generate_random_spd_matrix
from pyriemann.transferlearning import MDWM
from pyriemann.classification import MDM
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


print(__doc__)

###############################################################################
# Set parameters for sampling from the Riemannian Gaussian distribution
n_matrices_source = 100  # how many source SPD matrices to generate
n_matrices_target = 50  # how many target SPD matrices to generate
n_matrices_target_selected = 2  # target matrices available for training
n_dim = 4  # number of dimensions of the SPD matrices
sigma_source = 1.2  # dispersion of source Gaussian distribution
sigma_target = 1.0  # dispersion of target Gaussian distribution
epsilon_source = 3.0  # parameter for controlling the distance between centers
epsilon_target = 2.0  # parameter for controlling the distance between centers
random_state = 42  # random generator seed to ensure reproducibility
random_state_2 = 43  # random generator seed to ensure reproducibility

###############################################################################
# Generate two classes sample matrices for source and target subjects.
mean = generate_random_spd_matrix(n_dim, random_state=32)  # reference point

# Source subject
src_sub_cl_1 = sample_gaussian_spd(n_matrices=n_matrices_source,
                                   mean=mean,
                                   sigma=sigma_source,
                                   random_state=random_state)

src_sub_cl_2 = sample_gaussian_spd(n_matrices=n_matrices_source,
                                   mean=epsilon_source*mean,
                                   sigma=sigma_source,
                                   random_state=random_state_2)

# Target subject
trg_sub_cl_1 = sample_gaussian_spd(n_matrices=n_matrices_target,
                                   mean=epsilon_target*mean + 0.5,
                                   sigma=sigma_target,
                                   random_state=random_state)

trg_sub_cl_2 = sample_gaussian_spd(n_matrices=n_matrices_target,
                                   mean=epsilon_target*epsilon_source*mean+1.0,
                                   sigma=sigma_target,
                                   random_state=random_state)


###############################################################################
# Apply the spectral embedding over the SPD matrices

# Stack all of the samples into one data array for the embedding
samples = np.concatenate([src_sub_cl_1, src_sub_cl_2,
                          trg_sub_cl_1, trg_sub_cl_2
                          ])
labels = np.array(n_matrices_source*[1] + n_matrices_source*[2] +
                  n_matrices_target*[3] + n_matrices_target*[4])

lapl = Embedding(metric='riemann', n_components=2)
embd = lapl.fit_transform(X=samples)

###############################################################################
# Plot source and target matrices
# ---------------------------------------
#

fig, ax = plt.subplots(figsize=(8, 6))

colors = {1: 'C0', 2: 'C1', 3: 'C0', 4: 'C1'}
markers = {1: 'o', 2: 'o', 3: 'v', 4: 'v'}
source_alpha = 0.2
alphas = {1: source_alpha, 2: source_alpha, 3: 1, 4: 1}
# edgecolors = {1: 'none, 2: 'none', 3: 'face', 4: 'face'}
for i in range(len(samples)):
    ax.scatter(embd[i, 0], embd[i, 1], c=colors[labels[i]], s=81,
               marker=markers[labels[i]], alpha=alphas[labels[i]],
               edgecolors='none')
ax.scatter([], [], c='C0', s=50, marker='o', label=r'src class 1',
           alpha=source_alpha, edgecolors='none')
ax.scatter([], [], c='C1', s=50, marker='o', label=r'src class 2',
           alpha=source_alpha, edgecolors='none')
ax.scatter([], [], c='C0', s=50, marker='v', label=r'trg class 1',
           edgecolors='none')
ax.scatter([], [], c='C1', s=50, marker='v', label=r'trg class 2',
           edgecolors='none')
ax.set_xticks([-1, -0.5, 0, 0.5, 1.0])
ax.set_xticklabels([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
ax.set_yticklabels([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
ax.set_title(r'Embedded subjects matrices',
             fontsize=14)
ax.set_xlabel(r'$\phi_1$', fontsize=14)
ax.set_ylabel(r'$\phi_2$', fontsize=14)
_ = ax.legend()

###############################################################################
# The distribution of matrices from the target subject is shifted to the left.
# Consequently, the two subject cannot share the same classification boundary


###############################################################################
# Plot a sub-sample of the target matrices
# ---------------------------------------
#
# Assuming that only a limitted number of matrices is available from the
# target subject
embd_source = embd[:2*n_matrices_source, :]
embd_target = embd[2*n_matrices_source:, :]
labels_source = labels[:2*n_matrices_source]
labels_target = labels[2*n_matrices_source:]

# Draw n_matrices_target_selected matrices from target subject
rng = np.random.RandomState(random_state)
random_target_class_1 = rng.choice(np.arange(len(labels_target)//2),
                                   n_matrices_target_selected)
random_target_class_2 = rng.choice(np.arange(len(labels_target)//2,
                                   len(labels_target)),
                                   n_matrices_target_selected)
random_target_index = np.concatenate((random_target_class_1,
                                      random_target_class_2))
embd_sub_sample = np.concatenate((embd_source,
                                  embd_target[random_target_index, :]))
labels_sub_sample = np.concatenate((labels_source,
                                    labels_target[random_target_index]))

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(labels_sub_sample)):
    ax.scatter(embd_sub_sample[i, 0], embd_sub_sample[i, 1],
               c=colors[labels_sub_sample[i]], s=81,
               marker=markers[labels_sub_sample[i]], alpha=alphas[labels[i]],
               edgecolors='none')
ax.scatter([], [], c='C0', s=50, marker='o', label=r'src class 1',
           alpha=source_alpha, edgecolors='none')
ax.scatter([], [], c='C1', s=50, marker='o', label=r'src class 2',
           alpha=source_alpha, edgecolors='none')
ax.scatter([], [], c='C0', s=50, marker='v', label=r'trg class 1',
           edgecolors='none')
ax.scatter([], [], c='C1', s=50, marker='v', label=r'trg class 2',
           edgecolors='none')
ax.set_xticks([-1, -0.5, 0, 0.5, 1.0])
ax.set_xticklabels([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
ax.set_yticklabels([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
ax.set_title(r'Embedded subjects matrices sub-sample',
             fontsize=14)
ax.set_xlabel(r'$\phi_1$', fontsize=14)
ax.set_ylabel(r'$\phi_2$', fontsize=14)
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.15, 1.15)
_ = ax.legend()

###############################################################################
# Apply MDWM transfer learning for classification of target matrices
# ---------------------------------------

X_source = np.concatenate([src_sub_cl_1, src_sub_cl_2])
y_source = np.array(n_matrices_source*[1] + n_matrices_source*[2])

X_target = np.concatenate([trg_sub_cl_1, trg_sub_cl_2])
y_target = np.array(n_matrices_target*[1] + n_matrices_target*[2])

cv = StratifiedKFold(n_splits=50, shuffle=True, random_state=43)
coef_scores_array = []
transfer_coef_array = np.linspace(0, 1, 10)
for transfer_coef in transfer_coef_array:
    scores_array = []
    for test_idxs, train_idxs in cv.split(X_target, y_target):
        trans = MDWM(transfer_coef=transfer_coef)
        trans.fit(
            X=X_target[train_idxs, :], y=y_target[train_idxs],
            X_source=X_source, y_source=y_source,
            sample_weight=None
        )

        y_hat = trans.predict(X_target[test_idxs, :])
        scores_array.append(accuracy_score(y_target[test_idxs], y_hat))

    coef_scores_array.append(np.array(scores_array).mean())

###############################################################################
# Plot transfer classification results
fig, ax = plt.subplots(figsize=(7.5, 5.9))
ax.plot(transfer_coef_array, coef_scores_array, lw=3.0)
ax.set_xlabel(r'Transfer coefficient $\lambda$', fontsize=12)
ax.set_ylabel(r'score', fontsize=12)
ax.set_title(r'Classification score vs. transfer coefficient',
             fontsize=12)
_ = ax.grid(True)

###############################################################################
# Applying transfer (i.e. increasing transfer coeficcient) improves
# classification performance.


###############################################################################
# MDWM vs MDM Comparison
# -----------------------------------------------
# In this section we compare the performance of classification with
# MDWM transfer learning classification to performance of MDM classification.
# For MDW we consider 2 cases:
# (1) using all available source and target data for training, and
# (2) only using available target data for training

cv = StratifiedKFold(n_splits=50, shuffle=True, random_state=43)
transfer_coef = 0.43
scores_array_mdwm = []  # mdwm scores
scores_array_mdm = []  # mdm scores using both source and target data
scores_array_mdm_t = []  # mdm scores using only target data
for test_idxs, train_idxs in cv.split(X_target, y_target):

    trans = MDWM(transfer_coef=transfer_coef)
    trans.fit(
        X=X_target[train_idxs, :], y=y_target[train_idxs],
        X_source=X_source, y_source=y_source,
        sample_weight=None
    )
    y_hat_mdwm = trans.predict(X_target[test_idxs, :])
    scores_array_mdwm.append(accuracy_score(y_target[test_idxs], y_hat_mdwm))

    clf = MDM()
    clf.fit(X=np.concatenate([X_target[train_idxs, :], X_source]),
            y=np.concatenate([y_target[train_idxs], y_source]))
    y_hat_mdm = clf.predict(X_target[test_idxs, :])
    scores_array_mdm.append(accuracy_score(y_target[test_idxs], y_hat_mdm))

    clf_t = MDM()
    clf_t.fit(X=X_target[train_idxs, :], y=y_target[train_idxs])
    y_hat_mdm_t = clf_t.predict(X_target[test_idxs, :])
    scores_array_mdm_t.append(accuracy_score(y_target[test_idxs], y_hat_mdm_t))

###############################################################################
# Plot MDWM vs MDM results
fig, ax = plt.subplots(figsize=(7.5, 5.9))
ax.boxplot([scores_array_mdm, scores_array_mdm_t, scores_array_mdwm])

# ax.set_xlabel(r'Transfer coefficient $\lambda$', fontsize=12)
# ax.set_xticks([0, 1])
ax.set_xticklabels(['MDM(s+t)', 'MDM(t)', f"MDWM($\\lambda={transfer_coef}$)"],
                   fontsize=12)
ax.set_ylabel(r'score', fontsize=12)
ax.set_title(r'MDM vs. MDWM',
             fontsize=12)
_ = ax.grid(True)

plt.show()
