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
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


print(__doc__)

###############################################################################
# Set parameters for sampling from the Riemannian Gaussian distribution
n_matrices_source = 100  # how many SPD matrices to generate
n_matrices_target = 50  # how many SPD matrices to generate
n_dim = 4  # number of dimensions of the SPD matrices
sigma_source = 1.2  # dispersion of the Gaussian distribution
sigma_target = 1.0  # dispersion of the Gaussian distribution
epsilon_source = 4.0  # parameter for controlling the distance between centers
epsilon_target = 3.0  # parameter for controlling the distance between centers
random_state = 42  # ensure reproducibility
random_state_2 = 43  # ensure reproducibility

# Generate the samples on three different conditions
mean = generate_random_spd_matrix(n_dim)  # random reference point
print(f"[LOG] mean: {mean}")

src_sub_cl_1 = sample_gaussian_spd(n_matrices=n_matrices_source,
                                   mean=mean,
                                   sigma=sigma_source,
                                   random_state=random_state)

src_sub_cl_2 = sample_gaussian_spd(n_matrices=n_matrices_source,
                                   mean=epsilon_source*mean,
                                   sigma=sigma_source,
                                   random_state=random_state_2)

trg_sub_cl_1 = sample_gaussian_spd(n_matrices=n_matrices_target,
                                   mean=epsilon_target*mean + 0.5,
                                   sigma=sigma_target,
                                   random_state=random_state)

trg_sub_cl_2 = sample_gaussian_spd(n_matrices=n_matrices_target,
                                   mean=epsilon_target*epsilon_source*mean + 1.0,
                                   sigma=sigma_target,
                                   random_state=random_state)

# Stack all of the samples into one data array for the embedding
samples = np.concatenate([src_sub_cl_1, src_sub_cl_2, trg_sub_cl_1, trg_sub_cl_2])
labels = np.array(n_matrices_source*[1] + n_matrices_source*[2] + n_matrices_target*[3] + n_matrices_target*[4])

###############################################################################
# Apply the spectral embedding over the SPD matrices
lapl = Embedding(metric='riemann', n_components=2)
embd = lapl.fit_transform(X=samples)

###############################################################################
# Plot the results

fig, ax = plt.subplots(figsize=(8, 6))

colors = {1: 'C0', 2: 'C1', 3: 'C0', 4: 'C1'}
markers = {1: 'o', 2: 'o', 3: 'v', 4: 'v'}
for i in range(len(samples)):
    ax.scatter(embd[i, 0], embd[i, 1], c=colors[labels[i]], s=81, marker=markers[labels[i]])
ax.scatter([], [], c='C0', s=50, marker='o', label=r'src class 1')
ax.scatter([], [], c='C1', s=50, marker='o', label=r'src class 2')
ax.scatter([], [], c='C0', s=50, marker='v', label=r'trg class 1')
ax.scatter([], [], c='C1', s=50, marker='v', label=r'trg class 2')
ax.set_xticks([-1, -0.5, 0, 0.5, 1.0])
ax.set_xticklabels([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
ax.set_yticklabels([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
ax.set_title(r'Embedded subjects matrices',
             fontsize=14)
ax.set_xlabel(r'$\phi_1$', fontsize=14)
ax.set_ylabel(r'$\phi_2$', fontsize=14)
ax.legend()

###############################################################################
# Plot the sub-sample of target subjet
n_matrices_target_selected = 10
embd_source = embd[:2*n_matrices_source,:]
embd_target = embd[2*n_matrices_source:,:]
labels_source = labels[:2*n_matrices_source]
labels_target = labels[2*n_matrices_source:]

rng = np.random.RandomState(random_state)
random_target_class_1 = rng.choice(np.arange(len(labels_target)//2), n_matrices_target_selected)
random_target_class_2 = rng.choice(np.arange(len(labels_target)//2, len(labels_target)), 
                                   n_matrices_target_selected)
random_target_index = np.concatenate((random_target_class_1, random_target_class_2))
print(f"[LOG] random_target_index: {random_target_index}")
embd_sub_sample = np.concatenate((embd_source, embd_target[random_target_index,:]))
labels_sub_sample = np.concatenate((labels_source,labels_target[random_target_index]))
print(f"[LOG] embd_sub_sample.shape: {embd_sub_sample.shape}")

fig, ax = plt.subplots(figsize=(8, 6))

colors = {1: 'C0', 2: 'C1', 3: 'C0', 4: 'C1'}
markers = {1: 'o', 2: 'o', 3: 'v', 4: 'v'}
for i in range(len(labels_sub_sample)):
    ax.scatter(embd_sub_sample[i, 0], embd_sub_sample[i, 1], 
               c=colors[labels_sub_sample[i]], s=81, marker=markers[labels_sub_sample[i]])
ax.scatter([], [], c='C0', s=50, marker='o', label=r'src class 1')
ax.scatter([], [], c='C1', s=50, marker='o', label=r'src class 2')
ax.scatter([], [], c='C0', s=50, marker='v', label=r'trg class 1')
ax.scatter([], [], c='C1', s=50, marker='v', label=r'trg class 2')
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
ax.legend()

trans = MDWM(transfer_coef=0.5)
trans.fit(np.concatenate([trg_sub_cl_1, trg_sub_cl_2]), 
          np.array(n_matrices_target*[1] + n_matrices_target*[2]), 
          np.concatenate([src_sub_cl_1, src_sub_cl_2]),
          np.array(n_matrices_source*[1] + n_matrices_source*[2]), 
          None)

y_hat = trans.predict(np.concatenate([trg_sub_cl_1, trg_sub_cl_2]))
acc = accuracy_score(np.array(n_matrices_target*[1] + n_matrices_target*[2]), y_hat)
print(f"acc:{acc}")


scores_array = []
for transfer_coef in np.linspace(0, 1, 10):
    trans = MDWM(transfer_coef=transfer_coef)
    trans.fit(np.concatenate([trg_sub_cl_1, trg_sub_cl_2]), 
            np.array(n_matrices_target*[1] + n_matrices_target*[2]), 
            np.concatenate([src_sub_cl_1, src_sub_cl_2]),
            np.array(n_matrices_source*[1] + n_matrices_source*[2]), 
            None)

    y_hat = trans.predict(np.concatenate([trg_sub_cl_1, trg_sub_cl_2]))
    acc = accuracy_score(np.array(n_matrices_target*[1] + n_matrices_target*[2]), y_hat)
    print(f"transfer_coef/acc:{transfer_coef}/{acc}")


# scores_array = np.array(scores_array)

plt.show()
