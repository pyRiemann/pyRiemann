"""
=====================================================================
Sample from the Riemannian Gaussian distribution in the SPD manifold
=====================================================================

Spectral embedding of samples from the Riemannian Gaussian distribution
with different centerings and dispersions

"""
# Authors: Pedro Rodrigues <pedro.rodrigues@melix.org>
#
# License: BSD (3-clause)

from pyriemann.embedding import Embedding
from pyriemann.sampling import sample_gaussian_spd, generate_random_spd_matrix

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


print(__doc__)

###############################################################################
# Set parameters for sampling from the Riemannian Gaussian distribution
n_samples = 100  # how many samples to generate
n_dim = 4  # number of dimensions of the SPD matrices
sigma = 1.0  # dispersion of the Gaussian distribution
epsilon = 4.0  # parameter for controlling the distance between centers

# Generate the samples on three different conditions
Ybar = generate_random_spd_matrix(n_dim)  # random reference point
samples_1 = sample_gaussian_spd(n_samples=n_samples,
                                Ybar=Ybar,
                                sigma=sigma,
                                show_progress_bar=True)
samples_2 = sample_gaussian_spd(n_samples=n_samples,
                                Ybar=Ybar,
                                sigma=sigma/2,
                                show_progress_bar=True)
samples_3 = sample_gaussian_spd(n_samples=n_samples,
                                Ybar=epsilon*Ybar,
                                sigma=sigma,
                                show_progress_bar=True)

# Stack all of the samples into one data array for the embedding
samples = np.concatenate([samples_1, samples_2, samples_3])
labels = np.array(n_samples*[1] + n_samples*[2] + n_samples*[3])

###############################################################################
# Apply the spectral embedding over the SPD matrices
lapl = Embedding(metric='riemann', n_components=2)
embd = lapl.fit_transform(X=samples)

###############################################################################
# Plot the results

fig, ax = plt.subplots(figsize=(8, 6))

colors = {1: 'C0', 2: 'C1', 3: 'C2'}
for i in range(len(samples)):
    ax.scatter(embd[i, 0], embd[i, 1], c=colors[labels[i]], s=50)
ax.scatter([], [], c='C0', s=50, label=r'$\varepsilon = 1.00, \sigma = 1.00$')
ax.scatter([], [], c='C1', s=50, label=r'$\varepsilon = 1.00, \sigma = 0.50$')
ax.scatter([], [], c='C2', s=50, label=r'$\varepsilon = 4.00, \sigma = 1.00$')
ax.set_xticks([-1, -0.5, 0, 0.5, 1.0])
ax.set_xticklabels([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
ax.set_yticklabels([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
ax.set_title(r'Spectral embedding of data points (fixed $n_{dim} = 4$)',
             fontsize=14)
ax.set_xlabel(r'$\phi_1$', fontsize=14)
ax.set_ylabel(r'$\phi_2$', fontsize=14)
ax.legend()

plt.show()
