"""
=====================================================================
Visualize SPD dataset using the Riemannian t-SNE algorithm.
=====================================================================

Using the Riemannian t-SNE to visualize samples from 
the Riemannian Gaussian distribution with different centerings and dispersions.
More details on the Riemannian t-SNE can be found in [1]_.


"""
# Authors: Thibault de Surrel <thibault.de-surrel@lamsade.dauphine.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

from pyriemann.embedding import tSNE
from pyriemann.datasets import make_matrices, sample_gaussian_spd


print(__doc__)

###############################################################################
# Set parameters for sampling from the Riemannian Gaussian distribution

n_matrices_per_class = 50  # how many SPD matrices to generate in each class
n_dim = 5  # number of dimensions of the SPD matrices
sigma = 1.0  # dispersion of the Gaussian distribution
epsilon = 4.0  # parameter for controlling the distance between centers
random_state = 42  # ensure reproducibility

# Generate the samples on three different conditions
mean = make_matrices(1, n_dim, "spd")[0]  # random reference point

samples_1 = sample_gaussian_spd(n_matrices=n_matrices_per_class,
                                mean=mean,
                                sigma=sigma,
                                random_state=random_state)
samples_2 = sample_gaussian_spd(n_matrices=n_matrices_per_class,
                                mean=mean,
                                sigma=sigma/2,
                                random_state=random_state)

# Stack all of the samples into one data array for the embedding
samples = np.concatenate([samples_1, samples_2])
labels = np.array(n_matrices_per_class*[1] + n_matrices_per_class*[2])
n_total_matrices = samples.shape[0]
###############################################################################
# Apply the tSNE over the SPD matrices

tSNE_ = tSNE(n_components=2, perplexity = int(0.75*n_total_matrices), verbosity = 0, max_it=10000, max_time=2000)
embd = tSNE_.fit_transform(X=samples)

###############################################################################
# Plot the results. A dynamic display is required if you want to rotate or zoom the 3D figure.
# This 3D plot can be tricky to interpret. 2x2 SPD matrices can be viewed as
# spatial coordinates contained in a hyper-cone. 

fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')

colors = {1: "C0", 2: "C1"}
for i in range(len(samples)):
    ax.scatter(embd[i, 0,0], embd[i, 0,1], embd[i,1,1], c=colors[labels[i]], s=50)
ax.scatter([], [], c="C0", s=50, label=r"$\varepsilon = 1.00, \sigma = 1.00$")
ax.scatter([], [], c="C1", s=50, label=r"$\varepsilon = 1.00, \sigma = 0.50$")
ax.legend()
ax.set_xlabel("a", fontsize=14)
ax.set_ylabel("b", fontsize=14)
ax.set_zlabel("c", fontsize=14)
ax.set_title("The reduced version of the data points using the t-SNE")

plt.show()

###############################################################################
# References
# ----------
# .. [1] `Geometry-Aware visualization of high dimensional Symmetric 
#    Positive Definite matrices
#    <https://openreview.net/pdf?id=DYCSRf3vby>`_
#    T. de Surrel, S. Chevallier, F. Lotte and F. Yger. 
#    Transactions on Machine Learning Research, 2025