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

from pyriemann.embedding import TSNE
from pyriemann.datasets import make_matrices, sample_gaussian_spd


print(__doc__)

###############################################################################
# Set parameters for sampling from the Riemannian Gaussian distribution

n_matrices_per_class = 50  # how many SPD matrices to generate in each class
n_dim = 6  # number of dimensions of the SPD matrices
sigma = 1.0  # dispersion of the Gaussian distribution
epsilon = 4.0  # parameter for controlling the distance between centers
random_state = 42  # ensure reproducibility

# Generate the samples on three different conditions
# random reference points
means = make_matrices(2, n_dim, "spd", rs=random_state)

samples_1 = sample_gaussian_spd(
    n_matrices=n_matrices_per_class,
    mean=means[0],
    sigma=sigma,
    random_state=random_state,
)
samples_2 = sample_gaussian_spd(
    n_matrices=n_matrices_per_class,
    mean=means[1],
    sigma=sigma / 2,
    random_state=random_state,
)

# Stack all of the samples into one data array for the embedding
samples = np.concatenate([samples_1, samples_2])
labels = np.array(n_matrices_per_class * [1] + n_matrices_per_class * [2])
n_total_matrices = samples.shape[0]
###############################################################################
# Apply the t-SNE over the SPD matrices

n_components = 2  # Dimension of the SPD matrices in the output space
perplexity = int(
    0.75 * n_total_matrices
)  # Perplexity parameter for the t-SNE (recommended to be 0.75*n_samples)
metric = "riemann"  # Metric used to compute the distances between SPD matrices
# can be "riemann" or "logeuclid"
verbosity = 0  # Verbosity level of the t-SNE
max_it = 10000  # Maximum number of iterations
max_time = 60  # Maximum time for the computation in seconds

TSNE_ = TSNE(
    n_components=n_components,
    perplexity=perplexity,
    metric=metric,
    verbosity=2,
    max_it=max_it,
    max_time=max_time,
    random_state=random_state,
)
embd = TSNE_.fit_transform(X=samples)

###############################################################################
# Plot the results. A dynamic display is required if you want to rotate
# or zoom the 3D figure. This 3D plot can be tricky to interpret.
# 2x2 SPD matrices can be viewed as spatial coordinates contained in
# a hyper-cone. We can see that the two Gaussians are well reduced,
# with two different means and the second Gaussian having a smaller dispersion.

fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection="3d")

colors = {1: "C0", 2: "C1"}
for i in range(len(samples)):
    ax.scatter(embd[i, 0, 0], embd[i, 0, 1], embd[i, 1, 1],
               c=colors[labels[i]], s=50)
ax.scatter([], [], c="C0", s=50, label=r"First Gaussian")
ax.scatter([], [], c="C1", s=50, label=r"Second Gaussian")
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
