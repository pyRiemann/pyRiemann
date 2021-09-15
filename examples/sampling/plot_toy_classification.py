"""
=====================================================================
Illustrate classification accuracy versus class separability
=====================================================================

Generate several datasets containing data points from two-classes. Each class
is generated with a Riemannian Gaussian distribution centered at the class mean
and with the same dispersion sigma. The distance between the class means is
parametrized by Delta, which we make vary between zero and 5*sigma. We
illustrate how the accuracy of the MDM classifier varies when Delta increases.

"""
# Authors: Pedro Rodrigues <pedro.rodrigues@melix.org>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from pyriemann.classification import MDM
from pyriemann.datasets import make_gaussian_blobs


print(__doc__)


###############################################################################
# Set general parameters for the illustrations


n_dim = 4  # dimensionality of the data points
sigma = 1.00  # dispersion of the Gaussian distributions
random_state = 42  # ensure reproducibility

###############################################################################
# Loop over different levels of separability between the classes
scores_array = []
deltas_array = np.linspace(0, 5*sigma, 10)

for delta in deltas_array:
    # generate data points for a classification problem
    X, y = make_gaussian_blobs(n_samples=250,
                               n_dim=n_dim,
                               class_sep=delta,
                               class_disp=sigma,
                               random_state=random_state)

    # which classifier to consider
    clf = MDM()

    # get the classification score for this setup
    scores_array.append(
        cross_val_score(clf, X, y, cv=5, scoring='roc_auc').mean())
scores_array = np.array(scores_array)

###############################################################################
# Plot the results
fig, ax = plt.subplots(figsize=(7.5, 5.9))
ax.plot(deltas_array, scores_array, lw=3.0, label=sigma)
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels([0, 1, 2, 3, 4, 5], fontsize=12)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=12)
ax.set_xlabel(r'$\Delta/\sigma$', fontsize=14)
ax.set_ylabel(r'score', fontsize=12)
ax.set_title(r'Classification score Vs class separability ($n_{dim} = 4$)',
             fontsize=12)
ax.grid(True)
ax.legend(loc='lower right', title=r'$\sigma$')

plt.show()
