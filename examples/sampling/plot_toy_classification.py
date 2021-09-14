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
from pyriemann.classification import MDM
from sklearn.model_selection import KFold
from tqdm import tqdm

from pyriemann.sampling import sample_gaussian_spd, generate_random_spd_matrix

np.random.seed(42)


print(__doc__)

###############################################################################
# Define helper functions for the example


def get_classification_score(clf, X, y):

    kf = KFold(n_splits=5)
    score = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        score.append(clf.score(X_test, y_test))
    score = np.mean(score)

    return score


def make_classification_problem(n_samples=100, n_dim=2, class_sep=1.0,
                                class_disp=1.0):

    # generate dataset for class 0
    CO = generate_random_spd_matrix(n_dim)
    X0 = sample_gaussian_spd(n_samples=n_samples, Ybar=CO, sigma=class_disp,
                             show_progress_bar=False)
    y0 = np.zeros(n_samples)

    # generate dataset for class 1
    epsilon = np.exp(class_sep/np.sqrt(n_dim))
    C1 = epsilon * CO
    X1 = sample_gaussian_spd(n_samples=n_samples, Ybar=C1, sigma=class_disp,
                             show_progress_bar=False)
    y1 = np.ones(n_samples)

    X = np.concatenate([X0, X1])
    y = np.concatenate([y0, y1])
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    return X, y

###############################################################################
# Set general parameters for the illustrations


n_dim = 4  # dimensionality of the data points
sigma = 1.00  # dispersion of the Gaussian distributions

###############################################################################
# Loop over different levels of separability between the classes
scores_array = []
deltas_array = np.linspace(0, 5*sigma, 10)
for delta in tqdm(deltas_array):

    # generate data points for a classification problem
    X, y = make_classification_problem(n_samples=250, n_dim=n_dim,
                                       class_sep=delta, class_disp=sigma)

    # which classifier to consider
    clf = MDM()

    # get the classification score for this setup
    scores_array.append(get_classification_score(clf, X, y))
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
