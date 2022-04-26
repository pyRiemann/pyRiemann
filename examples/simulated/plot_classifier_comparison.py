"""
===============================================================================
Classifier comparison
===============================================================================

A comparison of several Riemannian classifiers on low-dimensional synthetic
datasets, adapted to SPD matrices from [1]_.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.

"""
# Authors: Quentin Barthélemy
#
# License: BSD (3-clause)

from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

from pyriemann.datasets import make_covariances, make_gaussian_blobs
from pyriemann.classification import MDM, KNearestNeighbor


###############################################################################


@partial(np.vectorize, excluded=['cov_11', 'clf'])
def get_proba(cov_00, cov_01, cov_11, clf):
    cov = np.array([[cov_00, cov_01], [cov_01, cov_11]])
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return clf.predict_proba(cov[np.newaxis, ...])[0, 1]


###############################################################################

names = [
    "MDM",
    "Nearest Neighbors"
]
classifiers = [
    MDM(),
    KNearestNeighbor(n_neighbors=3)
]

rs = np.random.RandomState(2022)
n_matrices, n_channels = 50, 2

datasets = [
    (
        np.concatenate([
            make_covariances(
                n_matrices, n_channels, rs, evals_mean=10, evals_std=1
            ),
            make_covariances(
                n_matrices, n_channels, rs, evals_mean=15, evals_std=1
            )
        ]),
        np.concatenate([np.zeros(n_matrices), np.ones(n_matrices)])
    ),
    (
        np.concatenate([
            make_covariances(
                n_matrices, n_channels, rs, evals_mean=10, evals_std=2
            ),
            make_covariances(
                n_matrices, n_channels, rs, evals_mean=12, evals_std=2
            )
        ]),
        np.concatenate([np.zeros(n_matrices), np.ones(n_matrices)])
    ),
    make_gaussian_blobs(
        2*n_matrices, n_channels, random_state=rs, class_sep=3., class_disp=0.5
    ),
    make_gaussian_blobs(
        2*n_matrices, n_channels, random_state=rs, class_sep=1., class_disp=0.7
    )
]


###############################################################################

figure = plt.figure(figsize=(8, 10))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # split dataset into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min, x_max = X[:, 0, 0].min() - 0.5, X[:, 0, 0].max() + 0.5
    y_min, y_max = X[:, 0, 1].min() - 0.5, X[:, 0, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, (x_max - x_min) / 50),
        np.arange(y_min, y_max, (y_max - y_min) / 50)
    )

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(
        X_train[:, 0, 0],
        X_train[:, 0, 1],
        c=y_train,
        cmap=cm_bright,
        edgecolors="k"
    )
    # Plot the testing points
    ax.scatter(
        X_test[:, 0, 0],
        X_test[:, 0, 1],
        c=y_test,
        cmap=cm_bright,
        alpha=0.6,
        edgecolors="k"
    )
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary for the horizontal 2D plane going through
        # the mean value of the third coordinates
        Z = get_proba(xx, yy, cov_11=X[:, 1, 1].mean(), clf=clf)
        Z = np.ma.masked_where(~np.isfinite(Z), Z)

        # Put the result into a color plot
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

        # Plot the training points
        ax.scatter(
            X_train[:, 0, 0],
            X_train[:, 0, 1],
            c=y_train,
            cmap=cm_bright,
            edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            X_test[:, 0, 0],
            X_test[:, 0, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6
        )

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            xx.max() - 0.3,
            yy.min() + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

plt.tight_layout()
plt.show()


###############################################################################
# References
# ----------
# .. [1] https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
