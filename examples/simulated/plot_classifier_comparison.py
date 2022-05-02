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


@partial(np.vectorize, excluded=['clf'])
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
n_classifs = len(classifiers)

rs = np.random.RandomState(2022)
n_matrices, n_channels = 50, 2
y = np.concatenate([np.zeros(n_matrices), np.ones(n_matrices)])

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
        y
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
        y
    ),
    make_gaussian_blobs(
        2*n_matrices, n_channels, random_state=rs, class_sep=3., class_disp=0.5
    ),
    make_gaussian_blobs(
        2*n_matrices, n_channels, random_state=rs, class_sep=1., class_disp=0.7
    )
]
n_datasets = len(datasets)

cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])


###############################################################################

figure = plt.figure(figsize=(8, 10))
i = 1
# iterate over datasets
for ds_cnt, (X, y) in enumerate(datasets):
    # split dataset into training and test part
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min, x_max = X[:, 0, 0].min(), X[:, 0, 0].max()
    y_min, y_max = X[:, 0, 1].min(), X[:, 0, 1].max()
    z_min, z_max = X[:, 1, 1].min(), X[:, 1, 1].max()

    # just plot the dataset first
    ax = plt.subplot(n_datasets, n_classifs + 1, i, projection='3d')
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(
        X_train[:, 0, 0],
        X_train[:, 0, 1],
        X_train[:, 1, 1],
        c=y_train,
        cmap=cm_bright,
        edgecolors="k"
    )
    # Plot the testing points
    ax.scatter(
        X_test[:, 0, 0],
        X_test[:, 0, 1],
        X_test[:, 1, 1],
        c=y_test,
        cmap=cm_bright,
        alpha=0.6,
        edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xticklabels(())
    ax.set_yticklabels(())
    ax.set_zticklabels(())
    i += 1

    rx = np.arange(x_min, x_max, (x_max - x_min) / 50)
    ry = np.arange(y_min, y_max, (y_max - y_min) / 50)
    rz = np.arange(z_min, z_max, (z_max - z_min) / 50)

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(n_datasets, n_classifs + 1, i, projection='3d')
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundaries for horizontal 2D planes going through
        # the mean value of the third coordinates
        xx, yy = np.meshgrid(rx, ry)
        zz = get_proba(xx, yy, X[:, 1, 1].mean() * np.ones_like(xx), clf=clf)
        zz = np.ma.masked_where(~np.isfinite(zz), zz)
        ax.contourf(xx, yy, zz, zdir='z', offset=z_min, cmap=cm, alpha=0.5)

        xx, zz = np.meshgrid(rx, rz)
        yy = get_proba(xx, X[:, 0, 1].mean() * np.ones_like(xx), zz, clf=clf)
        yy = np.ma.masked_where(~np.isfinite(yy), yy)
        ax.contourf(xx, yy, zz, zdir='y', offset=y_max, cmap=cm, alpha=0.5)

        yy, zz = np.meshgrid(ry, rz)
        xx = get_proba(X[:, 0, 0].mean() * np.ones_like(yy), yy, zz, clf=clf)
        xx = np.ma.masked_where(~np.isfinite(xx), xx)
        ax.contourf(xx, yy, zz, zdir='x', offset=x_min, cmap=cm, alpha=0.5)

        # Plot the training points
        ax.scatter(
            X_train[:, 0, 0],
            X_train[:, 0, 1],
            X_train[:, 1, 1],
            c=y_train,
            cmap=cm_bright,
            edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            X_test[:, 0, 0],
            X_test[:, 0, 1],
            X_test[:, 1, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6
        )

        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            1.3 * x_max,
            y_min,
            z_min,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
            verticalalignment="bottom"
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_zticks(())

        i += 1

plt.show()


###############################################################################
# References
# ----------
# .. [1] https://scikit-learn.org/stable/auto_examples/classification/\
#    plot_classifier_comparison.html
