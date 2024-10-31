"""
====================================================================
Comparison of pipelines for transfer learning
====================================================================

We compare the classification performance of MDM on different strategies
for transfer learning.

These include re-centering the datasets as done in [1]_,
matching the statistical distributions in a semi-supervised way with Riemannian
Procrustes Analysis (RPA) [2]_,
improving the MDM classifier with a weighting scheme (MDWM) [3]_,
aligning vectors in tangent space by Procrustes analysis [4]_.

Matrices are simulated from a toy model based on the
Riemannian Gaussian distribution and the differences in statistics between
source and target distributions are determined by a set of parameters that have
control over the distance between the centers of each dataset, the angle of
rotation between the means of each class, and the differences in dispersion
of the matrices from each dataset.
"""

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline

from pyriemann.classification import MDM
from pyriemann.datasets.simulated import make_classification_transfer
from pyriemann.tangentspace import TangentSpace
from pyriemann.transfer import (
    TlSplitter,
    TlDummy,
    TlCenter,
    TlScale,
    TlRotate,
    TlClassifier,
    MDWM,
)


###############################################################################

# Choose seed for reproducible results
seed = 100

# We consider several types of pipeline for transfer learning:
# calibration : use only data from target-train partition for the classifier
# dummy : no transformation of dataset between the domains
# rct : re-center data from each domain to the Identity
# rpa : re-center, stretch and rotate (Riemannian Procrustes Analysis)
# mdwm : transfer learning with minimum distance to weighted mean
methods = ["calibration", "dummy", "rct", "rpa", "mdwm", "tsa"]
scores = {meth: [] for meth in methods}

# Create a dataset with two domains, each with two classes both datasets
# are generated by the same generative procedure with the SPD Gaussian
# and one of them is transformed by a matrix A, i.e. X <- A @ X @ A.T
X_enc, y_enc = make_classification_transfer(
    n_matrices=100,
    class_sep=0.75,
    class_disp=1.0,
    domain_sep=5.0,
    theta=3*np.pi/5,
    random_state=seed,
)

# Object for splitting the datasets into training and validation partitions
# the training set is composed of all matrices from the source domain
# plus a partition of the target domain whose size we can control
target_domain = "target_domain"
n_splits = 5  # how many times to split the target domain into train/test
tl_cv = TlSplitter(
    target_domain=target_domain,
    cv=StratifiedShuffleSplit(n_splits=n_splits, random_state=seed),
)

# Base classifier to consider in manifold
clf_base = MDM()

###############################################################################

# Vary the proportion of the target domain for training
target_train_frac_array = np.linspace(0.01, 0.20, 10)
for target_train_frac in tqdm(target_train_frac_array):

    # Change fraction of the target training partition
    tl_cv.cv.train_size = target_train_frac

    # Create dict for storing results of this particular CV split
    scores_cv = {meth: [] for meth in scores.keys()}

    # Carry out the cross-validation
    for train_idx, test_idx in tl_cv.split(X_enc, y_enc):

        # Split the dataset into training and testing
        X_enc_train, X_enc_test = X_enc[train_idx], X_enc[test_idx]
        y_enc_train, y_enc_test = y_enc[train_idx], y_enc[test_idx]

        # Calibration: use only data from target-train partition.
        # Classifier is trained only with matrices from the target domain.
        pipeline = make_pipeline(
            TlClassifier(
                target_domain=target_domain,
                estimator=clf_base,
                domain_weight={"source_domain": 0.0, "target_domain": 1.0},
            ),
        )

        pipeline.fit(X_enc_train, y_enc_train)
        scores_cv["calibration"].append(pipeline.score(X_enc_test, y_enc_test))

        # Dummy pipeline: no transfer learning at all.
        # Classifier is trained only with matrices from the source dataset.
        pipeline = make_pipeline(
            TlDummy(),
            TlClassifier(
                target_domain=target_domain,
                estimator=clf_base,
                domain_weight={"source_domain": 1.0, "target_domain": 0.0},
            ),
        )

        pipeline.fit(X_enc_train, y_enc_train)
        scores_cv["dummy"].append(pipeline.score(X_enc_test, y_enc_test))

        # RCT pipeline: recenter data from each domain to identity [1]_.
        # Classifier is trained only with matrices from the source domain.
        pipeline = make_pipeline(
            TlCenter(target_domain=target_domain),
            TlClassifier(
                target_domain=target_domain,
                estimator=clf_base,
                domain_weight={"source_domain": 1.0, "target_domain": 0.0},
            ),
        )

        pipeline.fit(X_enc_train, y_enc_train)
        scores_cv["rct"].append(pipeline.score(X_enc_test, y_enc_test))

        # RPA pipeline: recenter, stretch, and rotate [2]_.
        # Classifier is trained with matrices from source and target.
        pipeline = make_pipeline(
            TlCenter(target_domain=target_domain),
            TlScale(
                target_domain=target_domain,
                final_dispersion=1,
                centered_data=True,
            ),
            TlRotate(target_domain=target_domain, metric="euclid"),
            TlClassifier(
                target_domain=target_domain,
                estimator=clf_base,
                domain_weight={"source_domain": 0.5, "target_domain": 0.5},
            ),
        )

        pipeline.fit(X_enc_train, y_enc_train)
        scores_cv["rpa"].append(pipeline.score(X_enc_test, y_enc_test))

        # MDWM pipeline [3]_
        domain_tradeoff = 1 - np.exp(-25*target_train_frac)
        pipeline = MDWM(
            domain_tradeoff=domain_tradeoff,
            target_domain=target_domain,
            metric="riemann",
        )

        pipeline.fit(X_enc_train, y_enc_train)
        scores_cv["mdwm"].append(pipeline.score(X_enc_test, y_enc_test))

        # TSA pipeline: center, normalize and rotate in tangent space [4]_
        pipeline = make_pipeline(
            TangentSpace(metric="riemann"),
            TlCenter(target_domain=target_domain),
            TlScale(target_domain=target_domain),
            TlRotate(target_domain=target_domain),
            TlClassifier(
                target_domain=target_domain,
                estimator=LogisticRegression(),
                domain_weight={"source_domain": 0.5, "target_domain": 0.5},
            ),
        )

        pipeline.fit(X_enc_train, y_enc_train)
        scores_cv["tsa"].append(pipeline.score(X_enc_test, y_enc_test))

    # Get the average score of each pipeline
    for meth in scores.keys():
        scores[meth].append(np.mean(scores_cv[meth]))

# Store the results for each method on this particular seed
for meth in scores.keys():
    scores[meth] = np.array(scores[meth])

###############################################################################
# Plot the results, reproducing Figure 2 of [2]_.

fig, ax = plt.subplots(figsize=(6.7, 5.7))
for meth in scores.keys():
    ax.plot(
        target_train_frac_array,
        scores[meth],
        label=meth,
        lw=3.0,
    )
ax.legend(loc="lower right")
ax.set_ylim(0.5, 0.75)
ax.set_yticks([0.5, 0.6, 0.7])
ax.set_xlim(0.00, 0.21)
ax.set_xticks([0.01, 0.05, 0.10, 0.15, 0.20])
ax.set_xticklabels([1, 5, 10, 15, 20])
ax.set_xlabel("Percentage of training partition in target domain")
ax.set_ylabel("Classification accuracy")
ax.set_title("Comparison of transfer learning pipelines")

plt.show()


###############################################################################
# References
# ----------
# .. [1] `Transfer Learning: A Riemannian Geometry Framework With Applications
#    to Brain–Computer Interfaces
#    <https://hal.archives-ouvertes.fr/hal-01923278/>`_
#    P Zanini et al, IEEE Transactions on Biomedical Engineering, vol. 65,
#    no. 5, pp. 1107-1116, August, 2017
# .. [2] `Riemannian Procrustes analysis: transfer learning for
#    brain-computer interfaces
#    <https://hal.archives-ouvertes.fr/hal-01971856>`_
#    PLC Rodrigues et al, IEEE Transactions on Biomedical Engineering, vol. 66,
#    no. 8, pp. 2390-2401, December, 2018
# .. [3] `Transfer Learning for SSVEP-based BCI using Riemannian similarities
#    between users
#    <https://hal.uvsq.fr/hal-01911092>`_
#    E Kalunga et al, 26th European Signal Processing Conference (EUSIPCO 2018)
#    Sep 2018, Rome, Italy, pp.1685-1689
# .. [4] `Tangent space alignment: Transfer learning for brain-computer
#    interface
#    <https://www.frontiersin.org/articles/10.3389/fnhum.2022.1049985/pdf>`_
#    A. Bleuzé, J. Mattout and M. Congedo, Frontiers in Human Neuroscience,
#    2022
