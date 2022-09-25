"""
====================================================================
Comparison of pipelines for transfer learning
====================================================================

Compare the classificaton performance of four pipelines for transfer learning.
The data points are all simulated from a toy model based on the Riemannian
Gaussian distribution.
"""

import warnings
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit

from pyriemann.classification import MDM
from pyriemann.datasets.simulated import make_classification_transfer
from pyriemann.transfer import (
    TLSplitter,
    TLDummy,
    TLCenter,
    TLStretch,
    TLRotate,
    TLEstimator,
    # TLMDM
)

warnings.filterwarnings("ignore")


###############################################################################

# Choose seed for reproducible results
seed = 100

# We consider several types of pipeline for transfer learning
# dct : no transformation of dataset between the domains
# rct : re-center the data points from each domain to the Identity
# rpa : re-center, stretch and rotate (Riemannian Procrustes Analysis)
# tlmdm : ???
# calibration : use only data from target-train partition for the classifier
scores = {meth: [] for meth in ['dummy', 'rct', 'rpa', 'calibration']}

# Create a dataset with two domains, each with two classes both datasets
# are generated by the same generative procedure with the SPD Gaussian
# and one of them is transformed by a matrix A, i.e. X <- A @ X @ A.T
X_enc, y_enc = make_classification_transfer(n_matrices=100,
                                            class_sep=0.75,
                                            class_disp=1.0,
                                            domain_sep=5.0,
                                            theta=3*np.pi/5,
                                            random_state=seed)

# Object for splitting the datasets into training and validation partitions
# the training set is composed of all data points from the source domain
# plus a partition of the target domain whose size we can control
target_domain = 'target_domain'
n_splits = 5  # how many times to split the target domain into train/test
cv = TLSplitter(
    target_domain=target_domain,
    cv_iterator=StratifiedShuffleSplit(n_splits=n_splits, random_state=seed))

# Vary the proportion of the target domain for training
target_train_frac_array = np.linspace(0.01, 0.20, 10)
for target_train_frac in tqdm(target_train_frac_array):

    # Change fraction of the target training partition
    cv.cv_iterator.train_size = target_train_frac

    # Create dict for storing results of this particular CV split
    scores_cv = {meth: [] for meth in scores.keys()}

    # Carry out the cross-validation
    for train_idx, test_idx in cv.split(X_enc, y_enc):

        # Split the dataset into training and testing
        X_enc_train, X_enc_test = X_enc[train_idx], X_enc[test_idx]
        y_enc_train, y_enc_test = y_enc[train_idx], y_enc[test_idx]

        # (1) Dummy pipeline: no transfer learning at all
        # - The classifier is trained only with samples from the source dataset

        # Instantiate
        step1 = TLDummy()
        clf = TLEstimator(
            target_domain=target_domain,
            estimator=MDM(),
            domain_weight={'source_domain': 1.0, 'target_domain': 0.0})
        pipeline = make_pipeline(step1, clf)

        # Fit
        pipeline.fit(X_enc_train, y_enc_train)

        # Get the accuracy score
        scores_cv['dummy'].append(
            pipeline.score(X_enc_test, y_enc_test))

        # (2) RCT pipeline: recenter the data from each domain to identity [1]_
        # - The classifier is trained only with points from the source domain

        # Instantiate
        step1 = TLCenter(target_domain=target_domain)
        clf = TLEstimator(
            target_domain=target_domain,
            estimator=MDM(),
            domain_weight={'source_domain': 1.0, 'target_domain': 0.0})
        pipeline = make_pipeline(step1, clf)

        # Fit
        pipeline.fit(X_enc_train, y_enc_train)

        # Get the accuracy score
        scores_cv['rct'].append(pipeline.score(X_enc_test, y_enc_test))

        # (3) RPA pipeline: recenter, stretch, and rotate [2]_
        # - The classifier is trained with points from source and target

        # Instantiate
        step1 = TLCenter(target_domain=target_domain)
        step2 = TLStretch(
            target_domain=target_domain,
            final_dispersion=1,
            centered_data=True)
        step3 = TLRotate(
            target_domain=target_domain,
            metric='riemann')
        clf = TLEstimator(
            target_domain=target_domain,
            estimator=MDM(),
            domain_weight={'source_domain': 0.5, 'target_domain': 0.5})
        pipeline = make_pipeline(step1, step2, step3, clf)

        # Fit the pipeline
        pipeline.fit(X_enc_train, y_enc_train)

        # Get the accuracy score
        scores_cv['rpa'].append(pipeline.score(X_enc_test, y_enc_test))

        # (4) TLMDM pipeline -- ??
        # clf = TLMDM(transfer_coef=0.3, target_domain=target_domain)
        # pipeline = make_pipeline(clf)
        # pipeline.fit(X_enc_train, y_enc_train)
        # scores_cv['tlmdm'].append(pipeline.score(X_enc_test, y_enc_test))

        # (5) Calibration: use only data from target-train partition
        # - The classifier is trained only with points from the target domain

        # Instantiate
        clf = TLEstimator(
            target_domain=target_domain,
            estimator=MDM(),
            domain_weight={'source_domain': 0.0, 'target_domain': 1.0})
        pipeline = make_pipeline(clf)

        # Fit
        pipeline.fit(X_enc_train, y_enc_train)

        # Get the score
        scores_cv['calibration'].append(
            pipeline.score(X_enc_test, y_enc_test))

    # Get the average score of each pipeline
    for meth in scores.keys():
        scores[meth].append(np.mean(scores_cv[meth]))

# Store the results for each method on this particular seed
for meth in scores.keys():
    scores[meth] = np.array(scores[meth])

# Plot the results
fig, ax = plt.subplots(figsize=(6.7, 5.7))
for meth in scores.keys():
    ax.plot(
        target_train_frac_array,
        scores[meth],
        label=meth,
        lw=3.0)
ax.legend(loc='upper right')
ax.set_ylim(0.45, 0.75)
ax.set_yticks([0.5, 0.6, 0.7])
ax.set_xlim(0.00, 0.21)
ax.set_xticks([0.01, 0.05, 0.10, 0.15, 0.20])
ax.set_xticklabels([1, 5, 10, 15, 20])
ax.set_xlabel('percentage of training partition in target domain')
ax.set_ylabel('classification accuracy')
ax.set_title('comparison of transfer learning pipelines')

plt.show()


###############################################################################
# References
# ----------
# .. [1] `Transfer Learning: A Riemannian Geometry Framework With Applications
#    to Brain–Computer Interfaces
#    <https://hal.archives-ouvertes.fr/hal-01923278/>`_.
#    P Zanini et al, IEEE Transactions on Biomedical Engineering, vol. 65,
#     no. 5, pp. 1107-1116, August, 2017
# .. [2] `Riemannian Procrustes analysis: transfer learning for
#    brain-computer interfaces
#    <https://hal.archives-ouvertes.fr/hal-01971856>`_
#    PLC Rodrigues et al, IEEE Transactions on Biomedical Engineering, vol. 66,
#    no. 8, pp. 2390-2401, December, 2018
