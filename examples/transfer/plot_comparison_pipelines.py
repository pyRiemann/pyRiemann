"""
====================================================================
Comparison of pipelines for transfer learning
====================================================================

Compare the classificaton performance of four pipelines for transfer learning.
The data points are all simulated from a toy model based on the Riemannian
Gaussian distribution.
"""
import numpy as np
import matplotlib.pyplot as plt

from pyriemann.classification import MDM
from pyriemann.datasets.simulated import make_example_transfer_learning

from sklearn.pipeline import make_pipeline
from tqdm import tqdm

from pyriemann.transfer import (
    encode_domains,
    decode_domains,
    TLSplitter,
    TLDummy,
    TLCenter,
    TLStretch,
    TLRotate,
    TLClassifier,
    # TLMDM
)

import warnings
warnings.filterwarnings("ignore")

# choose seed for reproducible results
seed = 100

# we consider several types of pipeline for transfer learning
# dct : no transformation of dataset between the domains
# rct : re-center the data points from each domain to the Identity
# rpa : re-center, stretch and rotate (Riemannian Procrustes Analysis)
# tlmdm : ???
# calibration : use only data from target-train partition for the classifier
scores = {meth: [] for meth in ['dummy', 'rct', 'rpa', 'calibration']}

# create a dataset with two domains, each with two classes both datasets
# are generated by the same generative procedure with the SPD Gaussian
# and one of them is transformed by a matrix A, i.e. X <- A @ X @ A.T
X_enc, y_enc = make_example_transfer_learning(n_matrices=100,
                                              class_sep=0.75,
                                              class_disp=1.0,
                                              domain_sep=5.0,
                                              theta=3*np.pi/5,
                                              random_state=seed)

# object for splitting the datasets into training and validation partitions
# the training set is composed of all data points from the source domain
# plus a partition of the target domain whose size we can control
target_domain = 'target_domain'
n_splits = 5  # how many times to split the target domain into train/test
cv = TLSplitter(n_splits=n_splits,
                target_domain=target_domain,
                random_state=seed)

# vary the proportion of the target domain for training
target_train_frac_array = np.linspace(0.01, 0.20, 10)
for target_train_frac in tqdm(target_train_frac_array):

    # change fraction of the target training partition
    cv.target_train_frac = target_train_frac

    # create dict for storing results of this particular CV split
    scores_cv = {meth: [] for meth in scores.keys()}

    # carry out the cross-validation
    for train_idx, test_idx in cv.split(X_enc, y_enc):

        # split the dataset into training and testing
        X_enc_train, X_enc_test = X_enc[train_idx], X_enc[test_idx]
        y_enc_train, y_enc_test = y_enc[train_idx], y_enc[test_idx]

        # (1) Dummy pipeline: no transfer learning at all

        # instantiate the pipeline
        step1 = TLDummy()
        clf = TLClassifier(target_domain=target_domain, clf=MDM())
        pipeline = make_pipeline(step1, clf)

        # the classifier is trained only with points from the source domain
        X_train_dummy, y_train_dummy, domains = decode_domains(
            X_enc_train,
            y_enc_train)
        y_train_dummy[domains == target_domain] = -1
        X_enc_train_dummy, y_enc_train_dummy = encode_domains(
            X_train_dummy,
            y_train_dummy,
            domains)

        # fit pipeline
        pipeline.fit(X_enc_train_dummy, y_enc_train_dummy)

        # get the accuracy score
        scores_cv['dummy'].append(
            pipeline.score(X_enc_test, y_enc_test))

        # (2) RCT pipeline: recenter the data from each domain to identity
        # see Zanini et al. "Transfer Learning: A Riemannian Geometry Framework
        # With Applications to Brain–Computer Interfaces". IEEE Transactions on
        # Biomedical Engineering, vol. 65, no. 5, pp. 1107-1116, August, 2017

        # instantiate the pipeline
        step1 = TLCenter(target_domain=target_domain)
        clf = TLClassifier(
            target_domain=target_domain,
            clf=MDM())
        pipeline = make_pipeline(step1, clf)

        # the classifier is trained only with points from the source domain
        X_train_rct, y_train_rct, domains = decode_domains(
            X_enc_train,
            y_enc_train)
        y_train_rct[domains == target_domain] = -1
        X_enc_train_rct, y_enc_train_rct = encode_domains(
            X_train_rct,
            y_train_rct,
            domains)

        # fit pipeline
        pipeline.fit(X_enc_train_rct, y_enc_train_rct)

        # get the accuracy score
        scores_cv['rct'].append(pipeline.score(X_enc_test, y_enc_test))

        # (3) RPA pipeline: recenter, stretch, and rotate
        # see PLC Rodrigues et al “Riemannian Procrustes analysis: transfer
        # learning for brain-computer interfaces”. IEEE Transactions on
        # Biomedical Engineering, vol. 66, no. 8, pp. 2390-2401, December, 2018

        # instantiate the pipeline
        step1 = TLCenter(target_domain=target_domain)
        step2 = TLStretch(
            target_domain=target_domain,
            final_dispersion=1,
            centered_data=True)
        step3 = TLRotate(
            target_domain=target_domain,
            metric='riemann')
        clf = TLClassifier(
            target_domain=target_domain,
            clf=MDM())
        pipeline = make_pipeline(step1, step2, step3, clf)

        # the classifier is trained with points from source and target
        X_enc_train_rpa, y_enc_train_rpa = X_enc_train, y_enc_train

        # fit the pipeline
        pipeline.fit(X_enc_train_rpa, y_enc_train_rpa)

        # get the accuracy score
        scores_cv['rpa'].append(pipeline.score(X_enc_test, y_enc_test))

        # (4) TLMDM pipeline -- ??
        # clf = TLMDM(transfer_coef=0.3, target_domain=target_domain)
        # pipeline = make_pipeline(clf)
        # pipeline.fit(X_enc_train, y_enc_train)
        # scores_cv['tlmdm'].append(pipeline.score(X_enc_test, y_enc_test))

        # (5) calibration: use only data from target-train partition

        # instantiate the pipeline
        clf = TLClassifier(target_domain=target_domain, clf=MDM())
        pipeline = make_pipeline(clf)

        # the classifier is trained only with points from the target domain
        X_train_clb, y_train_clb, domains = decode_domains(
            X_enc_train,
            y_enc_train)
        y_train_clb[domains != target_domain] = -1
        X_enc_train_clb, y_enc_train_clb = encode_domains(
            X_train_clb,
            y_train_clb,
            domains)

        # fit pipeline
        pipeline.fit(X_enc_train_clb, y_enc_train_clb)

        # get the score
        scores_cv['calibration'].append(
            pipeline.score(X_enc_test, y_enc_test))

    # get the average score of each pipeline
    for meth in scores.keys():
        scores[meth].append(np.mean(scores_cv[meth]))

# store the results for each method on this particular seed
for meth in scores.keys():
    scores[meth] = np.array(scores[meth])

# plot the results
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
fig.show()
