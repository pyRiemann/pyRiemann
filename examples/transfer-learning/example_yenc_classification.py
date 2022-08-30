import numpy as np
import matplotlib.pyplot as plt

from pyriemann.datasets import sample_gaussian_spd
from pyriemann.classification import MDM
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm, sqrtm, powm, expm

from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from tqdm import tqdm

from pyriemann.transferlearning_yenc import (
    _encode_domains,
    _decode_domains,
    TLSplitter,
    TLDummy,
    TLCenter,
    TLStretch,
    TLRotate,
    TLClassifier,
    TLMDM
)


def make_example_transfer_learning(N, class_sep=3.0, class_disp=1.0,
                                   domain_sep=5.0, theta=0.0,
                                   random_state=None):
    ''' Generate source and target toy datasets for transfer learning examples

    N : how many matrices to sample on each class for each domain
    class_sep : how separable the classes from each domain should be
    class_disp : intra class dispersion
    domain_sep : distance between global means for each domain
    theta : angle for rotation
    random_state : int, RandomState instance or None, default=None
    Pass an int for reproducible output across multiple function calls.
    '''

    rs = check_random_state(random_state)
    seeds = rs.randint(100, size=4)  # one seed for each sample_gaussian

    # the examples considered here are always for 2x2 matrices
    n_dim = 2

    # create a source dataset with two classes and global mean at identity
    M1_source = np.eye(n_dim)  # first class mean at Identity at first
    X1_source = sample_gaussian_spd(n_matrices=N,
                                    mean=M1_source,
                                    sigma=class_disp,
                                    random_state=seeds[0])
    y1_source = (1*np.ones(N)).astype(int)
    Pv = rs.randn(n_dim, n_dim)  # create random tangent vector
    Pv = (Pv + Pv.T)/2   # symmetrize
    Pv = Pv / np.linalg.norm(Pv)  # normalize
    P = expm(Pv)  # take it back to the SPD manifold
    M2_source = powm(P, alpha=class_sep)  # control distance to identity
    X2_source = sample_gaussian_spd(n_matrices=N,
                                    mean=M2_source,
                                    sigma=class_disp,
                                    random_state=seeds[1])
    y2_source = (2*np.ones(N)).astype(int)
    X_source = np.concatenate([X1_source, X2_source])
    M_source = mean_riemann(X_source)
    M_source_invsqrt = invsqrtm(M_source)
    for i in range(len(X_source)):  # center the dataset to Identity
        X_source[i] = M_source_invsqrt @ X_source[i] @ M_source_invsqrt
    y_source = np.concatenate([y1_source, y2_source])

    # create target dataset based on the source dataset
    X1_target = sample_gaussian_spd(
        n_matrices=N,
        mean=M1_source,
        sigma=class_disp,
        random_state=seeds[2])
    X2_target = sample_gaussian_spd(
        n_matrices=N,
        mean=M2_source,
        sigma=class_disp,
        random_state=seeds[3])
    X_target = np.concatenate([X1_target, X2_target])
    M_target = mean_riemann(X_target)
    M_target_invsqrt = invsqrtm(M_target)
    for i in range(len(X_target)):  # center the dataset to Identity
        X_target[i] = M_target_invsqrt @ X_target[i] @ M_target_invsqrt
    y_target = np.copy(y_source)

    # move the points in X_target with a random matrix A = P * Q

    # create SPD matrix for the translation between domains
    Pv = rs.randn(n_dim, n_dim)  # create random tangent vector
    Pv = (Pv + Pv.T)/2  # symmetrize
    Pv = Pv / np.linalg.norm(Pv)  # normalize
    P = expm(Pv)  # take it to the manifold
    P = powm(P, alpha=domain_sep)  # control distance to identity
    P = sqrtm(P)  # transport matrix

    # create orthogonal matrix for the rotation part
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    # transform the data points from the target domain
    A = P @ Q
    for i in range(len(X_target)):
        X_target[i] = A @ X_target[i] @ A.T

    # create array specifying the domain for each epoch
    domains = np.array(
        len(X_source)*['source_domain'] + len(X_target)*['target_domain']
    )

    # encode the labels and domains together
    X = np.concatenate([X_source, X_target])
    y = np.concatenate([y_source, y_target])
    X_enc, y_enc = _encode_domains(X, y, domains)

    return X_enc, y_enc


# we consider several types of pipeline for transfer learning
# dct : no transformation of dataset between the domains
# rct : re-center the data points from each domain to the Identity
# rpa : re-center, stretch and rotate
# tlmdm : ???
# calibration : use only data from target-train partition for the classifier
scores = {}
meth_list = ['dummy', 'rct', 'rpa', 'calibration']
for meth in meth_list:
    scores[meth] = []

# consider ten different seeds
for seed in range(10):

    # create a dataset with two domains, each with two classes both datasets
    # are generated by the same generative procedure with the SPD Gaussian
    # and one of them is transformed by a matrix A, i.e. X <- A @ X @ A.T
    X_enc, y_enc = make_example_transfer_learning(N=100,
                                                  class_sep=0.75,
                                                  class_disp=1.0,
                                                  domain_sep=5.0,
                                                  theta=np.pi/4,
                                                  random_state=seed)

    # object for splitting the datasets into training and validation partitions
    # the training set is composed of all data points from the source domain
    # plus a partition of the target domain whose size we can control
    target_domain = 'target_domain'
    n_splits = 5  # how many times to split the target domain into train/test
    cv = TLSplitter(n_splits=n_splits,
                    target_domain=target_domain,
                    random_state=seed)

    # create dict for storing results of this particular seed
    scores_seed = {}
    for meth in meth_list:
        scores_seed[meth] = []

    # vary the proportion of the target domain for training
    target_train_frac_array = np.linspace(0.01, 0.20, 10)
    for target_train_frac in tqdm(target_train_frac_array):

        # change fraction of the target training partition
        cv.target_train_frac = target_train_frac

        # create dict for storing results of this particular CV split
        scores_cv = {}
        for meth in meth_list:
            scores_cv[meth] = []

        # carry out the cross-validation
        for train_idx, test_idx in cv.split(X_enc, y_enc):

            # split the dataset into training and testing
            X_enc_train, X_enc_test = X_enc[train_idx], X_enc[test_idx]
            y_enc_train, y_enc_test = y_enc[train_idx], y_enc[test_idx]

            # (1) dummy pipeline: no transfer learning at all

            # instantiate the pipeline
            step1 = TLDummy()
            clf = TLClassifier(
                target_domain=target_domain,
                clf=MDM())
            pipeline = make_pipeline(step1, clf)

            # the classifier is trained only with points from the source domain
            X_train_dummy, y_train_dummy, domains = _decode_domains(
                X_enc_train,
                y_enc_train)
            y_train_dummy[domains == target_domain] = -1
            X_enc_train_dummy, y_enc_train_dummy = _encode_domains(
                X_train_dummy,
                y_train_dummy,
                domains)

            # fit pipeline
            pipeline.fit(X_enc_train_dummy, y_enc_train_dummy)

            # get the accuracy score
            scores_cv['dummy'].append(
                pipeline.score(X_enc_test, y_enc_test))

            # (2) rct pipeline: recenter the data from each domain to identity

            # instantiate the pipeline
            step1 = TLCenter(target_domain=target_domain)
            clf = TLClassifier(
                target_domain=target_domain,
                clf=MDM())
            pipeline = make_pipeline(step1, clf)

            # the classifier is trained only with points from the source domain
            X_train_rct, y_train_rct, domains = _decode_domains(
                X_enc_train,
                y_enc_train)
            y_train_rct[domains == target_domain] = -1
            X_enc_train_rct, y_enc_train_rct = _encode_domains(
                X_train_rct,
                y_train_rct,
                domains)

            # fit pipeline
            pipeline.fit(X_enc_train_rct, y_enc_train_rct)

            # get the accuracy score
            scores_cv['rct'].append(
                pipeline.score(X_enc_test, y_enc_test))

            # (3) rpa pipeline: recenter, stretch, and rotate

            # instantiate the pipeline
            step1 = TLCenter(target_domain=target_domain)
            step2 = TLStretch(
                target_domain=target_domain,
                final_dispersion=1,
                centered_data=True)
            step3 = TLRotate(
                target_domain=target_domain,
                distance_to_minimize='euc')
            clf = TLClassifier(
                target_domain=target_domain,
                clf=MDM())
            pipeline = make_pipeline(step1, step2, step3, clf)

            # the classifier is trained with points from source and target
            X_enc_train_rpa, y_enc_train_rpa = X_enc_train, y_enc_train

            # fit the pipeline
            pipeline.fit(X_enc_train_rpa, y_enc_train_rpa)

            # get the accuracy score
            scores_cv['rpa'].append(
                pipeline.score(X_enc_test, y_enc_test))

            # (4) tlmdm pipeline -- ??
            # clf = TLMDM(transfer_coef=0.3, target_domain=target_domain)
            # pipeline = make_pipeline(clf)
            # pipeline.fit(X_enc_train, y_enc_train)
            # scores_cv['tlmdm'].append(pipeline.score(X_enc_test, y_enc_test))

            # (5) calibration: use only data from target-train partition

            # instantiate the pipeline
            clf = TLClassifier(
                target_domain=target_domain,
                clf=MDM())
            pipeline = make_pipeline(clf)

            # the classifier is trained only with points from the target domain
            X_train_clb, y_train_clb, domains = _decode_domains(
                X_enc_train,
                y_enc_train)
            y_train_clb[domains != target_domain] = -1
            X_enc_train_clb, y_enc_train_clb = _encode_domains(
                X_train_clb,
                y_train_clb,
                domains)

            # fit pipeline
            pipeline.fit(X_enc_train_clb, y_enc_train_clb)

            # get the score
            scores_cv['calibration'].append(
                pipeline.score(X_enc_test, y_enc_test))

        # get the average score of each pipeline
        for meth in meth_list:
            scores_seed[meth].append(np.mean(scores_cv[meth]))

    # store the results for each method on this particular seed
    for meth in meth_list:
        scores[meth].append(scores_seed[meth])

# gather the results in a 2D array
for meth in meth_list:
    scores[meth] = np.array(scores[meth])

# plot the results
fig, ax = plt.subplots(figsize=(6.7, 5.7))
for meth in meth_list:
    ax.plot(
        target_train_frac_array,
        np.median(scores[meth], axis=0),
        label=meth,
        lw=3.0)
ax.legend(loc='upper left')
ax.set_ylim(0.45, 0.75)
ax.set_yticks([0.5, 0.6, 0.7])
ax.set_xlim(0.00, 0.21)
ax.set_xticks([0.01, 0.05, 0.10, 0.15, 0.20])
ax.set_xticklabels([1, 5, 10, 15, 20])
ax.set_xlabel('percentage of training partition in target domain')
ax.set_ylabel('classification accuracy')
ax.set_title('comparison of transfer learning pipelines')
fig.show()
