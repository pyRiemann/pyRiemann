import numpy as np
import matplotlib.pyplot as plt

from pyriemann.datasets import make_gaussian_blobs
from pyriemann.embedding import SpectralEmbedding
from pyriemann.classification import MDM
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm, sqrtm, powm

from sklearn.pipeline import make_pipeline

from pyriemann.transferlearning_yenc import (
    encode_domains,
    decode_domains,
    TLSplitter,
    TLDummy,
    TLCenter,
    TLClassifier
)


def make_example_dataset(N, theta, alpha, eps):
    '''
        theta : angle for the rotation matrix applied to dataset 2
        alpha : proxy of how far the mean of dataset 2 should be from dataset 1
    '''

    # create a large set of matrices distributed in the same way
    X, y = make_gaussian_blobs(n_matrices=2*N, class_sep=eps)
    X1, y1 = X[:2*N], y[:2*N]
    X2, y2 = X[2*N:], y[2*N:]

    # we will now do some manipulations over X2 to create a dataset shift
    M2 = mean_riemann(X2)
    invsqrtM2 = invsqrtm(M2)
    # re-center dataset to identity
    X2 = invsqrtM2 @ X2 @ invsqrtM2.T
    # rotate dataset with Q
    Q = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X2 = Q @ X2 @ Q.T
    # parallel transport the dataset to another place
    M1 = mean_riemann(X1)
    R = powm(M1, alpha=alpha)
    sqrtR = sqrtm(R)
    X2 = sqrtR @ X2 @ sqrtR.T

    # create array specifying the domain for each epoch
    domain = np.array(len(X1)*['source_domain'] + len(X2)*['target_domain'])

    # encode the labels and domains together
    X = np.concatenate([X1, X2])
    y = np.concatenate([y1, y2])
    X_enc, y_enc = encode_domains(X, y, domain)

    return X_enc, y_enc


def make_figure(X_enc, y_enc):
    '''Plot the spectral embedding of the both datasets together'''

    X, y, domain = decode_domains(X_enc, y_enc)

    emb = SpectralEmbedding(n_components=2, metric='riemann')
    S = emb.fit_transform(X)
    sel_source = (domain == 'source_domain')
    S1 = S[sel_source]
    sel_target = (domain == 'target_domain')
    S2 = S[sel_target]

    fig, ax = plt.subplots(figsize=(7.3, 6.6))
    y1 = y[sel_source]
    for i in range(len(S1)):
        if y1[i] == 0:
            ax.scatter(S1[i, 0], S1[i, 1], c='C0', s=40, marker='x')
        if y1[i] == 1:
            ax.scatter(S1[i, 0], S1[i, 1], c='C1', s=40, marker='x')
    y2 = y[sel_target]
    for i in range(len(S2)):
        if y2[i] == 0:
            ax.scatter(S2[i, 0], S2[i, 1], c='C0', s=40, alpha=0.5, marker='o')
        if y2[i] == 1:
            ax.scatter(S2[i, 0], S2[i, 1], c='C1', s=40, alpha=0.5, marker='o')

    ax.set_xlim([-1.05, +1.05])
    ax.set_ylim([-1.05, +1.05])

    return fig


np.random.seed(13)

# create a 2D dataset with two domains, each with two classes
# both datasets are generated by the same generative procedure (gaussian blobs)
# and then one of them is transformed by a matrix A, i.e. X2 <- A @ X2 @ A.T
# with A = QP, where Q is an orthogonal matrix and P a positive definite matrix
# parameter theta defines the angle of rotation for Q and alpha is a proxy
# for how far from the Identity the mean of the new dataset should be
X_enc, y_enc = make_example_dataset(N=50, theta=np.pi/4, alpha=5, eps=1.5)

# plot a figure with the joint spectral embedding of the simulated data points
fig = make_figure(X_enc, y_enc)
fig.show()

# use the MDM classifier
clf = MDM()

# new object for splitting the datasets into training-validation
# the training set is composed of all data points from the source domain
# plus a partition of the target domain whose size we can control
target_domain = 'target_domain'
target_train_frac = 0.25  # proportion of the target domain for training
n_splits = 5  # how many times to split the target domain into train/test
cv = TLSplitter(
    n_splits=n_splits,
    target_train_frac=target_train_frac,
    target_domain=target_domain)

# we consider two types of pipeline for transfer learning
# DCT : no transformation of dataset between the domains
# RCT : re-center the data points from each domain to the Identity
scores = {}
meth_list = ['Dummy', 'RCT']
for meth in meth_list:
    scores[meth] = []

# The are three modes for training the pipeline:
#   (1) train clf only on source domain + training partition from target
#   (2) train clf only on source domain
#   (3) train clf only on training partition from target
# these different choices yield very different results in the classification.
training_mode = 1

# carry out the cross-validation
for train_idx, test_idx in cv.split(X_enc, y_enc):

    # split the dataset into training and testing
    X_enc_train, X_enc_test = X_enc[train_idx], X_enc[test_idx]
    y_enc_train, y_enc_test = y_enc[train_idx], y_enc[test_idx]

    # Dummy pipeline -- no transfer learning at all between source and target
    dummy_preprocess = TLDummy()
    dummy_clf = TLClassifier(
        target_domain=target_domain,
        clf=MDM(),
        training_mode=training_mode)
    dummy_pipeline = make_pipeline(dummy_preprocess, dummy_clf)
    dummy_pipeline.fit(X_enc_train, y_enc_train)
    scores['Dummy'].append(dummy_pipeline.score(X_enc_test, y_enc_test))

    # RCT pipeline -- recenter the data from each domain to identity
    rct_preprocess = TLCenter(target_domain=target_domain)
    rct_clf = TLClassifier(
        target_domain=target_domain,
        clf=MDM(),
        training_mode=training_mode)
    rct_pipeline = make_pipeline(rct_preprocess, rct_clf)
    rct_pipeline.fit(X_enc_train, y_enc_train)
    scores['RCT'].append(rct_pipeline.score(X_enc_test, y_enc_test))

# get the average score of each pipeline
for meth in meth_list:
    scores[meth] = np.mean(scores[meth])
print(scores)
