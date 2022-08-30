import numpy as np
import matplotlib.pyplot as plt

from pyriemann.datasets import sample_gaussian_spd
from pyriemann.embedding import SpectralEmbedding
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm, sqrtm, powm, expm

from sklearn.utils import check_random_state

from pyriemann.transferlearning_yenc import (
    _encode_domains,
    _decode_domains,
    TLCenter,
    TLRotate,
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
    seeds = rs.randint(100, size=4)

    # the examples considered here are always for 2x2 matrices
    n_dim = 2

    # create a source dataset with two classes and global mean at identity
    M1_source = np.eye(n_dim)  # first class mean at Identity at first
    X1_source = sample_gaussian_spd(
        n_matrices=N,
        mean=M1_source,
        sigma=class_disp,
        random_state=seeds[0])
    y1_source = (1*np.ones(N)).astype(int)
    Pv = rs.randn(n_dim, n_dim)  # create random tangent vector
    Pv = (Pv + Pv.T)/2  # symmetrize
    Pv = Pv / np.linalg.norm(Pv)  # normalize
    P = expm(Pv)  # take it back to the SPD manifold
    M2_source = powm(P, alpha=class_sep)  # control distance to identity
    X2_source = sample_gaussian_spd(
        n_matrices=N,
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


# fix seed for reproducible results
seed = 66

# create source and target datasets
N = 50
X_enc, y_enc = make_example_transfer_learning(
    N=N,
    class_sep=2.0,
    class_disp=0.25,
    domain_sep=2.0,
    theta=np.pi/4,
    random_state=seed
)

# generate dataset
X_org, y, domain = _decode_domains(X_enc, y_enc)

# instantiate object for doing spectral embeddings
emb = SpectralEmbedding(n_components=2, metric='riemann')

# create dict to store the embedding after each step of RPA
embedded_points = {}

# embed the original source and target datasets
points = np.concatenate([X_org, np.eye(2)[None, :, :]])  # stack the identity
embedded_points['origin'] = emb.fit_transform(points)

# embed the source and target datasets after recentering
rct = TLCenter(target_domain='target_domain')
X_rct = rct.fit_transform(X_org, y_enc)
points = np.concatenate([X_rct, np.eye(2)[None, :, :]])  # stack the identity
embedded_points['rct'] = emb.fit_transform(points)

# embed the source and target datasets after recentering
rot = TLRotate(target_domain='target_domain')
X_rot = rot.fit_transform(X_rct, y_enc)

points = np.concatenate([X_org, X_rct, X_rot, np.eye(2)[None, :, :]])
S = emb.fit_transform(points)
S = S - S[-1]
embedded_points['origin'] = S[:4*N]
embedded_points['rct'] = S[4*N:8*N]
embedded_points['rot'] = S[8*N:-1]

# plot the results
fig, ax = plt.subplots(figsize=(13.5, 4.4), ncols=3, sharey=True)
plt.subplots_adjust(wspace=0.10)
steps = ['origin', 'rct', 'rot']
for axi, step in zip(ax, steps):
    S_step = embedded_points[step]
    S_source = S_step[domain == 'source_domain']
    y_source = y[domain == 'source_domain']
    S_target = S_step[domain == 'target_domain']
    y_target = y[domain == 'source_domain']
    axi.scatter(
        S_source[y_source == 1][:, 0],
        S_source[y_source == 1][:, 1],
        c='C0', s=50)
    axi.scatter(
        S_source[y_source == 2][:, 0],
        S_source[y_source == 2][:, 1],
        c='C1', s=50)
    axi.scatter(
        S_target[y_target == 1][:, 0],
        S_target[y_target == 1][:, 1],
        c='C0', s=50, alpha=0.50)
    axi.scatter(
        S_target[y_target == 2][:, 0],
        S_target[y_target == 2][:, 1],
        c='C1', s=50, alpha=0.50)
    axi.scatter(S[-1, 0], S[-1, 1], c='k', s=80, marker="*")
    axi.set_xlim(-0.60, +1.60)
    axi.set_ylim(-1.10, +1.25)
    axi.set_xticks([-0.5, 0.0, 0.5, 1.0, 1.5])
    axi.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
fig.show()
