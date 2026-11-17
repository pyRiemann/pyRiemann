"""
====================================================================
Data transformations in the Riemannian Procrustes Analysis
====================================================================

Use the SpectralEmbedding module to plot in 2D the transformations on the
matrices from source and target domains when applying the Riemannian Procrustes
Analysis [1]_ to match their statistics.
It also shows how recentering is biased when the target domain is
class-imbalanced, and how sample weights correct this bias.
"""

import matplotlib.pyplot as plt
import numpy as np

from pyriemann.classification import MDM
from pyriemann.datasets.simulated import make_classification
from pyriemann.embedding import SpectralEmbedding
from pyriemann.transfer import (
    decode_domains,
    encode_domains,
    TLCenter,
    TLRotate,
)


###############################################################################

# Fix seed for reproducible results
seed = 66

# create source and target datasets
n_matrices = 50
X_enc, y_enc = make_classification(
    n_matrices=n_matrices,
    class_sep=2.0,
    class_disp=0.25,
    domain_sep=2.0,
    theta=np.pi/4,
    random_state=seed
)

# generate dataset
X_org, y, domain = decode_domains(X_enc, y_enc)

# instantiate object for doing spectral embeddings
emb = SpectralEmbedding(n_components=2, metric="riemann")

# create dict to store the embedding after each step of RPA
embedded_points = {}

# embed the original source and target datasets
points = np.concatenate([X_org, np.eye(2)[None, :, :]])  # stack the identity
embedded_points["origin"] = emb.fit_transform(points)

# embed the source and target datasets after recentering
rct = TLCenter(target_domain="target_domain")
X_rct = rct.fit_transform(X_org, y_enc)
points = np.concatenate([X_rct, np.eye(2)[None, :, :]])  # stack the identity
embedded_points["rct"] = emb.fit_transform(points)

# embed the source and target datasets after recentering
rot = TLRotate(target_domain="target_domain", metric="riemann")
X_rot = rot.fit_transform(X_rct, y_enc)

points = np.concatenate([X_org, X_rct, X_rot, np.eye(2)[None, :, :]])
S = emb.fit_transform(points)
S = S - S[-1]
embedded_points["origin"] = S[:4*n_matrices]
embedded_points["rct"] = S[4*n_matrices:8*n_matrices]
embedded_points["rot"] = S[8*n_matrices:-1]


###############################################################################
# Plot the results, reproducing the Figure 1 of [1]_.

fig, ax = plt.subplots(figsize=(13.5, 4.4), ncols=3, sharey=True)
plt.subplots_adjust(wspace=0.10)
steps = ["origin", "rct", "rot"]
titles = ["original", "after recentering", "after rotation"]
for axi, step, title in zip(ax, steps, titles):
    S_step = embedded_points[step]
    S_source = S_step[domain == "source_domain"]
    y_source = y[domain == "source_domain"]
    S_target = S_step[domain == "target_domain"]
    y_target = y[domain == "target_domain"]
    axi.scatter(
        S_source[y_source == "1"][:, 0],
        S_source[y_source == "1"][:, 1],
        c="C0", s=50, alpha=0.50,
    )
    axi.scatter(
        S_source[y_source == "2"][:, 0],
        S_source[y_source == "2"][:, 1],
        c="C1", s=50, alpha=0.50,
    )
    axi.scatter(
        S_target[y_target == "1"][:, 0],
        S_target[y_target == "1"][:, 1],
        c="C0", s=50, alpha=0.30, marker="^",
    )
    axi.scatter(
        S_target[y_target == "2"][:, 0],
        S_target[y_target == "2"][:, 1],
        c="C1", s=50, alpha=0.30, marker="^",
    )
    axi.scatter(S[-1, 0], S[-1, 1], c="k", s=80, marker="*")
    axi.set_xlim(-0.60, +1.60)
    axi.set_ylim(-1.10, +1.25)
    axi.set_xticks([-0.5, 0.0, 0.5, 1.0, 1.5])
    axi.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axi.set_title(title, fontsize=14)
ax[0].scatter([], [], c="C0", label="source - class 0")
ax[0].scatter([], [], c="C1", label="source - class 1")
ax[0].scatter([], [], marker="^", c="C0", label="target - class 0")
ax[0].scatter([], [], marker="^", c="C1", label="target - class 1")
ax[0].legend(loc="upper right")
plt.show()


###############################################################################
# Recentering with a class-imbalanced target domain
# -------------------------------------------------
#
# Recentering estimates the center of each domain as the mean of its matrices,
# giving the same weight to all matrices. It implicitly assumes that classes
# have similar proportions in source and target domains.
# When the target domain is class-imbalanced, its center is pulled towards the
# majority class, and recentering no longer matches the two domains.
#
# Here, the target domain keeps all matrices of the first class (blue), but
# only five matrices of the second class (orange). Class-balanced weights,
# inversely proportional to class sizes and given to parameter
# ``sample_weight``, correct this bias.

# keep the source domain, and an imbalanced subset of the target domain
rs = np.random.RandomState(seed)
is_target = domain == "target_domain"
idx = np.concatenate([
    np.where(~is_target)[0],
    np.where(is_target & (y == "1"))[0],
    rs.choice(np.where(is_target & (y == "2"))[0], 5, replace=False),
])
X_imb, y_imb, domain_imb = X_org[idx], y[idx], domain[idx]
_, y_enc_imb = encode_domains(X_imb, y_imb, domain_imb)
is_target_imb = domain_imb == "target_domain"

# class-balanced weights: each class of target domain has the same total weight
weights = np.ones(len(X_imb))
for label in np.unique(y_imb[is_target_imb]):
    mask = is_target_imb & (y_imb == label)
    weights[mask] = 1 / np.sum(mask)

# recenter with and without weights, and evaluate a classifier trained on the
# source domain on the whole (balanced) target domain
mdm = MDM()
mdm.fit(X_org[~is_target], y[~is_target])
print("Accuracy on the whole target domain")
score = mdm.score(X_org[is_target], y[is_target])
print(f"  without recentering: {score:.2f}")
X_rct_imb = {}
for name, sample_weight in [("unweighted", None), ("weighted", weights)]:
    rct = TLCenter(target_domain="target_domain")
    X_rct_imb[name] = rct.fit_transform(
        X_imb, y_enc_imb, sample_weight=sample_weight
    )
    mdm.fit(X_rct_imb[name][~is_target_imb], y_imb[~is_target_imb])
    score = mdm.score(rct.transform(X_org[is_target]), y[is_target])
    print(f"  after {name} recentering: {score:.2f}")


###############################################################################
# Plot the imbalanced target domain before and after recentering.

n_imb = len(X_imb)
points = np.concatenate([
    X_imb,
    X_rct_imb["unweighted"],
    X_rct_imb["weighted"],
    np.eye(2)[None, :, :],
])
S = emb.fit_transform(points)
S = S - S[-1]

fig, ax = plt.subplots(figsize=(13.5, 4.4), ncols=3, sharey=True)
plt.subplots_adjust(wspace=0.10)
titles = ["original", "after recentering", "after weighted recentering"]
for i, (axi, title) in enumerate(zip(ax, titles)):
    S_step = S[i * n_imb:(i + 1) * n_imb]
    for dom, marker, alpha in [
        ("source_domain", "o", 0.50),
        ("target_domain", "^", 0.30),
    ]:
        for label, color in [("1", "C0"), ("2", "C1")]:
            mask = (domain_imb == dom) & (y_imb == label)
            axi.scatter(
                S_step[mask, 0],
                S_step[mask, 1],
                c=color, s=50, alpha=alpha, marker=marker,
            )
    axi.scatter(0, 0, c="k", s=80, marker="*")
    axi.set_title(title, fontsize=14)
ax[0].scatter([], [], c="C0", label="source - class 0")
ax[0].scatter([], [], c="C1", label="source - class 1")
ax[0].scatter([], [], marker="^", c="C0", label="target - class 0")
ax[0].scatter([], [], marker="^", c="C1", label="target - class 1")
ax[0].legend(loc="lower right")
plt.show()

###############################################################################
# Unweighted recentering moves the majority class of the target domain onto
# the identity, between the two classes of the source domain, which decreases
# accuracy. Weighted recentering brings each class of the target domain closer
# to the same class of the source domain.
#
# Here, weights are computed from target labels, which are available in RPA
# since :class:`pyriemann.transfer.TLRotate` needs them.
# In online use, the target domain is often a buffer of recent unlabeled
# matrices, whose class proportions are not controlled.
# Its labels can then be estimated by a classifier trained on the source
# domain, applied to target matrices recentered with the source center, ie
# before target recentering. However, these estimated labels are not reliable
# when the shift between domains is large.


###############################################################################
# References
# ----------
# .. [1] `Riemannian Procrustes analysis: transfer learning for
#    brain-computer interfaces
#    <https://hal.archives-ouvertes.fr/hal-01971856>`_
#    PLC Rodrigues et al, IEEE Transactions on Biomedical Engineering, vol. 66,
#    no. 8, pp. 2390-2401, December, 2018
