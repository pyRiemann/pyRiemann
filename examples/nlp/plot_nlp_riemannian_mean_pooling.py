"""
===============================================================================
Riemannian Mean Pooling of Language Model Token Metrics
===============================================================================

Riemannian Mean Pooling (RMP) [1]_ turns a sentence into one symmetric
positive-definite (SPD) matrix. Each token embedding of a language model is
mapped by an encoder to a low-dimensional latent space, the Jacobian of the
encoder at that token defines a local metric tensor, and the sentence is
represented by the Riemannian mean of its token metrics.

This example classifies sentences of the Corpus of Linguistic Acceptability
(CoLA) [2]_ from precomputed RMP matrices, and compares a trained encoder with
a random encoder of the same architecture.
"""
# Authors: Szczepan Konior, Gregoire Cattan
#
# License: BSD (3-clause)

import os
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline

from pyriemann.geometry.distance import distance_riemann
from pyriemann.geometry.mean import mean_riemann
from pyriemann.preprocessing import Whitening
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils._data import get_data_path


###############################################################################
# From token embeddings to sentence SPD matrices
# ----------------------------------------------
#
# The matrices were built from layer 9 of ``bert-base-uncased``. An encoder
# :math:`f : \mathbb{R}^{768} \to \mathbb{R}^{64}` maps each token embedding
# :math:`x_t` to a latent space. With :math:`J_t` the Jacobian of :math:`f` at
# :math:`x_t`, the token metric is the pullback
#
# .. math::
#     G_t = (J_t J_t^\top + \epsilon I)^{-1}
#
# regularised to be well conditioned. A sentence of :math:`T` tokens is
# represented by the Riemannian mean of its token metrics under the
# affine-invariant metric
#
# .. math::
#     C = \arg\min_{G} \sum_{t=1}^{T} \delta_R^2(G, G_t)
#
# which is what :func:`pyriemann.geometry.mean.mean_riemann` computes.
#
# Two encoders share one architecture. The trained encoder is the encoder of
# an Intrinsic Green's Learning (IGL) model [3]_, trained without labels to
# reconstruct the tokens. The random encoder keeps its random initialisation.
#
# Running BERT and the encoders requires PyTorch, so this example downloads
# the resulting matrices: 400 training and 200 validation sentences, balanced
# by class, stored as upper triangles.

REVISION = "408e37b8313103ebf96fa8d7727f75d0addd8932"
URL = (
    "https://huggingface.co/datasets/szczepank112/rmp-cola-demo/resolve/"
    f"{REVISION}/rmp_cola.npz"
)

data_path = get_data_path("rmp_cola")
os.makedirs(data_path, exist_ok=True)
filename = os.path.join(data_path, "rmp_cola.npz")
if not os.path.exists(filename):
    urlretrieve(URL, filename)
data = np.load(filename)

n_channels = 64
iu = np.triu_indices(n_channels)


def unpack(triu):
    """Rebuild symmetric matrices from their upper triangles."""
    X = np.zeros((len(triu), n_channels, n_channels))
    X[:, iu[0], iu[1]] = triu
    return X + np.triu(X, 1).transpose(0, 2, 1)


encoders = ["trained", "random"]
X_train = {enc: unpack(data[f"{enc}_train"]) for enc in encoders}
X_val = {enc: unpack(data[f"{enc}_val"]) for enc in encoders}
y_train, y_val = data["y_train"], data["y_val"]

print(f"Training set: {X_train['trained'].shape}, {np.bincount(y_train)}")
print(f"Validation set: {X_val['trained'].shape}, {np.bincount(y_val)}")


###############################################################################
# Spectra of the sentence matrices
# --------------------------------
#
# The two encoders produce matrices at very different scales: the eigenvalues
# of the trained encoder's matrices are about two orders of magnitude larger.
# The spread of the matrices around their Riemannian mean, measured by the
# affine-invariant distance, is what a classifier on the manifold sees.

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
colors = {"trained": "C0", "random": "C1"}

for enc in encoders:
    eigvals = np.linalg.eigvalsh(X_train[enc])[:, ::-1]
    q25, q50, q75 = np.percentile(eigvals, [25, 50, 75], axis=0)
    rank = np.arange(1, n_channels + 1)
    axes[0].plot(rank, q50, color=colors[enc], label=enc)
    axes[0].fill_between(rank, q25, q75, color=colors[enc], alpha=0.3)

    dist = distance_riemann(X_train[enc], mean_riemann(X_train[enc]))
    axes[1].hist(dist, bins=30, color=colors[enc], alpha=0.6, label=enc)

axes[0].set(
    yscale="log", xlabel="Eigenvalue rank", ylabel="Eigenvalue",
    title="Eigenvalues, median and interquartile range",
)
axes[1].set(
    xlabel="Affine-invariant distance to the mean", ylabel="Sentences",
    title="Spread around the Riemannian mean",
)
for ax in axes:
    ax.legend()
plt.tight_layout()
plt.show()


###############################################################################
# Classification in the tangent space
# -----------------------------------
#
# :class:`pyriemann.preprocessing.Whitening` moves the Riemannian mean of the
# training matrices to the identity, which removes the difference in scale
# between the encoders. :class:`pyriemann.tangentspace.TangentSpace` then
# projects the matrices onto the tangent space at the identity, where a
# logistic regression is trained. Both steps are fitted on the training set
# only.

fig, ax = plt.subplots(figsize=(5, 5))

for enc in encoders:
    clf = make_pipeline(
        Whitening(metric="riemann"),
        TangentSpace(metric="riemann"),
        LogisticRegression(C=1.0, max_iter=2000),
    )
    clf.fit(X_train[enc], y_train)
    y_score = clf.predict_proba(X_val[enc])[:, 1]
    auc = roc_auc_score(y_val, y_score)
    print(f"{enc} encoder: ROC AUC = {auc:.3f}")
    fpr, tpr, _ = roc_curve(y_val, y_score)
    ax.plot(fpr, tpr, color=colors[enc], label=f"{enc} (AUC = {auc:.3f})")

ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
ax.set(
    xlabel="False positive rate", ylabel="True positive rate",
    title="Acceptability classification on CoLA",
)
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()


###############################################################################
# Discussion
# ----------
#
# The trained encoder gives a higher area under the ROC curve than the random
# encoder. Both arms share the sentences, the pipeline and the regularisation
# of the classifier, so the difference comes from the encoder that defines the
# token metrics.
#
# This comparison does not make RMP a strong sentence classifier. On the full
# datasets, a logistic regression on the mean-pooled BERT embeddings matches
# or beats RMP once the regularisation of both classifiers is tuned.


###############################################################################
# References
# ----------
# .. [1] `Riemannian Geometry for Pre-trained Language Model Embeddings
#    <https://openreview.net/forum?id=afqK8YeXQ7>`_
#    S. Konior, A. Quemy, P. Klocek, B. Sobieski, and G. Cattan,
#    EMNLP 2026 GroundLM Workshop, 2026.
# .. [2] `Neural Network Acceptability Judgments
#    <https://aclanthology.org/Q19-1040/>`_
#    A. Warstadt, A. Singh, and S. R. Bowman, Transactions of the Association
#    for Computational Linguistics, vol. 7, pp. 625-641, 2019.
# .. [3] `Intrinsic Green's Learning: Supervised Learning on Manifolds via
#    Inverse PDE
#    <https://openreview.net/forum?id=Y6RpdS98l8>`_
#    A. Quemy, ICLR Workshop on AI and PDE, 2026.
