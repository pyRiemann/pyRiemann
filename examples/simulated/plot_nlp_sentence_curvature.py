"""
===============================================================================
Riemannian Curvature of Sentence Trajectories
===============================================================================

Each sentence ("I love Alice", "I hate Bob", ...) is a trajectory of token
embeddings in a shared latent space. Motivated by the observation that curved
regions in LLM residual streams encode distinct semantic concerns [1]_ [2]_,
we ask: do love/hate sentences trace geometrically distinguishable trajectories?

Each token's local geometry is captured as a symmetric positive-definite (SPD)
matrix via neighbourhood tangent patches, then sentences are represented by the
Riemannian mean of their token SPD matrices, and finally classified with MDM.
"""
# Authors: Szczepan Konor, Gregoire Cattan
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import distance_riemann
from pyriemann.geometry.mean import mean_riemann


###############################################################################

D = 8

FEMALE_NAMES = ["Alice", "Clara", "Eva", "Fiona", "Grace", "Helen"]
MALE_NAMES = ["Bob", "David", "Frank", "Henry", "Ivan", "James"]
ALL_NAMES = FEMALE_NAMES + MALE_NAMES


def _love_token(rng):
    """'love' embedding: point on a unit 3-sphere → curved local geometry."""
    v = rng.standard_normal(3)
    v /= np.linalg.norm(v)
    emb = np.zeros(D)
    emb[1:4] = v
    emb += rng.standard_normal(D) * 0.04
    return emb


def _hate_token(rng):
    """'hate' embedding: point on a flat 2-D plane → zero local curvature."""
    emb = np.zeros(D)
    emb[4] = rng.standard_normal() * 0.6
    emb[5] = rng.standard_normal() * 0.6
    emb += rng.standard_normal(D) * 0.04
    return emb


def _name_token(name, rng):
    """Name embedding: female → +dim6, male → -dim6."""
    emb = np.zeros(D)
    emb[6] = 1.0 if name in FEMALE_NAMES else -1.0
    emb += rng.standard_normal(D) * 0.04
    return emb


class NeighborhoodPatchExtractor(BaseEstimator, TransformerMixin):
    """Build a patch of unit tangent directions to k nearest neighbours.

    Input  : [n_tokens_total, d]
    Output : [n_tokens_total, d, k]  — ready for pyriemann.Covariances
    """

    def __init__(self, k=6):
        self.k = k

    def fit(self, *_):
        return self

    def transform(self, X):
        n, d = X.shape
        knn = NearestNeighbors(n_neighbors=self.k, algorithm="auto").fit(X)
        nn_idx = knn.kneighbors(X, return_distance=False)
        patches = np.zeros((n, d, self.k))
        for i in range(n):
            for j, nb in enumerate(nn_idx[i]):
                v = X[nb] - X[i]
                norm = np.linalg.norm(v)
                if norm > 1e-10:
                    patches[i, :, j] = v / norm
        return patches


class SentenceAggregator(BaseEstimator, TransformerMixin):
    """Aggregate per-token SPD tensors into one SPD matrix per sentence.

    Input  : [n_sentences × n_tokens, d, d]
    Output : [n_sentences, d, d]
    """

    def __init__(self, n_tokens=3):
        self.n_tokens = n_tokens

    def fit(self, *_):
        return self

    def transform(self, metrics):
        n_sentences = metrics.shape[0] // self.n_tokens
        out = np.zeros((n_sentences, metrics.shape[1], metrics.shape[2]))
        for i in range(n_sentences):
            block = metrics[i * self.n_tokens: (i + 1) * self.n_tokens]
            out[i] = mean_riemann(block)
        return out


###############################################################################
# Generate synthetic sentence embeddings
# ---------------------------------------
#
# Each sentence "I love/hate [name]" is represented by three token embeddings.
# "love" tokens lie on a unit 3-sphere (positive curvature, K > 0), while
# "hate" tokens lie on a flat 2-D plane (zero curvature, K = 0) [1]_ [2]_.
# Name tokens are split into two gender clusters shared across both classes.

N_NAMES = 6    # sentences per class
N_TOKENS = 3   # [I, verb, name]
K = 6          # neighbours for local metric estimation

rng = np.random.default_rng(0)
i_base = np.zeros(D)
i_base[0] = 1.0

names = ALL_NAMES[:N_NAMES]
x_list, y, sentence_labels = [], [], []

for name in names:
    x_list.append(np.stack([
        i_base + rng.standard_normal(D) * 0.02,
        _love_token(rng),
        _name_token(name, rng),
    ]))
    y.append(0)
    sentence_labels.append(f"I love {name}")

for name in names:
    x_list.append(np.stack([
        i_base + rng.standard_normal(D) * 0.02,
        _hate_token(rng),
        _name_token(name, rng),
    ]))
    y.append(1)
    sentence_labels.append(f"I hate {name}")

y = np.array(y)
x_global = np.vstack(x_list)


###############################################################################
# Build and fit the pipeline
# --------------------------
#
# The pipeline standardises embeddings, extracts neighbourhood tangent patches,
# estimates a local metric tensor (SPD matrix) per token with Covariances,
# aggregates token tensors into one SPD matrix per sentence via the Riemannian
# mean (SentenceAggregator), and classifies with MDM [2]_.

pipeline = make_pipeline(
    StandardScaler(),
    NeighborhoodPatchExtractor(k=K),
    Covariances(estimator="lwf"),
    SentenceAggregator(n_tokens=N_TOKENS),
    MDM(metric="riemann"),
    memory=None,
)
pipeline.fit(x_global, y)
y_pred = pipeline.predict(x_global)

print(f"Accuracy: {(y_pred == y).mean():.0%}")

# Extract sentence-level SPD features for visualisation
sentence_spd = x_global
for _, step in pipeline.steps[:-1]:
    sentence_spd = step.transform(sentence_spd)


###############################################################################
# Visualise results
# -----------------
#
# Three panels show (1) token trajectories in PCA space, (2) the Riemannian
# mean metric tensor per class, and (3) the MDM decision space with the
# Riemannian distance to each class centroid.

palette = {"love": "#3498DB", "hate": "#E84C3D"}
mdm = pipeline[-1]

all_tokens = np.vstack(x_list)
all_2d = PCA(n_components=2, random_state=0).fit_transform(all_tokens)
dist_love = np.array([distance_riemann(m, mdm.covmeans_[0]) for m in sentence_spd])
dist_hate = np.array([distance_riemann(m, mdm.covmeans_[1]) for m in sentence_spd])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    'Sentence trajectories: "I love/hate [name]"\n'
    "local metric tensor (SPD) per token → Riemannian mean per sentence → MDM",
    fontsize=11, fontweight="bold",
)

ax = axes[0]
token_cls = np.repeat(y, N_TOKENS)
for cls, verb in enumerate(["love", "hate"]):
    mask = token_cls == cls
    ax.scatter(
        all_2d[mask, 0], all_2d[mask, 1],
        c=palette[verb], label=verb,
        s=60, edgecolors="k", linewidths=0.3, alpha=0.85,
    )
traj_2d = all_2d.reshape(-1, N_TOKENS, 2)
for traj, cls in zip(traj_2d, y):
    verb = ["love", "hate"][cls]
    ax.plot(traj[:, 0], traj[:, 1], "-", color=palette[verb], alpha=0.25, lw=1.2)
ax.set_title("Token embeddings (PCA)\neach line = one sentence trajectory")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.legend(title="class")

ax = axes[1]
mean_love = mean_riemann(sentence_spd[y == 0])
mean_hate = mean_riemann(sentence_spd[y == 1])
combined = np.block([
    [mean_love, np.full_like(mean_love, np.nan)],
    [np.full_like(mean_hate, np.nan), mean_hate],
])
cmap = plt.cm.RdBu_r.copy()
cmap.set_bad("white")
im = ax.imshow(combined, cmap=cmap, aspect="auto")
d = mean_love.shape[0]
ax.axhline(d - 0.5, color="k", lw=2)
ax.axvline(d - 0.5, color="k", lw=2)
ax.text(d / 2 - 0.5, -0.8, "love mean", ha="center", color=palette["love"], fontsize=9)
ax.text(d + d / 2 - 0.5, -0.8, "hate mean", ha="center", color=palette["hate"], fontsize=9)
ax.set_title("Riemannian mean metric tensor\nper class (top-left vs bottom-right)")
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(im, ax=ax)

ax = axes[2]
correct = y_pred == y
for cls, verb in enumerate(["love", "hate"]):
    mask = y == cls
    fc = [palette[verb] if ok else "lightgray" for ok in correct[mask]]
    ec = ["k" if ok else "#888888" for ok in correct[mask]]
    ax.scatter(
        dist_love[mask], dist_hate[mask],
        c=fc, edgecolors=ec,
        marker="o" if cls == 0 else "s",
        s=90, linewidths=0.6,
        label=f"{verb} ({'correct' if all(correct[mask]) else 'some errors'})",
    )
lims = [
    min(dist_love.min(), dist_hate.min()) * 0.95,
    max(dist_love.max(), dist_hate.max()) * 1.05,
]
ax.plot(lims, lims, "k--", lw=1, label="decision boundary")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_title(f"MDM decision space\nAccuracy: {correct.mean():.0%}")
ax.set_xlabel("Riemannian dist. to love centroid")
ax.set_ylabel("Riemannian dist. to hate centroid")
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()


###############################################################################
# References
# ----------
# .. [1] `10 Claude Analysis Prompts That Felt Like Magic
#    <https://medium.com/@ThinkingLoop/10-claude-analysis-prompts-that-felt-like-magic-42e05f63aa3e>`_
#    ThinkingLoop, Medium.
#
# .. [2] `Curved Inference: Concern-Sensitive Geometry in Large Language Model
#    Residual Streams
#    <https://arxiv.org/abs/2507.21107v1>`_
#    R. Manson, Jul. 08, 2025.
