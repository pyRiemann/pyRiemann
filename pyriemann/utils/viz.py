"""Helpers for vizualization."""
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from pyriemann.embedding import Embedding
import matplotlib.pyplot as plt


def plot_confusion_matrix(targets, predictions, target_names,
                          title='Confusion matrix', cmap="Blues"):
    """Plot Confusion Matrix."""
    cm = confusion_matrix(targets, predictions)
    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    df = pd.DataFrame(data=cm, columns=target_names, index=target_names)
    g = sns.heatmap(df, annot=True, fmt=".1f", linewidths=.5, vmin=0, vmax=100,
                    cmap=cmap)
    g.set_title(title)
    g.set_ylabel('True label')
    g.set_xlabel('Predicted label')
    return g


def plot_embedding(X, y=None, metric='riemann',
                   title='Spectral embedding of covariances'):
    """Plot 2D embedding of covariance matrices using Diffusion maps."""
    lapl = Embedding(n_components=2, metric=metric)
    embd = lapl.fit_transform(X)

    if y is None:
        y = np.ones(embd.shape[0])

    fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    for label in np.unique(y):
        idx = (y == label)
        ax.scatter(embd[idx, 0], embd[idx, 1], s=36)

    ax.set_xlabel(r'$\varphi_1$', fontsize=16)
    ax.set_ylabel(r'$\varphi_2$', fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.grid(False)
    ax.set_xticks([-1.0, -0.5, 0.0, +0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0.0, +0.5, 1.0])
    ax.legend(list(np.unique(y)))

    return fig
