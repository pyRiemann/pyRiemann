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

def plot_embedding_diffmaps(X, y=None, metric='riemann', eps=None, tdiff=0):
    """Plot 2D embedding of covariance matrices using Diffusion maps"""
    
    diffmaps = Embedding(metric)
    u,l = diffmaps.fit_transform(X)

    if y is None:
        y = np.ones(u.shape[0])

    fig = plt.figure(figsize=(6.24,5.63))
    for label in np.unique(y):
        idx = (y == label)
        print label
        plt.scatter(u[1,idx], u[2,idx])
    plt.xlabel(r'$\varphi_1$', fontsize=18)        
    plt.ylabel(r'$\varphi_2$', fontsize=18)        
    plt.title('2D embedding via Diffusion Maps', fontsize=20)    
        
    return fig       






