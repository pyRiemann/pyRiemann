"""Helpers for vizualization."""
import numbers
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from pyriemann.embedding import Embedding
from pyriemann.utils import deprecated


@deprecated(
    "plot_confusion_matrix is deprecated and will be remove in 0.4.0; "
    "please use sklearn confusion_matrix and ConfusionMatrixDisplay; "
    "see examples/ERP/plot_classify_EEG_tangentspace.py"
)
def plot_confusion_matrix(
    targets, predictions, target_names, title="Confusion matrix", cmap="Blues"
):
    """Plot Confusion Matrix."""
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("Install seaborn to plot confusion matrix")

    cm = confusion_matrix(targets, predictions)
    cm = 100 * cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    df = pd.DataFrame(data=cm, columns=target_names, index=target_names)
    g = sns.heatmap(
        df, annot=True, fmt=".1f", linewidths=0.5, vmin=0, vmax=100, cmap=cmap
    )
    g.set_title(title)
    g.set_ylabel("True label")
    g.set_xlabel("Predicted label")
    return g


def plot_embedding(
    X, y=None, metric="riemann", title="Spectral embedding of covariances"
):
    """Plot 2D embedding of covariance matrices using Diffusion maps."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Install matplotlib to plot embeddings")
    lapl = Embedding(n_components=2, metric=metric)
    embd = lapl.fit_transform(X)

    if y is None:
        y = np.ones(embd.shape[0])

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
    for label in np.unique(y):
        idx = y == label
        ax.scatter(embd[idx, 0], embd[idx, 1], s=36)

    ax.set_xlabel(r"$\varphi_1$", fontsize=16)
    ax.set_ylabel(r"$\varphi_2$", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.grid(False)
    ax.set_xticks([-1.0, -0.5, 0.0, +0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0.0, +0.5, 1.0])
    ax.legend(list(np.unique(y)))

    return fig


def plot_cospectra(cosp, freqs, ylabels=None, title="Cospectra"):
    """Plot cospectral matrices

    Parameters
    ----------
    cosp : ndarray, shape (n_freqs, n_channels, n_channels)
        ndarray of cospectra.
    freqs : ndarray, shape (n_freqs,)
        The frequencies associated to cospectra.

    Returns
    -------
    fig : matplotlib figure
        Figure of cospectra.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Install matplotlib to plot cospectra")
    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(title)
    n_freqs = min(cosp.shape[0], freqs.shape[0])
    for f in range(n_freqs):
        ax = plt.subplot((n_freqs - 1) // 8 + 1, 8, f + 1)
        plt.imshow(cosp[f], cmap=plt.get_cmap("Reds"))
        plt.title("{} Hz".format(freqs[f]))
        plt.xticks([])
        if ylabels and f == 0:
            plt.yticks(np.arange(0, len(ylabels), 2), ylabels[::2])
            ax.tick_params(axis="both", which="major", labelsize=7)
        elif ylabels and f == 8:
            plt.yticks(np.arange(1, len(ylabels), 2), ylabels[1::2])
            ax.tick_params(axis="both", which="major", labelsize=7)
        else:
            plt.yticks([])

    return fig


def plot_erp(X, display='all', *, chax=0, t=None, **kwargs):
    ''' Display repetitions of a multichannel ERP.

    Parameters
    ----------
    X : ndarray, shape (n_reps, n_channels, n_times)
        Repetitions of the multichannel ERP.

    display : {'all', 'mean', 'hist'}
        Type of display:

        * 'all' for all the repetitions;
        * 'mean+/-std' for the mean +/- standard deviation of the repetitions;
        * 'hist' for the 2D histogram of the repetitions.

    chax : int | ndarray, shape(n_channels,) of subplots (default 0)
        If `chax` is an integer, it defines the channel index to display, and
        function returns only the axis with the figure of this channel.
        If `chax` is an array of shape(n_channels,) of subplots, all channels
        are displayed (one by subplot) and function returns the axes of all
        channels.

    t : None | ndarray, shape (n_times,) (default None)
        Values to display time on x-axis.

    Returns
    -------
    axes : figure axis
        Axis of the figure.
    '''
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Install matplotlib to plot erp")

    def _plot_erp_all(ax, t, X, **kwargs):
        color = kwargs.get('color', 'gray')
        alpha = kwargs.get('alpha', 0.5)
        for i_rep in range(X.shape[0]):
            ax.plot(t, X[i_rep, :], color=color, alpha=alpha)

    def _plot_erp_mean(ax, t, mean, std, **kwargs):
        linewidth = kwargs.get('linewidth', 1.5)
        color_mean = kwargs.get('color_mean', 'k')
        color_std = kwargs.get('color_std', 'gray')
        ax.plot(t, mean, color=color_mean, linewidth=linewidth)
        ax.fill_between(t, mean - std, mean + std, color=color_std) 

    def _plot_erp_hist(ax, t, X, **kwargs):
        n_bins = kwargs.get('n_bins', 50)
        cmap = kwargs.get('cmap', plt.cm.Greys)
        ax.hist2d(t.ravel(), X.ravel(), bins=(X.shape[-1], n_bins), cmap=cmap)

    if X.ndim != 3:
        raise Exception('Input X has not 3 dimensions')
    n_reps, n_channels, n_times = X.shape
    if t is None:
        t = np.arange(n_times)
    elif t.shape != (n_times,):
        raise Exception(
            'Parameter t has not the same number of times as X')

    if isinstance(chax, numbers.Integral):
        channels = [chax]
        axes = [plt.gca()]
    elif isinstance(chax, np.ndarray):
        if chax.shape != (n_channels,):
            raise Exception(
                'Parameter chax has not the same number of channels as X')
        channels = np.arange(n_channels)
        axes = chax
    else:
        raise Exception('Parameter chax unknown %s' % chax)

    if display == 'all':
        for (channel, ax) in zip(channels, axes):
            _plot_erp_all(ax, t, X[:, channel, :], **kwargs)

    elif display == 'mean+/-std':
        mean, std = np.mean(X, axis=0), np.std(X, axis=0)
        for (channel, ax) in zip(channels, axes):
            _plot_erp_mean(ax, t, mean[channel, :], std[channel, :], **kwargs)

    elif display == 'hist':
        t_rep = np.repeat(t[np.newaxis, :], n_reps, axis=0)
        for (channel, ax) in zip(channels, axes):
            _plot_erp_hist(ax, t_rep, X[:, channel, :], **kwargs)

    else:
        raise Exception('Parameter display unknown %s' % display)

    if isinstance(chax, numbers.Integral):
        return axes[0]
    elif isinstance(chax, np.ndarray):
        for ax in axes[:-1]:
            ax.set_xticklabels([])  # remove xticklabels
        return axes
