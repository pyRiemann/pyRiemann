"""Helpers for vizualization."""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from ..embedding import Embedding
from . import deprecated


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


def plot_cospectra(cosp, freqs, *, ylabels=None, title="Cospectra"):
    """Plot cospectral matrices.

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

    Notes
    -----
    .. versionadded:: 0.2.7
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Install matplotlib to plot cospectra")

    if cosp.ndim != 3:
        raise Exception('Input cosp has not 3 dimensions')
    n_freqs, n_channels, _ = cosp.shape
    if freqs.shape != (n_freqs,):
        raise Exception(
            'Input freqs has not the same number of frequencies as cosp')

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(title)
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


def plot_waveforms(X, display, *, times=None, **kwargs):
    ''' Display repetitions of a multichannel waveform.

    Parameters
    ----------
    X : ndarray, shape (n_reps, n_channels, n_times)
        Repetitions of the multichannel waveform.
    display : {'all', 'mean', 'mean+/-std', 'hist'}
        Type of display:

        * 'all' for all the repetitions;
        * 'mean' for the mean of the repetitions;
        * 'mean+/-std' for the mean +/- standard deviation of the repetitions;
        * 'hist' for the 2D histogram of the repetitions.
    time : None | ndarray, shape (n_times,) (default None)
        Values to display on x-axis.

    color : matplotlib color, optional
        Color of the lines, when ``display=all``.
    alpha : float, optional
        Alpha value used to cumulate repetitions, when ``display=all``.
    linewidth : float, optional
        Line width in points, when ``display=mean``.
    color_mean : matplotlib color, optional
        Color of the mean line, when ``display=mean``.
    color_std : matplotlib color, optional
        Color of the standard deviation area, when ``display=mean+/-std``.
    n_bins : int, optional
        Number of vertical bins for the 2D histogram, when ``display=hist``.
    cmap : Colormap or str, optional
        Color map for the histogram, when ``display=hist``.

    Returns
    -------
    fig : matplotlib figure
        Figure of waveform (one subplot by channel).

    Notes
    -----
    .. versionadded:: 0.2.8
    '''
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Install matplotlib to plot waveforms")

    if X.ndim != 3:
        raise Exception('Input X has not 3 dimensions')
    n_reps, n_channels, n_times = X.shape
    if times is None:
        times = np.arange(n_times)
    elif times.shape != (n_times,):
        raise Exception(
            'Parameter times has not the same number of values as X')

    fig, axes = plt.subplots(nrows=n_channels, ncols=1)
    if n_channels == 1:
        axes = [axes]
    channels = np.arange(n_channels)

    if display == 'all':
        color = kwargs.get('color', 'gray')
        alpha = kwargs.get('alpha', 0.5)
        for (channel, ax) in zip(channels, axes):
            for i_rep in range(n_reps):
                ax.plot(times, X[i_rep, channel], c=color, alpha=alpha)

    elif display in ['mean', 'mean+/-std']:
        linewidth = kwargs.get('linewidth', 1.5)
        color_mean = kwargs.get('color_mean', 'k')
        mean = np.mean(X, axis=0)
        for (channel, ax) in zip(channels, axes):
            ax.plot(times, mean[channel], c=color_mean, lw=linewidth)
        if display == 'mean+/-std':
            color_std = kwargs.get('color_std', 'gray')
            std = np.std(X, axis=0)
            for (channel, ax) in zip(channels, axes):
                ax.fill_between(times, mean[channel] - std[channel],
                                mean[channel] + std[channel], color=color_std)

    elif display == 'hist':
        n_bins = kwargs.get('n_bins', 50)
        cmap = kwargs.get('cmap', plt.cm.Greys)
        times_rep = np.repeat(times[np.newaxis, :], n_reps, axis=0)
        for (channel, ax) in zip(channels, axes):
            ax.hist2d(times_rep.ravel(), X[:, channel, :].ravel(),
                      bins=(n_times, n_bins), cmap=cmap)

    else:
        raise Exception('Parameter display unknown %s' % display)

    if n_channels > 1:
        for ax in axes[:-1]:
            ax.set_xticklabels([])  # remove xticklabels
    return fig
