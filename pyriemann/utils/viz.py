"""Helpers for vizualization."""
import numpy as np
from ..embedding import SpectralEmbedding, LocallyLinearEmbedding


def plot_embedding(X,
                   y=None,
                   *,
                   metric="riemann",
                   title="Embedding of covariances",
                   embd_type='Spectral',
                   normalize=True):
    """Plot 2D embedding of SPD matrices.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    y : None | ndarray, shape (n_matrices,), default=None
        Labels for each matrix.
    metric : string, default='riemann'
        Metric used in the embedding. Can be {'riemann' ,'logeuclid' ,
        'euclid'} for Locally Linear Embedding and {'riemann' ,'logeuclid' ,
        'euclid' , 'logdet', 'kullback', 'kullback_right', 'kullback_sym'}
        for Spectral Embedding.
    title : str, default="Embedding of covariances"
        Title string for plot.
    embd_type : {'Spectral' ,'LocallyLinear'}, default='Spectral'
        Embedding type.
    normalize : bool, default=True
        If True, the plot is normalized from -1 to +1.

    Returns
    -------
    fig : matplotlib figure
        Figure of embedding.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Install matplotlib to plot embeddings")

    if embd_type == 'Spectral':
        lapl = SpectralEmbedding(n_components=2, metric=metric)
    elif embd_type == 'LocallyLinear':
        lapl = LocallyLinearEmbedding(n_components=2,
                                      n_neighbors=X.shape[1],
                                      metric=metric)
    else:
        raise ValueError("Invalid embedding type. Valid types are: 'Spectral',"
                         " 'LocallyLinear'")

    embd = lapl.fit_transform(X)

    if y is None:
        y = np.ones(embd.shape[0])

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
    for label in np.unique(y):
        idx = y == label
        ax.scatter(embd[idx, 0], embd[idx, 1], s=36)

    ax.set_xlabel(r"$\varphi_1$", fontsize=16)
    ax.set_ylabel(r"$\varphi_2$", fontsize=16)
    ax.set_title(f'{embd_type} {title}', fontsize=16)
    ax.grid(False)
    if normalize:
        ax.set_xticks([-1.0, -0.5, 0.0, +0.5, 1.0])
        ax.set_yticks([-1.0, -0.5, 0.0, +0.5, 1.0])
    ax.legend(list(np.unique(y)))

    return fig


def plot_cospectra(cosp, freqs, *, ylabels=None, title="Cospectra"):
    """Plot cospectral matrices.

    Parameters
    ----------
    cosp : ndarray, shape (n_freqs, n_channels, n_channels)
        Cospectral matrices.
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


def plot_waveforms(X, display, *, times=None, color='gray', alpha=0.5,
                   linewidth=1.5, color_mean='k', color_std='gray', n_bins=50,
                   cmap=None):
    """ Display repetitions of a multichannel waveform.

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
    time : None | ndarray, shape (n_times,), default=None
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
    .. versionadded:: 0.3
    """
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
        for (channel, ax) in zip(channels, axes):
            for i_rep in range(n_reps):
                ax.plot(times, X[i_rep, channel], c=color, alpha=alpha)

    elif display in ['mean', 'mean+/-std']:
        mean = np.mean(X, axis=0)
        for (channel, ax) in zip(channels, axes):
            ax.plot(times, mean[channel], c=color_mean, lw=linewidth)
        if display == 'mean+/-std':
            std = np.std(X, axis=0)
            for (channel, ax) in zip(channels, axes):
                ax.fill_between(times, mean[channel] - std[channel],
                                mean[channel] + std[channel], color=color_std)

    elif display == 'hist':
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


def _add_alpha(colors, alphas):
    """Add alphas to RGB channels"""
    try:
        from matplotlib.colors import to_rgb
    except ImportError:
        raise ImportError("Install matplotlib to add alpha")

    cols = [to_rgb(c) for c in colors]
    return [(c[0], c[1], c[2], a) for c, a in zip(cols, alphas[-len(cols):])]
