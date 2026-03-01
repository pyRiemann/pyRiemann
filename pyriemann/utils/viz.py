"""Helpers for vizualization."""
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np

from ..embedding import SpectralEmbedding, LocallyLinearEmbedding, TSNE


def plot_embedding(
    X,
    y=None,
    *,
    embd_type="Spectral",
    metric="riemann",
    title="Embedding of SPD matrices",
    normalize=True,
    max_iter=50,
):
    """Plot embedding of SPD matrices.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    y : None | ndarray, shape (n_matrices,), default=None
        Labels for each matrix.
    embd_type : {"Spectral", "LocallyLinear", "TSNE"}, default="Spectral"
        Type of the embedding.
    metric : string, default="riemann"
        Metric used for the embedding. Can be:

        - "riemann", "logeuclid", "euclid", "logdet", "kullback",
          "kullback_right", "kullback_sym" for Spectral Embedding;
        - "riemann", "logeuclid", "euclid" for Locally Linear Embedding;
        - "riemann", "logeuclid", "euclid" for TSNE.
    title : str, default="Embedding of SPD matrices"
        Title of figure.
    normalize : bool, default=True
        If True, the plot is normalized from -1 to +1.
    max_iter : int, default=50
        Maximum number of iterations used for the gradient descent of TSNE.

    Returns
    -------
    fig : matplotlib figure
        Figure of embedding.

    Notes
    -----
    .. versionadded:: 0.2.6
    """
    if embd_type == "Spectral":
        e = SpectralEmbedding(n_components=2, metric=metric)
    elif embd_type == "LocallyLinear":
        e = LocallyLinearEmbedding(n_components=2,
                                   n_neighbors=X.shape[1],
                                   metric=metric)
    elif embd_type == "TSNE":
        e = TSNE(n_components=2, metric=metric, max_iter=max_iter)
    else:
        raise ValueError(
            f"Unknown embedding type {embd_type}. "
            "Valid types are 'Spectral', 'LocallyLinear' or 'TSNE'."
        )

    embd = e.fit_transform(X)
    if y is None:
        y = np.ones(embd.shape[0])

    if embd_type in ["Spectral", "LocallyLinear"]:
        fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
        for label in np.unique(y):
            idx = y == label
            ax.scatter(embd[idx, 0], embd[idx, 1], s=36)
        if normalize:
            ax.set_xticks([-1.0, -0.5, 0.0, +0.5, 1.0])
            ax.set_yticks([-1.0, -0.5, 0.0, +0.5, 1.0])

    else:
        fig = plt.subplots(figsize=(7, 7), facecolor="white")
        ax = plt.axes(projection="3d")
        for label in np.unique(y):
            idx = y == label
            ax.scatter(embd[idx, 0, 0], embd[idx, 0, 1], embd[idx, 1, 1], s=36)
        ax.set_zlabel(r"$\varphi_3$", fontsize=16)
        if normalize:
            ax.set_xticks([-1.0, -0.5, 0.0, +0.5, 1.0])
            ax.set_yticks([-1.0, -0.5, 0.0, +0.5, 1.0])
            ax.set_zticks([-1.0, -0.5, 0.0, +0.5, 1.0])

    ax.set_title(f"{embd_type} {title}", fontsize=16)
    ax.set_xlabel(r"$\varphi_1$", fontsize=16)
    ax.set_ylabel(r"$\varphi_2$", fontsize=16)
    ax.grid(False)
    ax.legend(list(np.unique(y)), title="Classes")

    return fig


def plot_cospectra(X, freqs, *, ylabels=None, title="Cospectra"):
    """Plot cospectral matrices.

    Parameters
    ----------
    X : ndarray, shape (n_freqs, n_channels, n_channels)
        Cospectral matrices.
    freqs : ndarray, shape (n_freqs,)
        The frequencies associated to cospectra.
    ylabels : list of str, default=None
        ylabels of figure.
    title : str, default="Cospectra"
        Title of figure.

    Returns
    -------
    fig : matplotlib figure
        Figure of cospectra.

    Notes
    -----
    .. versionadded:: 0.2.7
    """
    if X.ndim != 3:
        raise ValueError("Input X has not 3 dimensions")
    n_freqs, n_channels, _ = X.shape
    if freqs.shape != (n_freqs,):
        raise ValueError(
            "Input freqs has not the same number of frequencies as X")

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(title)
    for f in range(n_freqs):
        ax = plt.subplot((n_freqs - 1) // 8 + 1, 8, f + 1)
        plt.imshow(X[f], cmap=plt.get_cmap("Reds"))
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


def plot_waveforms(X, display, *, times=None, color="gray", alpha=0.5,
                   linewidth=1.5, color_mean="k", color_std="gray", n_bins=50,
                   cmap=None):
    """Plot repetitions of a multichannel waveform.

    Parameters
    ----------
    X : ndarray, shape (n_reps, n_channels, n_times)
        Repetitions of the multichannel waveform.
    display : {"all", "mean", "mean+/-std", "hist"}
        Type of display:

        * "all" for all the repetitions;
        * "mean" for the mean of the repetitions;
        * "mean+/-std" for the mean +/- standard deviation of the repetitions;
        * "hist" for the 2D histogram of the repetitions.
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
    if X.ndim != 3:
        raise ValueError("Input X has not 3 dimensions")
    n_reps, n_channels, n_times = X.shape
    if times is None:
        times = np.arange(n_times)
    elif times.shape != (n_times,):
        raise ValueError(
            "Parameter times has not the same number of values as X")

    fig, axes = plt.subplots(nrows=n_channels, ncols=1)
    if n_channels == 1:
        axes = [axes]
    channels = np.arange(n_channels)

    if display == "all":
        for (channel, ax) in zip(channels, axes):
            for i_rep in range(n_reps):
                ax.plot(times, X[i_rep, channel], c=color, alpha=alpha)

    elif display in ["mean", "mean+/-std"]:
        mean = np.mean(X, axis=0)
        for (channel, ax) in zip(channels, axes):
            ax.plot(times, mean[channel], c=color_mean, lw=linewidth)
        if display == "mean+/-std":
            std = np.std(X, axis=0)
            for (channel, ax) in zip(channels, axes):
                ax.fill_between(times, mean[channel] - std[channel],
                                mean[channel] + std[channel], color=color_std)

    elif display == "hist":
        times_rep = np.repeat(times[np.newaxis, :], n_reps, axis=0)
        for (channel, ax) in zip(channels, axes):
            ax.hist2d(times_rep.ravel(), X[:, channel, :].ravel(),
                      bins=(n_times, n_bins), cmap=cmap)

    else:
        raise ValueError(f"Unknown parameter display {display}")

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


def plot_cov_ellipse(ax, X, n_std=2.5, **kwds):
    """Plot 2x2 covariance matrix as an ellipse.

    Parameters
    ----------
    ax : matplotlib axis
        Axis of figure.
    X : ndarray, shape (2, 2)
        Covariance matrix.
    n_std : float, default=2.5
        Number of standard deviations.
    **kwds : dict
        Any further parameters are passed directly to the Ellipse.

    Returns
    -------
    ax : matplotlib axis
        Axis of figure.

    Notes
    -----
    .. versionadded:: 0.6

    References
    ----------
    .. [1] https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    """  # noqa
    if X.shape != (2, 2):
        raise ValueError("Input X must be a 2x2 covariance matrix")

    pearson = X[0, 1] / np.sqrt(X[0, 0] * X[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor='none', **kwds)
    scale_x = np.sqrt(X[0, 0]) * n_std
    scale_y = np.sqrt(X[1, 1]) * n_std
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y)
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

    return ax


def plot_bihist(X, y, n_bins=10, title="Histogram"):
    """Plot histogram of bi-class predictions.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, 2)
        Predictions, distances or probabilities.
    y : ndarray, shape (n_matrices,)
        Labels for each matrix.
    n_bins : int, default=10
        Number of bins of histogram.
    title : str, default="Histogram"
        Title of figure.

    Returns
    -------
    fig : matplotlib figure
        Figure of histogram.

    Notes
    -----
    .. versionadded:: 0.6
    """
    if X.ndim != 2:
        raise ValueError("Input X has not 2 dimensions")
    if X.shape[1] != 2:
        raise ValueError("Input X has not 2 classes")

    classes = np.unique(y)
    if classes.shape[0] != 2:
        raise ValueError("Input y has not 2 labels")

    X = X / np.sum(X, axis=1, keepdims=True)
    X0 = X[y == classes[0], 0]
    X1 = 1 - X[y == classes[1], 1]

    def get_bins(X, n_bins, target=0.5):
        """Estimate bins with the garantee to have target value in bin edges"""
        bins = np.histogram_bin_edges(X, bins=n_bins)
        idx = (np.abs(bins - target)).argmin()
        bins[idx] = target
        return bins

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axvline(x=0.5, c="k", linestyle=":")
    ax.hist(X0, bins=get_bins(X0, n_bins), label=classes[0], alpha=0.5)
    ax.hist(X1, bins=get_bins(X1, n_bins), label=classes[1], alpha=0.5)

    (Xmin, Xmax) = ax.get_xlim()
    Xm = min(Xmin, 1 - Xmax)
    ax.set_xlim(Xm, 1 - Xm)
    ax.set(xlabel="Rescaled predictions", ylabel="Frequency", title=title)
    ax.legend(title="Classes", loc="upper left")

    return fig


def plot_biscatter(X, y):
    """Plot scatter of bi-class predictions.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, 2)
        Predictions, distances or probabilities.
    y : ndarray, shape (n_matrices,)
        Labels for each matrix.

    Returns
    -------
    fig : matplotlib figure
        Figure of scatter plot.

    Notes
    -----
    .. versionadded:: 0.6
    """

    if X.ndim != 2:
        raise ValueError("Input X has not 2 dimensions")
    if X.shape[1] != 2:
        raise ValueError("Input X has not 2 classes")

    classes = np.unique(y)
    if classes.shape[0] != 2:
        raise ValueError("Input y has not 2 labels")

    X0 = X[y == classes[0]]
    X1 = X[y == classes[1]]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X0[:, 0], X0[:, 1], label=classes[0], alpha=1)
    ax.scatter(X1[:, 0], X1[:, 1], label=classes[1], alpha=0.5)
    ax.legend(title="Classes", loc="upper left")

    (Xmin, Xmax) = ax.get_xlim()
    (Ymin, Ymax) = ax.get_ylim()
    XYmin, XYmax = min(Xmin, Ymin), max(Xmax, Ymax)
    ax.plot([XYmin, XYmax], [XYmin, XYmax], c="k", linestyle=":")
    ax.set_xlim([XYmin, XYmax])
    ax.set_ylim([XYmin, XYmax])

    return fig
