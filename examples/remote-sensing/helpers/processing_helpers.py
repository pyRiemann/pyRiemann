"""
=================================
Processing Remote Sensing Helpers
=================================

This file contains helper functions for handling remote sensing processes
"""

import numpy as np
import numpy.linalg as la
from sklearn.base import BaseEstimator, TransformerMixin
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike
from math import ceil


###############################################################################
def pca_image(image: ArrayLike, nb_components: int):
    """ A function that centers data and applies PCA on an image.

    Parameters
    ----------
        image : ndarray, shape (n_rows, n_columns, n_features)
            An image of shape (n_rows, n_columns, n_features).
        nb_components : int
            Number of components to keep.

    Written by Antoine Collas for:
    https://github.com/antoinecollas/pyCovariance/
    """
    # center pixels
    h, w, p = image.shape
    X = image.reshape((h*w, p))
    mean = np.mean(X, axis=0)
    image = image - mean
    X = X - mean
    # check pixels are centered
    assert (np.abs(np.mean(X, axis=0)) < 1e-8).all()

    # apply PCA
    SCM = (1/len(X))*X.conj().T@X
    d, Q = la.eigh(SCM)
    reverse_idx = np.arange(len(d)-1, -1, step=-1)
    Q = Q[:, reverse_idx]
    Q = Q[:, :nb_components]
    image = image@Q

    return image


###############################################################################
class RemoveMeanImage(BaseEstimator, TransformerMixin):
    """ Remove the global mean of an three-dimensional image.
    """
    def fit(self, X: ArrayLike, y=None):
        return self

    def transform(self, X: ArrayLike):
        return X - np.mean(X, axis=(0, 1))

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)


class PCAImage(BaseEstimator, TransformerMixin):
    """PCA on an image of shape (n_rows, n_columns, n_features).

    Parameters
    ----------
    n_components : int
        Number of components to keep.
    """

    def __init__(self, n_components: int):
        assert n_components > 0, 'Number of components must be positive.'
        self.n_components = n_components

    def fit(self, X: ArrayLike, y=None):
        return self

    def transform(self, X: ArrayLike):
        if self.n_components == X.shape[2]:
            return X
        return pca_image(X, self.n_components)

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)


class SlidingWindowVectorize(BaseEstimator, TransformerMixin):
    """Sliding window over an image of shape (n_rows, n_columns, n_features).

    Parameters
    ----------
    window_size : int
        Size of the window.
    overlap : int
        Overlap between windows. Default is 0.
    """

    def __init__(self, window_size: int, overlap: int = 0):
        assert window_size % 2 == 1, 'Window size must be odd.'
        assert overlap >= 0, 'Overlap must be positive.'
        assert overlap <= window_size//2, \
            'Overlap must be smaller or equal than int(window_size/2).'
        self.window_size = window_size
        self.overlap = overlap

    def fit(self, X: ArrayLike, y=None):
        return self

    def transform(self, X: ArrayLike):
        """Transforms a multidimensional array (ndim=3) into a sliding window
        view over the first two dimensions.

        Parameters
        ----------
        X: ndarray, shape (n_rows, n_columns, n_features)
            input array

        Returns
        -------
        ndarray, shape (n_pixels, window_size**2, n_features)
            output array, with n_pixels =
                (n_rows-window_size+1)(n_columns-window_size+1)//overlap^2
        """
        X = sliding_window_view(
                X,
                window_shape=(self.window_size, self.window_size),
                axis=(0, 1))
        if self.overlap is not None:
            if self.overlap > 0:
                X = X[::self.overlap, ::self.overlap]
        else:
            X = X[::self.window_size//2, ::self.window_size//2]
            self.overlap = self.window_size//2

        # Reshape to (n_pixels, n_samples, n_features) with
        # n_pixels = axis0*axis1
        # n_samples = axis3*axis_4
        # n_features = axis2
        X = X.reshape((-1, X.shape[2], X.shape[3]*X.shape[4]))
        return X

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)


class LabelsToImage(BaseEstimator, TransformerMixin):
    """Predicted labels to image taking into account sliding windows.

    Parameters
    ----------
    height : int
        Height of the original image.
    width : int
        Width of the original image.
    window_size : int
        Size of the window.
    """

    def __init__(self, height: int, width: int,
                 window_size: int, overlap: int = 0):
        assert window_size % 2 == 1, 'Window size must be odd.'
        assert overlap >= 0, 'Overlap must be positive.'
        assert overlap <= window_size//2, \
            'Overlap must be smaller or equal than int(window_size/2).'
        self.height = height
        self.width = width
        self.overlap = overlap
        self.window_size = window_size

    def fit(self, X: ArrayLike, y=None):
        return self

    def transform(self, X: ArrayLike):
        # Compute reshape size thanks ot window-size before overlap
        height = self.height - self.window_size + 1
        width = self.width - self.window_size + 1
        # Taking into account overlap
        if self.overlap > 0:
            height = ceil(height/self.overlap)
            width = ceil(width/self.overlap)

        # Reshape to (height, weight)
        return X.reshape((height, width))

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)
