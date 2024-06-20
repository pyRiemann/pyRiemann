"""
=================================
Processing Remote Sensing Helpers
=================================

This file contains helper functions for handling remote sensing processes
"""
from math import ceil

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


###############################################################################

class RemoveMeanImage(BaseEstimator, TransformerMixin):
    """Mean removal for three-dimensional image."""
    def fit(self, X: ArrayLike, y=None):
        return self

    def transform(self, X: ArrayLike):
        return X - np.mean(X, axis=(0, 1))

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)


class PCAImage(BaseEstimator, TransformerMixin):
    """PCA for three-dimensional image.

    Parameters
    ----------
    n_components : int, float or 'mle', default=None
        Number of components to keep, passed to sklearn.decomposition.PCA
    """

    def __init__(self, n_components: int | float | str = None):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, X: ArrayLike, y=None):
        return self

    def transform(self, X: ArrayLike):
        """Apply PCA over image to reduce the dimensions.

        Parameters
        ----------
        X : ndarray, shape (n_rows, n_columns, n_features)
            Input image

        Returns
        -------
        Xnew : ndarray, shape (n_rows, n_columns, n_components)
            Output image.
        """
        # Reshaping to pass it to sklearn PCA
        X_new = X.reshape((np.prod(X.shape[:2]), X.shape[2]))
        X_new = self.pca.fit_transform(X_new)
        return X_new.reshape(X.shape[:2] + (X_new.shape[-1],))

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)


class SlidingWindowVectorize(BaseEstimator, TransformerMixin):
    """Sliding window for three-dimensional image.

    Parameters
    ----------
    window_size : int
        Size of the window.
    overlap : int, default=0
        Overlap between windows.
    """

    def __init__(self, window_size: int, overlap: int = 0):
        assert window_size % 2 == 1, "Window size must be odd."
        assert overlap >= 0, "Overlap must be positive."
        assert overlap <= window_size//2, \
            "Overlap must be smaller or equal than int(window_size/2)."
        self.window_size = window_size
        self.overlap = overlap

    def fit(self, X: ArrayLike, y=None):
        return self

    def transform(self, X: ArrayLike):
        """Transforms a multidimensional array (ndim=3) into a sliding window
        view over the first two dimensions.

        Parameters
        ----------
        X : ndarray, shape (n_rows, n_columns, n_features)
            Input image

        Returns
        -------
        Xnew : ndarray, shape (n_pixels, window_size**2, n_features)
            Output array, with n_pixels =
                (n_rows-window_size+1)(n_columns-window_size+1)//overlap^2
        """
        X = sliding_window_view(
            X,
            window_shape=(self.window_size, self.window_size),
            axis=(0, 1),
        )
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
    n_rows : int
        n_rows of the original image.
    n_columns : int
        n_columns of the original image.
    window_size : int
        Size of the window.
    """

    def __init__(self, n_rows: int, n_columns: int,
                 window_size: int, overlap: int = 0):
        assert window_size % 2 == 1, "Window size must be odd."
        assert overlap >= 0, "Overlap must be positive."
        assert overlap <= window_size//2, \
            "Overlap must be smaller or equal than int(window_size/2)."
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.overlap = overlap
        self.window_size = window_size

    def fit(self, X: ArrayLike, y=None):
        return self

    def transform(self, X: ArrayLike):
        """Transforms the output of a classifier over a sliding windows from
        SlidingWindowVectorize back to an image shape.

        Parameters
        ----------
        X : ndarray, shape (n_pixels,)
            Input classes

        Returns
        -------
        ndarray, shape (H, W)
            Output classified image with H = (n_rows-window_size+1)//overlap
            and W = (n_columns-window_size+1)//overlap

        """
        # Compute reshape size thanks ot window_size before overlap
        n_rows = self.n_rows - self.window_size + 1
        n_columns = self.n_columns - self.window_size + 1
        # Taking into account overlap
        if self.overlap > 0:
            n_rows = ceil(n_rows/self.overlap)
            n_columns = ceil(n_columns/self.overlap)

        return X.reshape((n_rows, n_columns))

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)
