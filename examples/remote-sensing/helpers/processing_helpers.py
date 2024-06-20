"""
=================================
Processing Remote Sensing Helpers
=================================

This file contains helper functions for handling remote sensing processes
"""
from math import ceil

import numpy as np
import numpy.linalg as la
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


###############################################################################

def pca_image(image: ArrayLike, n_components: int):
    """ A function that centers data and applies PCA on an image.

    Parameters
    ----------
    image : ndarray, shape (n_rows, n_columns, n_features)
        An image.
    n_components : int
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
    Q = Q[:, :n_components]
    image = image@Q

    return image


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
        Number of components to keep, passed to sklearn.decomposition.PCA when
        data is real. Should be int when data is complex.
    """

    def __init__(self, n_components=None):
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
        if np.iscomplexobj(X):
            assert isinstance(self.n_components, int), \
                "n_components should be an int when using complex data."
            if self.n_components == X.shape[2]:
                return X
            return pca_image(X, self.n_components)

        # Reshaping to pass it to sklearn PCA when real
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
        """Keep in memory n_rows and n_columns of data for inverse_predict.

        Parameters
        ----------
        X : ndarray, shape (n_rows, n_columns, n_features)
            Input image
        """
        self.n_rows = X.shape[0]
        self.n_columns = X.shape[1]
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
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

    def inverse_predict(self, preds: ArrayLike) -> ArrayLike:
        """Transforms the prediction of a classifier over a sliding windows
        back to an image shape.

        Parameters
        ----------
        pred : ndarray, shape (n_pixels,)
            Input classes

        Returns
        -------
        ndarray, shape (H, W)
            Output classified image with H = (n_rows-window_size+1)//overlap
            and W = (n_columns-window_size+1)//overlap

        """
        # Compute reshape size thanks ot window_size before overlap
        H = self.n_rows - self.window_size + 1
        W = self.n_columns - self.window_size + 1
        # Taking into account overlap
        if self.overlap > 0:
            H = ceil(H/self.overlap)
            W = ceil(W/self.overlap)

        return preds.reshape((H, W))
