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
    """Dimension reduction on 3rd dimension using PCA.

    Parameters
    ----------
    n_components : int, float or "mle", default=None
        Number of components to keep, passed to sklearn.decomposition.PCA when
        data is real. Should be int when data is complex.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X: ArrayLike, y=None):
        return self

    def transform(self, X: ArrayLike):
        """Reduce 3rd dimension of data using PCA.

        Parameters
        ----------
        X : ndarray, shape (n_rows, n_columns, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray, shape (n_rows, n_columns, n_components)
            Output data, reduced along its 3rd dimension.
        """
        if np.iscomplexobj(X):
            assert isinstance(self.n_components, int), \
                "n_components should be an int when using complex data."
            if self.n_components >= X.shape[2]:
                return X
            return self._complex_pca(X)

        # reshape to pass it to sklearn PCA when real
        X_new = X.reshape((np.prod(X.shape[:2]), X.shape[2]))
        pca = PCA(n_components=self.n_components)
        X_new = pca.fit_transform(X_new)
        return X_new.reshape(X.shape[:2] + (X_new.shape[-1],))

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)

    def _complex_pca(self, X: ArrayLike):
        """Center and reduce data by PCA.

        Parameters
        ----------
        X : ndarray, shape (n_rows, n_columns, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray, shape (n_rows, n_columns, n_components)
            Output data.
        """
        # center pixels
        n_rows, n_columns, n_features = X.shape
        Xr = X.reshape((n_rows*n_columns, n_features))
        Xr_mean = np.mean(Xr, axis=0)
        X = X - Xr_mean
        Xr = Xr - Xr_mean
        # check pixels are centered
        assert (np.abs(np.mean(Xr, axis=0)) < 1e-8).all()

        # apply PCA
        scm = Xr.conj().T @ Xr / len(Xr)
        _, eigvecs = np.linalg.eigh(scm)
        eigvecs = np.fliplr(eigvecs)
        X_new = X @ eigvecs[:, :self.n_components]
        return X_new


class SlidingWindowVectorize(BaseEstimator, TransformerMixin):
    """Sliding window for three-dimensional data.

    Parameters
    ----------
    window_size : int
        Size of the sliding window.
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
        """Keep in memory n_rows and n_columns of original data.

        Parameters
        ----------
        X : ndarray, shape (n_rows, n_columns, n_features)
            Input image

        Returns
        -------
        self : SlidingWindowVectorize instance
            The SlidingWindowVectorize instance.
        """
        self.n_rows, self.n_columns, _ = X.shape
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Transform original data with a sliding window.

        Transform original three-dimensional data into a sliding window view
        over the first two dimensions.

        Parameters
        ----------
        X : ndarray, shape (n_rows, n_columns, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray, shape (n_pixels, window_size**2, n_features)
            Output data, with n_pixels = (n_rows-window_size+1) x
            (n_columns-window_size+1) // overlap^2
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

        # reshape to (n_pixels, n_samples, n_features) with
        # n_pixels = axis0*axis1, n_samples = axis3*axis_4, n_features = axis2
        X_new = X.reshape((-1, X.shape[2], X.shape[3]*X.shape[4]))
        return X_new

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)

    def inverse_predict(self, y: ArrayLike) -> ArrayLike:
        """Transform predictions over sliding windows back to original data.

        Transform the predictions over sliding windows data back to original
        data shape.

        Parameters
        ----------
        y : ndarray, shape (n_pixels,)
            Predictions.

        Returns
        -------
        X : ndarray, shape (n_new_rows, n_new_columns)
            Output predicted data, with n_new_rows = (n_rows-window_size+1) //
            overlap and n_new_columns = (n_columns-window_size+1) // overlap.
        """
        # compute reshape size thanks to window_size before overlap
        n_new_rows = self.n_rows - self.window_size + 1
        n_new_columns = self.n_columns - self.window_size + 1

        # take into account overlap
        if self.overlap > 0:
            n_new_rows = ceil(n_new_rows/self.overlap)
            n_new_columns = ceil(n_new_columns/self.overlap)

        return y.reshape((n_new_rows, n_new_columns))
