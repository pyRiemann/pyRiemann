"""
====================================================================
Remote sensing: Clustering on hyperspectral/sar images with Riemannian
geometry
====================================================================

This example compares clustering pipelines based on covariance matrices for
hyperspectral/sar data custering.
"""

# Author: Ammar Mian

import os
import urllib.request
import numpy as np
import numpy.linalg as la
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.clustering import Kmeans
from scipy.io import loadmat
from typing import Tuple, Dict

from numpy.lib.stride_tricks import sliding_window_view
from math import ceil 

from sklearn.base import BaseEstimator, TransformerMixin
from numpy.typing import ArrayLike


class RemoveMeanImage(BaseEstimator, TransformerMixin):
    """ Remove the global mean of an three-dimensional image.
    """
    def __init__(self):
        pass

    def fit(self, X: ArrayLike, y = None):
        return self

    def transform(self, X: ArrayLike):
        return X - np.mean(X, axis=(0, 1))

    def fit_transform(self, X: ArrayLike, y = None):
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

    def fit(self, X: ArrayLike, y = None):
        return self

    def transform(self, X: ArrayLike):
        if self.n_components == X.shape[2]:
            return X
        return pca_image(X, self.n_components)

    def fit_transform(self, X: ArrayLike, y = None):
        return self.fit(X).transform(X)


class SlidingWindowVectorize(BaseEstimator, TransformerMixin):
    """Slinding window over an image of shape (n_rows, n_columns, n_features).

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
        assert overlap <= window_size//2,\
                'Overlap must be smaller or equal than int(window_size/2).'
        self.window_size = window_size
        self.overlap = overlap

    def fit(self, X: ArrayLike, y = None):
        return self

    def transform(self, X: ArrayLike):
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

        # Reshape to (n_pixels, n_samples, n_features) with n_pixels=axis0*axis1
        # n_samples=axis3*axis_4 and n_features=axis2
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


def pca_image(image: ArrayLike, nb_components: int):
    """ A function that centers data and applies PCA on an image.

    Parameters
    ----------
        image : ArrayLike
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


def download_salinas(data_path: str):
    """Download the Salinas dataset.

    Parameters
    ----------
    data_path: str
        Path to the data folder to download the data.
    """
    urls = [
            'https://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat',
            'https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
            'https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat',
            ]
    filenames = [os.path.basename(url) for url in urls]
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    if not all([os.path.exists(os.path.join(data_path, filename))
                for filename in filenames]):
        print('Downloading Salinas dataset...')
        for url in urls:
            urllib.request.urlretrieve(url, os.path.join(data_path,
                                                 os.path.basename(url)))
        print('Done.')
    else:
        print('Salinas dataset already downloaded.')


def read_salinas(data_path: str, version: str="corrected") -> \
        Tuple[ArrayLike, ArrayLike, Dict[int, str]]:
    """
    Read Salinas hyperspectral data.

    Parameters
    ----------
    data_path: str
        Path to the data folder
    version: str
        Version of the data to read. Can be either "corrected" or "raw".
        Default is "corrected".


    Returns
    -------
    data: ArrayLike
        Data array of shape (512, 217, 204).

    labels: ArrayLike
        Labels array of shape (512, 217).

    labels_names: Dict[int, str]
        Dictionary mapping labels to their names.
    """
    if version == "corrected":
        data_file = os.path.join(data_path, "Salinas_corrected.mat")
    else:
        data_file = os.path.join(data_path, "Salinas.mat")
    data = loadmat(data_file)['salinas_corrected']
    labels = loadmat(os.path.join(data_path,
                                  'Salinas_gt.mat'))['salinas_gt']
    labels_names = {
                0: 'Undefined',
                1: 'Brocoli_green_weeds_1',
                2: 'Brocoli_green_weeds_2',
                3: 'Fallow',
                4: 'Fallow_rough_plow',
                5: 'Fallow_smooth',
                6: 'Stubble',
                7: 'Celery',
                8: 'Grapes_untrained',
                9: 'Soil_vinyard_develop',
                10: 'Corn_senesced_green_weeds',
                11: 'Lettuce_romaine_4wk',
                12: 'Lettuce_romaine_5wk',
                13: 'Lettuce_romaine_6wk',
                14: 'Lettuce_romaine_7wk',
                15: 'Vinyard_untrained',
                16: 'Vinyard_vertical_trellis'
            }
    return data, labels, labels_names


def dowload_uavsar(data_path: str):
    """Download the UAVSAR dataset.
    Parameters
    ----------
    data_path: str
        Path to the data folder to download the data.
    """
    url = 'https://zenodo.org/records/10625505/files/scene1.npy?download=1' 
    filename = "scene1.npy"
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    if not os.path.exists(os.path.join(data_path, filename)):
        print('Downloading UAVSAR dataset...')
        urllib.request.urlretrieve(url, os.path.join(data_path, filename))
        print('Done.')
    else:
        print('UAVSAR dataset already downloaded.')


if __name__ == "__main__":

    # Parameters
    window_size = 5
    data_path = './data'
    dataset = "uavsar"

    # Load data
    if dataset == "salinas":
        download_salinas(data_path)
        data, labels, labels_names = read_salinas(data_path)
        n_clusters = len(labels_names)
        n_components = 5
    elif dataset == "uavsar":
        dowload_uavsar(data_path)
        data = np.load(os.path.join(data_path, 'scene1.npy'))
        data = data[:,:,:,0] # Select one date only
        n_components = data.shape[2]
        n_clusters = 4
    else:
        raise ValueError('Unknown dataset.')
    height, width, n_features = data.shape

    # Pipelines definition
    pipeline_scm = Pipeline([
        ('remove_mean', RemoveMeanImage()),
        ('pca', PCAImage(n_components=n_components)),
        ('sliding_window', SlidingWindowVectorize(window_size=window_size)),
        ('covariances', Covariances(estimator='scm')),
        ('kmeans', Kmeans(n_clusters=n_clusters))],
        verbose=True)

    pipeline_tyler = Pipeline([
        ('remove_mean', RemoveMeanImage()),
        ('pca', PCAImage(n_components=n_components)),
        ('sliding_window', SlidingWindowVectorize(window_size=window_size)),
        ('covariances', Covariances(estimator='tyl')),
        ('kmeans', Kmeans(n_clusters=n_clusters))],
        verbose=True)

    pipelines = [pipeline_scm, pipeline_tyler]
    pipelines_names = ['SCM', 'Tyler']


    # Perform clustering
    results = {}
    for pipeline_name, pipeline in zip(pipelines_names, pipelines):
        print(f'Pipeline: {pipeline_name}')
        pipeline.fit(data)
        labels_pred = LabelsToImage(height, width, window_size).fit_transform(
                        pipeline.named_steps['kmeans'].labels_
                )
        results[pipeline_name] = labels_pred

    # Plot data
    if dataset == "uavsar":
        plot_value = np.sum(np.abs(data)**2, axis=2)
    else:
        plot_value = np.mean(data, axis=2)
    figure = plt.figure(figsize=(10, 5))
    plt.imshow(plot_value, cmap='gray')
    plt.title('Data')

    # Plot results
    for pipeline_name, labels_pred in results.items():
        figure = plt.figure(figsize=(10, 5))
        plt.imshow(labels_pred, cmap='tab20b')
        plt.title(f'Clustering results with {pipeline_name}')
        plt.colorbar()
    plt.show()







