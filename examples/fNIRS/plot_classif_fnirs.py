"""
====================================================================
Classify fNIRS data with block diagonal matrices for HbO and HbR
====================================================================

This example demonstrates how to classify functional near-infrared spectroscopy
(fNIRS) data using block diagonal matrices for oxyhemoglobin (HbO) and
deoxyhemoglobin (HbR) signals, using the ``BlockKernels`` estimator.
This estimator computes block kernel or covariance matrices for each block of
channels, allowing for separate processing of HbO and HbR signals.
We can then apply shrinkage to each block separately [1]_.
"""

# Author: Tim Näher
import itertools
import os
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from pyriemann.utils.covariance import covariances
from pyriemann.estimation import BlockCovariances, Shrinkage, kernel_functions
from pyriemann.classification import SVC
from pyriemann.utils.covariance import cov_est_functions


###############################################################################
# Parameters
# ----------

block_size = 62  # Size of each block (number of channels for HbO and HbR)
n_jobs = -1  # Use all available cores
cv_splits = 5  # Number of cross-validation folds
random_state = 42  # Random state for reproducibility

# Some example kernel metrics
kernel_metrics = ["poly", "rbf", "laplacian"]

# Some example covariance estimators
covariance_estimators = ["oas", "lwf"]

# Combine all metrics
all_metrics = kernel_metrics + covariance_estimators

# Generate all possible combinations of metrics for two blocks
metric_combinations = [
    list(tup) for tup in itertools.product(all_metrics, repeat=2)
]

# Shrinkage values to test
shrinkage_values = [
    1,
    0.01,
    [0.01, 0.1],  # different shrinkage for each block
]

###############################################################################
# Define the BlockKernels estimator
# ---------------------------------


class BlockKernels(BaseEstimator, TransformerMixin):
    """Estimation of block kernel or covariance matrices with
    customizable metrics and shrinkage.

    Perform a block matrix estimation for each given time series,
    computing either kernel matrices or covariance matrices for
    each block based on the specified metrics. Optionally apply
    shrinkage to each block's matrix.

    Parameters
    ----------
    block_size : int | list of int
        Sizes of individual blocks given as int for same-size blocks,
        or list for varying block sizes.
    metric : string | list of string, default='linear'
        The metric(s) to use when computing matrices between channels.
        For kernel matrices, supported metrics are those from
        ``pairwise_kernels``: 'linear', 'poly', 'polynomial',
        'rbf', 'laplacian', 'cosine', etc. For covariance matrices,
        supported estimators are those from pyRiemann:
        'scm', 'lwf', 'oas', 'mcd', etc.
        If a list is provided, it must match the number of blocks.
    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by
        breaking down the pairwise matrix into ``n_jobs`` even
        slices and computing them in parallel.
    shrinkage : float | list of float, default=0
        Shrinkage parameter(s) to regularize each block's matrix.
        If a single float is provided, it is applied to all blocks.
        If a list is provided, it must match the number of blocks.
    **kwds : dict
        Any further parameters are passed directly to the kernel function(s)
        or covariance estimator(s).

    See Also
    --------
    BlockCovariances
    Kernels
    Shrinkage
    """

    def __init__(
        self, block_size, metric="linear", n_jobs=None, shrinkage=0, **kwds
    ):
        self.block_size = block_size
        self.metric = metric
        self.n_jobs = n_jobs
        self.shrinkage = shrinkage
        self.kwds = kwds

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Multi-channel time series.
        y : None
            Not used, here for compatibility with scikit-learn API.

        Returns
        -------
        self : BlockKernels instance
            The BlockKernels instance.
        """
        return self

    def transform(self, X):
        """Estimate block kernel or covariance matrices
        with optional shrinkage.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Multi-channel time series.

        Returns
        -------
        M : ndarray, shape (n_samples, n_channels, n_channels)
            Block matrices (kernel or covariance matrices).
        """
        n_samples, n_channels, n_times = X.shape

        blocks = BlockCovariances._check_block_size(
            self.block_size,
            n_channels,
        )
        n_blocks = len(blocks)

        # Handle metric parameter
        if isinstance(self.metric, str):
            metrics = [self.metric] * n_blocks
        elif isinstance(self.metric, list):
            if len(self.metric) != n_blocks:
                raise ValueError(
                    f"Length of metric list ({len(self.metric)}) must"
                    f"match number of blocks ({n_blocks})"
                )
            metrics = self.metric
        else:
            raise ValueError(
                "Parameter metric must be a string or a list of strings."
            )

        # Handle shrinkage parameter
        if isinstance(self.shrinkage, (float, int)):
            shrinkages = [self.shrinkage] * n_blocks
        elif isinstance(self.shrinkage, list):
            if len(self.shrinkage) != n_blocks:
                raise ValueError(
                    f"Length of shrinkage list ({len(self.shrinkage)})"
                    f"must match number of blocks ({n_blocks})"
                )
            shrinkages = self.shrinkage
        else:
            raise ValueError(
                "Parameter shrinkage must be a float, or a list of floats."
            )

        M_matrices = []

        for i in range(n_samples):
            start = 0
            M_blocks = []
            for idx, (block_size, metric, shrinkage_value) in enumerate(
                zip(blocks, metrics, shrinkages)
            ):
                end = start + block_size
                # Extract the block of channels
                X_block = X[i, start:end, :]  # shape: (block_size, n_times)

                # Compute the matrix for this block
                if metric in kernel_functions:
                    # Compute kernel matrix
                    M_block = pairwise_kernels(
                        X_block, metric=metric, n_jobs=self.n_jobs, **self.kwds
                    )
                elif metric in cov_est_functions.keys():
                    # Compute covariance matrix
                    X_block_reshaped = X_block[np.newaxis, :, :]
                    M_block = covariances(
                        X_block_reshaped, estimator=metric, **self.kwds
                    )[0]
                else:
                    raise ValueError(
                        f"Metric '{metric}' is not recognized"
                        " as a kernel metric or a covariance estimator."
                    )

                # Apply shrinkage if specified
                if shrinkage_value != 0:
                    M_block_reshaped = M_block[np.newaxis, :, :]
                    shr = Shrinkage(shrinkage=shrinkage_value)
                    M_block = shr.fit_transform(M_block_reshaped)[0]

                M_blocks.append(M_block)
                start = end

            # Create the block diagonal matrix
            M_full = self._block_diag(M_blocks)
            M_matrices.append(M_full)

        return np.array(M_matrices)

    @staticmethod
    def _block_diag(matrices):
        """Construct a block diagonal matrix from a list of square matrices."""
        shapes = [m.shape[0] for m in matrices]
        total_size = sum(shapes)
        result = np.zeros((total_size, total_size), dtype=matrices[0].dtype)
        start = 0
        for m in matrices:
            end = start + m.shape[0]
            result[start:end, start:end] = m
            start = end
        return result


###############################################################################
# Download example data
# ---------------------

X_url = "https://zenodo.org/records/13841869/files/X.npy"
y_url = "https://zenodo.org/records/13841869/files/y.npy"

data_path = Path("./data")
data_path.mkdir(exist_ok=True)
X_path, y_path = data_path / "X.npy", data_path / "y.npy"


# Function to download the files if they don't already exist
def download_file(url, file_path):
    if not os.path.isfile(file_path):
        print(f"Downloading {file_path} from {url}")
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded {file_path}")


# Download
download_file(X_url, X_path)
download_file(y_url, y_path)

# Load the dataset
X, y = np.load(X_path), np.load(y_path)

print(
    f"Data loaded: {X.shape[0]} trials, {X.shape[1]} channels, "
    f"{X.shape[2]} time points"
)

###############################################################################
# Set up the pipeline
# -------------------

# Define the pipeline with BlockKernels and SVC classifier
pipeline = Pipeline(
    [
        ("block_kernels", BlockKernels(block_size=block_size)),
        ("classifier", SVC()),
    ]
)

###############################################################################
# Define hyperparameter cross-validation
# --------------------------------------

# Define the hyperparameter grid for grid search
param_grid = {
    "block_kernels__metric": metric_combinations,
    "block_kernels__shrinkage": shrinkage_values,
    "classifier__C": [0.1, 1, 10],
    "classifier__metric": ["euclid", "riemann", "logeuclid"],
}

# Define cross-validation
cv = StratifiedKFold(
    n_splits=cv_splits, shuffle=True, random_state=random_state
)

###############################################################################
# Run grid search
# ---------------

print("Starting grid search")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=n_jobs,
    verbose=1,
)

grid_search.fit(X, y)
print("Grid search completed")

###############################################################################
# Print results
# -------------

print(f"Best parameters:\n{grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

df_results = pd.DataFrame(grid_search.cv_results_)

# Display relevant columns
display_columns = [
    "mean_test_score",
    "std_test_score",
    "param_block_kernels__metric",
    "param_block_kernels__shrinkage",
]

print(
    df_results[display_columns].sort_values(
        by="mean_test_score", ascending=False
    )
)


###############################################################################
# Conclusion
# ----------

# The grid search allows us to find the best combination of metrics and
# shrinkage values for our fNIRS classification of mental imagery.
# By using block diagonal matrices for HbO and HbR signals, we can tune our
# classifier to HbO and HbR signals separately, which can improve
# classification performance.

###############################################################################
# References
# ----------
# .. [1] `Riemannian Geometry for the classification of brain states with fNIRS
#    <https://www.biorxiv.org/content/10.1101/2024.09.06.611347v1>`_
#    T. Näher, L. Bastian, A. Vorreuther, P. Fries, R. Goebel, B. Sorger.
