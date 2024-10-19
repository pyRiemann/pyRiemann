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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline, make_pipeline

from pyriemann.classification import SVC
from pyriemann.estimation import (
    Covariances,
    Kernels,
    Shrinkage,
    ker_est_functions,
)
from pyriemann.utils.covariance import cov_est_functions


###############################################################################
# Parameters
# ----------

block_size = 62  # Size of each block (number of channels for HbO and HbR)
n_jobs = -1  # Use all available cores
cv_splits = 5  # Number of cross-validation folds
random_state = 42  # Random state for reproducibility

# Some example kernel metrics
kernel_metrics = ["poly", "cosine"]

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
    0.01,  # same shrinkage for all blocks
    0.1,  # same shrinkage for all blocks
    [0.01, 0.1],  # different shrinkage for each block
]

###############################################################################
# Define the BlockKernels estimator and helper transformers
# ---------------------------------------------------------


class Stacker(TransformerMixin):
    """Stacks values of a DataFrame column into a 3D array."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame), "Input must be a DataFrame"
        assert X.shape[1] == 1, "DataFrame must have only one column"
        return np.stack(X.iloc[:, 0].values)


class FlattenTransformer(BaseEstimator, TransformerMixin):
    """Flattens the last two dimensions of an array.
        ColumnTransformer requires 2D output, so this transformer
        is needed"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(X.shape[0], -1)


class BlockKernels(BaseEstimator, TransformerMixin):
    """Estimation of block kernel or covariance matrices with
    customizable metrics and shrinkage.

    Perform block matrix estimation for each given time series,
    computing either kernel matrices or covariance matrices for
    each block based on the specified metrics. The block matrices
    are then concatenated to form the final block diagonal matrices.
    It is possible to add different shrinkage values for each block.
    This estimator is helpful when dealing with data from multiple
    sources or modalities (e.g. fNIRS, EMG, EEG, gyro, acceleration),
    where each block corresponds to a different source or modality
    and benefits from separate processing and tuning.

    Parameters
    ----------
    block_size : int | list of int
        Sizes of individual blocks given as int for same-size blocks,
        or list for varying block sizes.
    metrics : string | list of string, default='linear'
        The metric(s) to use when computing matrices between channels.
        For kernel matrices, supported metrics are those from
        ``pairwise_kernels``: 'linear', 'poly', 'polynomial',
        'rbf', 'laplacian', 'cosine', etc.
        For covariance matrices, supported estimators are those from
        pyRiemann: 'scm', 'lwf', 'oas', 'mcd', etc.
        If a list is provided, it must match the number of blocks.
    shrinkage : float | list of float, default=0
        Shrinkage parameter(s) to regularize each block's matrix.
        If a single float is provided, it is applied to all blocks.
        If a list is provided, it must match the number of blocks.
    n_jobs : int, default=None
        The number of jobs to use for the computation.
    **kwargs : dict
        Any further parameters are passed directly to the kernel function(s)
        or covariance estimator(s).

    See Also
    --------
    BlockCovariances
    """

    def __init__(self, block_size, metrics="linear", shrinkage=0,
                 n_jobs=None, **kwargs):
        self.block_size = block_size
        self.metrics = metrics
        self.shrinkage = shrinkage
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """Fit.

        Prepare per-block transformers.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time series.
        y : None
            Not used, here for compatibility with scikit-learn API.

        Returns
        -------
        self : BlockKernels instance
            The BlockKernels instance.
        """
        n_samples, n_channels, n_times = X.shape

        # Determine block sizes
        if isinstance(self.block_size, int):
            num_blocks = n_channels // self.block_size
            remainder = n_channels % self.block_size
            self.blocks = [self.block_size] * num_blocks
            if remainder > 0:
                self.blocks.append(remainder)
        elif isinstance(self.block_size, list):
            self.blocks = self.block_size
            if sum(self.blocks) != n_channels:
                raise ValueError(
                    "Sum of block sizes mustequal number of channels"
                    )
        else:
            raise ValueError(
                "block_size must be int or list of ints"
                )

        # Compute block indices
        self.block_indices = []
        start = 0
        for size in self.blocks:
            end = start + size
            self.block_indices.append((start, end))
            start = end

        # Handle metrics parameter
        n_blocks = len(self.blocks)
        if isinstance(self.metrics, str):
            self.metrics_list = [self.metrics] * n_blocks
        elif isinstance(self.metrics, list):
            if len(self.metrics) != n_blocks:
                raise ValueError(
                    f"Length of metrics list ({len(self.metrics)}) "
                    f"must match number of blocks ({n_blocks})"
                )
            self.metrics_list = self.metrics
        else:
            raise ValueError(
                "Parameter 'metrics' must be a string or a list of strings."
            )

        # Handle shrinkage parameter
        if isinstance(self.shrinkage, (float, int)):
            self.shrinkages = [self.shrinkage] * n_blocks
        elif isinstance(self.shrinkage, list):
            if len(self.shrinkage) != n_blocks:
                raise ValueError(
                    f"Length of shrinkage list ({len(self.shrinkage)}) "
                    f"must match number of blocks ({n_blocks})"
                )
            self.shrinkages = self.shrinkage
        else:
            raise ValueError(
                "Parameter 'shrinkage' must be a float or a list of floats."
            )

        # Build per-block pipelines
        self.block_names = [f"block_{i}" for i in range(n_blocks)]

        transformers = []
        for i, (indices, metric, shrinkage_value) in enumerate(
                zip(self.block_indices, self.metrics_list, self.shrinkages)):
            block_name = self.block_names[i]

            # Build the pipeline for this block
            block_pipeline = make_pipeline(
                Stacker(),
            )

            # Determine if the metric is a kernel or a covariance estimator
            if metric in ker_est_functions:
                # Use Kernels transformer
                estimator = Kernels(
                    metric=metric,
                    n_jobs=self.n_jobs,
                    **self.kwargs
                )
                block_pipeline.steps.append(('kernels', estimator))
            elif metric in cov_est_functions.keys():
                # Use Covariances transformer
                estimator = Covariances(
                    estimator=metric,
                    **self.kwargs
                )
                block_pipeline.steps.append(('covariances', estimator))
            else:
                raise ValueError(
                    f"Metric '{metric}' is not recognized as a kernel "
                    f"metric or a covariance estimator."
                )

            # add shrinkage if provided
            # TODO: add support for different shrinkage types at some point?
            if shrinkage_value != 0:
                shrinkage_transformer = Shrinkage(shrinkage=shrinkage_value)
                block_pipeline.steps.append(
                    ('shrinkage', shrinkage_transformer)
                    )

            # Add the flattening transformer at the end of the pipeline
            block_pipeline.steps.append(('flatten', FlattenTransformer()))

            transformers.append((block_name, block_pipeline, [block_name]))

        # create the columncransformer with per-block pipelines
        self.preprocessor = ColumnTransformer(transformers)

        # Prepare the DataFrame
        X_df = self._prepare_dataframe(X)

        # Fit the preprocessor
        self.preprocessor.fit(X_df)

        return self

    def transform(self, X):
        """Estimate block kernel or covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Multi-channel time series.

        Returns
        -------
        M : ndarray, shape (n_samples, n_channels, n_channels)
            Block diagonal matrices (kernel or covariance matrices).
        """
        check_is_fitted(self, 'preprocessor')

        # make the df wheren each block is 1 column
        X_df = self._prepare_dataframe(X)

        # Transform the data
        transformed_blocks = []
        data_transformed = self.preprocessor.transform(X_df)

        # calculate the number of features per block
        features_per_block = [size * size for size in self.blocks]

        # compute the indices where to split the data
        split_indices = np.cumsum(features_per_block)[:-1]

        # split the data into flattened blocks
        blocks_flat = np.split(data_transformed, split_indices, axis=1)

        # reshape each block back to its original shape
        for i, block_flat in enumerate(blocks_flat):
            size = self.blocks[i]
            block = block_flat.reshape(-1, size, size)
            transformed_blocks.append(block)

        # Reconstruct the block diagonal matrices
        n_samples = X.shape[0]
        M_matrices = []
        for i in range(n_samples):
            blocks = [Xt[i] for Xt in transformed_blocks]
            M_full = self._block_diag(blocks)
            M_matrices.append(M_full)
        return np.array(M_matrices)

    def _prepare_dataframe(self, X):
        """Converts the data into a df with eac hblock as column."""
        data_dict = {}
        for i, (start, end) in enumerate(self.block_indices):
            data_dict[self.block_names[i]] = list(X[:, start:end, :])
        return pd.DataFrame(data_dict)

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
# Download example data and plot
# ------------------------------

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

# Get trials with the label "OP" for "Own Paradigm"
MT_label = "OP"
MT_trials_indices = np.where(y == MT_label)[0]

# Average the data across the "OP" trials
X_MT_erp = np.mean(X[MT_trials_indices, :, :], axis=0)

# select example channel
channel_index = 2

# Plot the averaged signals
plt.figure(figsize=(10, 5))
plt.plot(X_MT_erp[channel_index, :], label="HbO", color="red")
plt.plot(X_MT_erp[channel_index + 62, :], label="HbR", color="blue")
plt.xlabel("Time samples")
plt.ylabel("Signal Amplitude")
plt.title(f"ERP for OP [Own Paradigm] trials in channel {channel_index}")
plt.legend()
plt.show()

###############################################################################
# Set up the pipeline
# -------------------

# Define the pipeline with BlockKernels and SVC classifier
pipeline = Pipeline(
    [
        ("block_kernels", BlockKernels(block_size=block_size)),
        ("classifier", SVC(metric="riemann")),
    ]
)

###############################################################################
# Define hyperparameter cross-validation
# --------------------------------------

# Define the hyperparameter grid for grid search
param_grid = {
    "block_kernels__metrics": metric_combinations,
    "block_kernels__shrinkage": shrinkage_values,
    "classifier__C": [0.1, 1],
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
    "param_block_kernels__metrics",
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
#
# The grid search allows us to find the best combination of metrics and
# shrinkage values for our fNIRS classification of mental imagery.
# By using block diagonal matrices for HbO and HbR signals, we can tune our
# classifier to HbO and HbR signals separately, which can improve
# classification performance, since fNIRS classification methods are often
# lacking in this regard. Since there is so few data available, we just
# calculate cross validation scores. This means that the classification
# accuracies will be overestimated. If more data per class are available
# one should use a train-test split to get a more realistic estimate of the
# classification performance.

###############################################################################
# References
# ----------
# .. [1] `Riemannian Geometry for the classification of brain states with fNIRS
#    <https://www.biorxiv.org/content/10.1101/2024.09.06.611347v1>`_
#    T. Näher, L. Bastian, A. Vorreuther, P. Fries, R. Goebel, B. Sorger.
