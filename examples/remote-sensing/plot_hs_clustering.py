"""
====================================================================
Clustering on hyperspectral with Riemannian geometry
====================================================================

This example compares clustering pipelines based on covariance matrices for
hyperspectral image clustering [1]_.
"""
# Author: Ammar Mian

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

from pyriemann.estimation import Covariances
from pyriemann.clustering import Kmeans
from helpers.datasets_helpers import download_salinas, read_salinas
from helpers.processing_helpers import (
    SlidingWindowVectorize,
    LabelsToImage,
    PCAImage,
    RemoveMeanImage,
)


###############################################################################
# Parameters
# ----------

window_size = 5
data_path = "./data"
n_jobs = -1
max_iter = 100
small_dataset = True  # Whole image can take time
n_components = 5  # For PCA dimensionality reduction
estimator = "scm"  # Chose any estimator from "scm", "lwf", "oas", "mcd", "hub"

###############################################################################
# Load data
# ---------

# TODO: Handle all the other datasets: IndianPines, etc
print("Loading Salinas data")
download_salinas(data_path)
data, labels, labels_names = read_salinas(data_path)
data_visualization = data.copy()  # To avoid aliasing when showing data
n_clusters = len(labels_names)

resolution = 3.7  # m, obtained from documentation of Salinas dataset

# For visualization of image
x_values = np.arange(0, data.shape[1]) * resolution
y_values = np.arange(0, data.shape[0]) * resolution
X_image, Y_image = np.meshgrid(x_values, y_values)

if small_dataset:
    reduce_factor = 4
    data = data[::reduce_factor, ::reduce_factor]
    max_iter = 10
    resolution = reduce_factor*resolution
height, width, n_features = data.shape

# For visualization of results
x_values = np.arange(window_size//2, width-window_size//2) * resolution
y_values = np.arange(window_size//2, height-window_size//2) * resolution
X_res, Y_res = np.meshgrid(x_values, y_values)

print("Reading done.")

###############################################################################
# Print configuration
# -------------------
print("-"*80)
print(f"Size of dataset: {data.shape}")
print(f"n_clusters = {n_clusters}")
print(f"window_size = {window_size}")
print(f"n_components = {n_components}")
print(f"n_jobs = {n_jobs}")
print(f"max_iter = {max_iter}")
print(f"estimator = {estimator}")
print("-"*80)

###############################################################################
# Pipelines definition
# --------------------

pipeline_euclid = Pipeline([
    ("remove_mean", RemoveMeanImage()),
    ("pca", PCAImage(n_components=n_components)),
    ("sliding_window", SlidingWindowVectorize(window_size=window_size)),
    ("covariances", Covariances(estimator=estimator)),
    ("kmeans", Kmeans(
        n_clusters=n_clusters,
        n_jobs=n_jobs,
        max_iter=max_iter,
        metric="euclid",
    ))
], verbose=True)

pipeline_riemann = Pipeline([
    ("remove_mean", RemoveMeanImage()),
    ("pca", PCAImage(n_components=n_components)),
    ("sliding_window", SlidingWindowVectorize(window_size=window_size)),
    ("covariances", Covariances(estimator=estimator)),
    ("kmeans", Kmeans(
        n_clusters=n_clusters,
        n_jobs=n_jobs,
        max_iter=max_iter,
        metric="riemann",
    ))
], verbose=True)

pipelines = [pipeline_euclid, pipeline_riemann]
pipelines_names = [
    f"{estimator} Euclidean distance",
    f"{estimator} Riemannian distance",
]

###############################################################################
# Perform clustering
# ------------------

print(f"\nStarting clustering with pipelines: {pipelines_names}")
results = {}
for pipeline_name, pipeline in zip(pipelines_names, pipelines):
    print("-"*60)
    print(f"Pipeline: {pipeline_name}")
    pipeline.fit(data)
    labels_pred = LabelsToImage(height, width, window_size).fit_transform(
        pipeline.named_steps["kmeans"].labels_
    )
    results[pipeline_name] = labels_pred
    print("-"*60)
print("Done")

###############################################################################
# Plot data
# ---------

print("Plotting")
plot_value = np.mean(data_visualization, axis=2)
figure, ax = plt.subplots(figsize=(7, 5))
plt.pcolormesh(X_image, Y_image, plot_value, cmap="gray")
plt.colorbar()
ax.invert_yaxis()
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Salinas dataset (mean over bands)")
plt.tight_layout()

###############################################################################
# Plot results
# ------------

for pipeline_name, labels_pred in results.items():
    figure, ax = plt.subplots(figsize=(7, 5))
    plt.pcolormesh(X_res, Y_res, labels_pred, cmap="tab20b")
    plt.xlim(X_image.min(), X_image.max())
    plt.ylim(Y_image.min(), Y_image.max())
    plt.title(f"Clustering with {pipeline_name}")
    plt.colorbar()
    ax.invert_yaxis()
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.tight_layout()
plt.show()

###############################################################################
# References
# ----------
# .. [1] "Hyperspectral Image Clustering: Current achievements and future
#    lines"
#    H. Zhai, H. Zhang, P. Li and L. Zhang.
#    IEEE Geoscience and Remote Sensing Magazine, pp. 35-67. 2021
