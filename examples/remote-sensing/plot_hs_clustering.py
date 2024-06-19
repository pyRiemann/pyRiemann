"""
====================================================================
Clustering on hyperspectral with Riemannian geometry
====================================================================

This example compares clustering pipelines based on covariance matrices for
hyperspectral image clustering.
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

###############################################################################
# Load data
# ---------

# TODO: Handle all the other datasets: IndianPines, etc
print("Loading Salinas data")
download_salinas(data_path)
data, labels, labels_names = read_salinas(data_path)
n_clusters = len(labels_names)
if small_dataset:
    data = data[::4, ::4]
    max_iter = 15
print("Done")
height, width, n_features = data.shape

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
print("-"*80)

###############################################################################
# Pipelines definition
# --------------------

pipeline_euclidean = Pipeline([
    ("remove_mean", RemoveMeanImage()),
    ("pca", PCAImage(n_components=n_components)),
    ("sliding_window", SlidingWindowVectorize(window_size=window_size)),
    ("covariances", Covariances(estimator="scm")),
    ("kmeans", Kmeans(
        n_clusters=n_clusters,
        n_jobs=n_jobs,
        max_iter=max_iter,
        metric="euclid"))],
    verbose=True)

pipeline_riemann = Pipeline([
    ("remove_mean", RemoveMeanImage()),
    ("pca", PCAImage(n_components=n_components)),
    ("sliding_window", SlidingWindowVectorize(window_size=window_size)),
    ("covariances", Covariances(estimator="tyl")),
    ("kmeans", Kmeans(
        n_clusters=n_clusters,
        n_jobs=n_jobs,
        max_iter=max_iter,
        metric="riemann"))],
    verbose=True)

pipelines = [pipeline_euclidean, pipeline_riemann]
pipelines_names = ["Euclidean distance", "Affine-invariant distance"]

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
plot_value = np.mean(data, axis=2)
figure = plt.figure()
plt.imshow(plot_value, cmap="gray", aspect="auto")
plt.title("Data")

###############################################################################
# Plot results
# ------------

for pipeline_name, labels_pred in results.items():
    figure = plt.figure()
    plt.imshow(labels_pred, cmap="tab20b", aspect="auto")
    plt.title(f"Clustering results with {pipeline_name}")
    plt.colorbar()
plt.show()

###############################################################################
# References
# ----------
# .. [1] "Hyperspectral Image Clustering: Current achievements and future
#    lines"
#    H. Zhai, H. Zhang, P. Li and L. Zhang.
#    IEEE Geoscience and Remote Sensing Magazine, pp. 35-67. 2021
