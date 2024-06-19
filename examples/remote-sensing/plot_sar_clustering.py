"""
====================================================================
Clustering on SAR images with Riemannian geometry
====================================================================

This example compares clustering pipelines based on covariance matrices for
SAR image clustering [1]_ [2]_.
"""
# Author: Ammar Mian

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

from pyriemann.estimation import Covariances
from pyriemann.clustering import Kmeans
from helpers.datasets_helpers import download_uavsar
from helpers.processing_helpers import SlidingWindowVectorize, LabelsToImage


###############################################################################
# Parameters
# ----------

window_size = 7
data_path = "./data"
scene = 1  # Chose between 1 or 2
date = 0
n_jobs = -1
max_iter = 100
small_dataset = True  # The full dataset will take a very long time

###############################################################################
# Load data
# ---------

print(f"Loading data from scene {scene}.")
download_uavsar(data_path, scene)
data = np.load(os.path.join(data_path, f"scene{scene}.npy"))
data = data[:, :, :, date]  # Select one date only
n_components = data.shape[2]
n_clusters = 4
if small_dataset:
    data = data[::7, ::7]
    max_iter = 15
print("Done.")
height, width, n_features = data.shape

###############################################################################
# Print configuration
# -------------------
print("-"*80)
print(f"Size of dataset: {data.shape}")
print(f"date = {date}")
print(f"window_size = {window_size}")
print(f"n_clusters = {n_clusters}")
print(f"n_jobs = {n_jobs}")
print(f"max_iter = {max_iter}")
print("-"*80)

###############################################################################
# Pipelines definition
# --------------------

# Logdet pipelines from [1]
pipeline_scm_logdet = Pipeline([
    ("sliding_window", SlidingWindowVectorize(window_size=window_size)),
    ("covariances", Covariances(estimator="scm")),
    ("kmeans", Kmeans(
        n_clusters=n_clusters,
        n_jobs=n_jobs,
        max_iter=max_iter,
        metric="logdet"))],
    verbose=True)

pipeline_tyler_logdet = Pipeline([
    ("sliding_window", SlidingWindowVectorize(window_size=window_size)),
    ("covariances", Covariances(estimator="tyl")),
    ("kmeans", Kmeans(
        n_clusters=n_clusters,
        n_jobs=n_jobs,
        max_iter=max_iter,
        metric="logdet"))],
    verbose=True)

# Riemannian pipelines from [2]
pipeline_scm_riemann = Pipeline([
    ('sliding_window', SlidingWindowVectorize(window_size=window_size)),
    ('covariances', Covariances(estimator="scm")),
    ('kmeans', Kmeans(
        n_clusters=n_clusters,
        n_jobs=n_jobs,
        max_iter=max_iter,
        metric="riemann"))],
    verbose=True)

pipeline_tyler_riemann = Pipeline([
    ("sliding_window", SlidingWindowVectorize(window_size=window_size)),
    ("covariances", Covariances(estimator="tyl")),
    ("kmeans", Kmeans(
        n_clusters=n_clusters,
        n_jobs=n_jobs,
        max_iter=max_iter,
        metric="riemann"))],
    verbose=True)

pipelines = [
    pipeline_scm_logdet,
    pipeline_tyler_logdet,
    pipeline_scm_riemann,
    pipeline_tyler_riemann
]
pipelines_names = [
    "SCM logdet",
    "Tyler logdet",
    "SCM riemann",
    "Tyler riemann",
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
plot_value = 20*np.log10(np.sum(np.abs(data)**2, axis=2))
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
# .. [1] `Statistical classification for heterogeneous polarimetric SAR
#    images
#    <https://hal.science/hal-00638829/>`_
#    Formont, P., Pascal, F., Vasile, G., Ovarlez, J. P., & Ferro-Famil, L.
#    IEEE Journal of selected topics in Signal Processing, 5(3), 567-576. 2010.
#
# .. [2] `On the use of matrix information geometry for polarimetric SAR image
#    classification
#    <https://hal.science/hal-02494996v1>`_
#    Formont, P., Ovarlez, J. P., & Pascal, F.
#    In Matrix Information Geometry (pp. 257-276). 2012.
