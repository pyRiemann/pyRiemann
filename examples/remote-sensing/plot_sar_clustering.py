"""
====================================================================
Clustering on SAR images with Riemannian geometry
====================================================================

This example compares clustering pipelines based on covariance matrices for
synthetic-aperture radar (SAR) image clustering [1]_ [2]_.
"""
# Author: Ammar Mian

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

from pyriemann.clustering import Kmeans
from pyriemann.estimation import Covariances
from helpers.datasets_helpers import download_uavsar
from helpers.processing_helpers import SlidingWindowVectorize


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
estimator = "scm"  # Chose any estimator from "scm", "lwf", "oas", "mcd", "hub"

###############################################################################
# Load data
# ---------

print(f"Loading data from scene {scene}.")
download_uavsar(data_path, scene)
data = np.load(os.path.join(data_path, f"scene{scene}.npy"))
data = data[:, :, :, date]  # Select one date only
data_visualization = data.copy()  # To avoid aliasing when showing data
n_components = data.shape[2]
n_clusters = 4
resolution_x = 1.6  # m, obtained from UAVSAR documentation
resolution_y = 0.6  # m, obtained from UAVSAR documentation

# For visualization of image
x_values = np.arange(0, data.shape[1]) * resolution_x
y_values = np.arange(0, data.shape[0]) * resolution_y
X_image, Y_image = np.meshgrid(x_values, y_values)

if small_dataset:
    reduce_factor_y = 14
    reduce_factor_x = 8
    data = data[::reduce_factor_y, ::reduce_factor_x]
    max_iter = 5
    resolution_x = reduce_factor_x*resolution_x
    resolution_y = reduce_factor_y*resolution_y
height, width, n_features = data.shape

# For visualization of results
x_values = np.arange(window_size//2, width-window_size//2) * resolution_x
y_values = np.arange(window_size//2, height-window_size//2) * resolution_y
X_res, Y_res = np.meshgrid(x_values, y_values)

print("Reading done.")

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
print(f"estimator = {estimator}")
print("-"*80)

###############################################################################
# Pipelines definition
# --------------------

# Logdet pipeline from [1]
pipeline_logdet = Pipeline([
    ("sliding_window", SlidingWindowVectorize(window_size=window_size)),
    ("covariances", Covariances(estimator=estimator)),
    ("kmeans", Kmeans(
        n_clusters=n_clusters,
        n_jobs=n_jobs,
        max_iter=max_iter,
        metric="logdet",
    ))
], verbose=True)

# Riemannian pipeline from [2]
pipeline_riemann = Pipeline([
    ("sliding_window", SlidingWindowVectorize(window_size=window_size)),
    ("covariances", Covariances(estimator=estimator)),
    ("kmeans", Kmeans(
        n_clusters=n_clusters,
        n_jobs=n_jobs,
        max_iter=max_iter,
        metric="riemann",
    ))
], verbose=True)

pipelines = [pipeline_logdet, pipeline_riemann]
pipelines_names = [f"{estimator} and logdet", f"{estimator} and Riemann"]

###############################################################################
# Perform clustering
# ------------------

print(f"\nStarting clustering with pipelines: {pipelines_names}")
results = {}
for pipeline_name, pipeline in zip(pipelines_names, pipelines):
    print("-"*60)
    print(f"Pipeline: {pipeline_name}")
    pipeline.fit(data)
    preds = pipeline.named_steps["kmeans"].labels_
    results[pipeline_name] = \
        pipeline.named_steps["sliding_window"].inverse_predict(preds)
    print("-"*60)
print("Done")

###############################################################################
# Plot data
# ---------

print("Plotting")
plot_value = 20*np.log10(np.sum(np.abs(data_visualization)**2, axis=2))
figure, ax = plt.subplots(figsize=(5, 5))
plt.pcolormesh(X_image, Y_image, plot_value, cmap="gray")
plt.colorbar()
ax.invert_yaxis()
plt.xlabel("Range (m)")
plt.ylabel("Azimuth (m)")
plt.title(r"SAR data: $20\log_{10}(x_{HH}^2 + x_{HV}^2 + x_{VV}^2$)")
plt.tight_layout()

###############################################################################
# Plot results
# ------------

for pipeline_name, labels_pred in results.items():
    figure, ax = plt.subplots(figsize=(5, 5))
    plt.pcolormesh(X_res, Y_res, labels_pred, cmap="tab20b")
    plt.xlim(X_image.min(), X_image.max())
    plt.ylim(Y_image.min(), Y_image.max())
    plt.title(f"Clustering with {pipeline_name}")
    plt.colorbar()
    ax.invert_yaxis()
    plt.xlabel("Range (m)")
    plt.ylabel("Azimuth (m)")
    plt.tight_layout()
plt.show()

###############################################################################
# References
# ----------
# .. [1] `Statistical classification for heterogeneous polarimetric SAR images
#    <https://hal.science/hal-00638829/>`_
#    Formont, P., Pascal, F., Vasile, G., Ovarlez, J. P., & Ferro-Famil, L.
#    IEEE Journal of selected topics in Signal Processing, 5(3), 567-576. 2010.
#
# .. [2] `On the use of matrix information geometry for polarimetric SAR image
#    classification
#    <https://hal.science/hal-02494996v1>`_
#    Formont, P., Ovarlez, J. P., & Pascal, F.
#    In Matrix Information Geometry (pp. 257-276). 2012.
