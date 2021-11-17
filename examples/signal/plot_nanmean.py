"""
===============================================================================
Estimate covariance mean with NaN values
===============================================================================

Estimate the mean of covariance matrices corrupted by NaN values [1]_.
"""
# Author: Quentin Barthélemy, Sylvain Chevallier and Florian Yger
#
# License: BSD (3-clause)

import numpy as np
from matplotlib import pyplot as plt

from pyriemann.datasets import make_covariances
from pyriemann.utils.mean import mean_riemann, nanmean_riemann
from pyriemann.utils.distance import distance_riemann


###############################################################################
# Generate data
# -------------

rs = np.random.RandomState(42)
n_matrices, n_channels = 100, 10
covmats = make_covariances(n_matrices, n_channels, rs,
                           evals_mean=100., evals_std=20.)

# Compute true Riemannian mean
C_ref = mean_riemann(covmats)

# Corrupt data randomly
n_corrup_channels_max = n_channels // 2

all_n_corrup_channels, all_corrup_channels = np.zeros(n_matrices), []
for i in range(len(covmats)):
    n_corrupt_channels = rs.randint(n_corrup_channels_max, size=1)
    corrup_channels = rs.randint(n_channels, size=n_corrupt_channels)
    for chan in corrup_channels:
        covmats[i, chan] = np.nan
        covmats[i, :, chan] = np.nan
        all_corrup_channels.append(chan)
    all_n_corrup_channels[i] = n_corrupt_channels


fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set(title='Histogram of the number of corrupted channels',
       xlabel='Channel count')
plt.hist(all_n_corrup_channels, bins=np.arange(n_corrup_channels_max + 1) - .5)
plt.show()


###############################################################################


fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set(title='Histogram of the indices of corrupted channels',
       xlabel='Channel index')
plt.hist(all_corrup_channels, bins=np.arange(n_channels + 1) - .5)
plt.show()


###############################################################################
# Estimate covariance means
# -------------------------

# Euclidean NaN-mean
C_naneucl = np.nanmean(covmats, axis=0)

# Riemannian NaN-mean
C_nanriem = nanmean_riemann(covmats)

# Riemannian mean, after matrix deletion: average non-corrupted matrices
isnan = np.isnan(np.sum(covmats, axis=(1, 2)))
covmats_ = np.delete(covmats, np.where(isnan == True), axis=0)
perc = len(covmats_) / n_matrices * 100
print("Percentage of non-corrupted matrices: {:.2f} %".format(perc))
C_mdriem = mean_riemann(covmats_)


###############################################################################
# Compare covariance means
# ------------------------

d_naneucl = distance_riemann(C_ref, C_naneucl)
print("Euclidean NaN-mean, dist = {:.3f}".format(d_naneucl))

d_nanriem = distance_riemann(C_ref, C_nanriem)
print("Riemannian NaN-mean, dist = {:.3f}".format(d_nanriem))

d_mdriem = distance_riemann(C_ref, C_mdriem)
print("Riemannian mean, after matrix deletion, dist = {:.3f}".format(d_mdriem))


###############################################################################
# References
# ----------
# .. [1] F. Yger, S. Chevallier, Q. Barthélemy, S. Sra. "Geodesically-convex
#    optimization for averaging partially observed covariance matrices", ACML
#    2021.
