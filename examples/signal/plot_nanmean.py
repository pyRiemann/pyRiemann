"""
===============================================================================
Estimate mean of SPD matrices with NaN values
===============================================================================

Estimate the mean of SPD matrices corrupted by NaN values [1]_.
"""
# Author: Quentin Barthélemy, Sylvain Chevallier and Florian Yger
#
# License: BSD (3-clause)

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from pyriemann.datasets import make_covariances
from pyriemann.utils.mean import mean_riemann, nanmean_riemann
from pyriemann.utils.distance import distance_riemann


###############################################################################


def corrupt(covmats, n_corrup_channels_max, rs):
    n_matrices, n_channels, _ = covmats.shape
    all_n_corrup_channels, all_corrup_channels = np.zeros(n_matrices), []
    for i_matrix in range(n_matrices):
        n_corrupt_channels = rs.randint(n_corrup_channels_max + 1, size=1)
        corrup_channels = rs.choice(
            np.arange(0, n_channels), size=n_corrupt_channels, replace=False)
        for i_channel in corrup_channels:
            covmats[i_matrix, i_channel] = np.nan
            covmats[i_matrix, :, i_channel] = np.nan
            all_corrup_channels.append(i_channel)
        all_n_corrup_channels[i_matrix] = n_corrupt_channels
    return covmats, all_n_corrup_channels, all_corrup_channels


###############################################################################
# Generate data
# -------------

rs = np.random.RandomState(42)
n_matrices, n_channels = 100, 10
covmats = make_covariances(
    n_matrices, n_channels, rs, evals_mean=100., evals_std=20.)

# Compute the reference, the Riemannian mean on all SPD matrices
C_ref = mean_riemann(covmats)

# Corrupt data randomly
n_corrup_channels_max = n_channels // 2
print("Maximum number of corrupted channels: {} over {}".format(
    n_corrup_channels_max, n_channels))

covmats, all_n_corrup_channels, all_corrup_channels = corrupt(
    covmats, n_corrup_channels_max, rs)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set(title='Histogram of the number of corrupted channels',
       xlabel='Channel count', ylabel='Occurrences')
plt.hist(all_n_corrup_channels, bins=np.arange(n_corrup_channels_max + 2) - .5)
plt.show()


###############################################################################


fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set(title='Histogram of the indices of corrupted channels',
       xlabel='Channel index', ylabel='Occurrences')
plt.hist(all_corrup_channels, bins=np.arange(n_channels + 1) - .5)
plt.show()


###############################################################################
# Estimate means of SPD matrices
# ------------------------------
#
# Riemannian mean could only be computed on full-rank matrices. A common
# strategy is called matrix deletion, that is removing all matrices with
# corrupted channels before computing mean.
# This results in discarding useful information as uncorrupted channels are
# removed from the computation of the mean.
# Nan-mean uses as much information as possible to estimate the mean [1]_.

# Euclidean NaN-mean
C_naneucl = np.nanmean(covmats, axis=0)

# Riemannian NaN-mean
C_nanriem = nanmean_riemann(covmats)

# Riemannian mean, after matrix deletion: average only uncorrupted matrices
isnan = np.isnan(np.sum(covmats, axis=(1, 2)))
covmats_ = np.delete(covmats, np.where(isnan), axis=0)
perc = len(covmats_) / n_matrices * 100
print("Percentage of uncorrupted matrices: {:.2f} %".format(perc))
C_mdriem = mean_riemann(covmats_)


###############################################################################
# Compare means
# -------------
#
# Compare distances between the different means and the reference.

d_naneucl = distance_riemann(C_ref, C_naneucl)
print(f"Euclidean NaN-mean, distance to ref = {d_naneucl:.3f}")

d_nanriem = distance_riemann(C_ref, C_nanriem)
print(f"Riemannian NaN-mean, distance to ref = {d_nanriem:.3f}")

d_mdriem = distance_riemann(C_ref, C_mdriem)
print(f"Riemannian mean after deletion, distance to ref = {d_mdriem:.3f}")

# Riemannian NaN-mean gives the best result, and Riemannian mean after matrix
# deletion is worst than Euclidean NaN-mean.


###############################################################################
# Evaluate influence of corrupted channels
# ----------------------------------------
#
# Repeat the previous experiment, varying the maximum number of corrupted
# channels [1]_.

covmats_orig = make_covariances(
    n_matrices, n_channels, rs, evals_mean=100., evals_std=20.)
C_ref = mean_riemann(covmats_orig)

df = []
for n_corrup_channels_max in range(0, n_channels // 2 + 1):
    covmats = np.copy(covmats_orig)
    covmats, _, _ = corrupt(covmats, n_corrup_channels_max, rs)

    C_naneucl = np.nanmean(covmats, axis=0)
    C_nanriem = nanmean_riemann(covmats)
    isnan = np.isnan(np.sum(covmats, axis=(1, 2)))
    covmats_ = np.delete(covmats, np.where(isnan), axis=0)
    C_mdriem = mean_riemann(covmats_)

    res_naneucl = {'n_corrupt': n_corrup_channels_max,
                   'dist': distance_riemann(C_ref, C_naneucl),
                   'Means': 'Euclidean NaN-mean'}
    res_nanriem = {'n_corrupt': n_corrup_channels_max,
                   'dist': distance_riemann(C_ref, C_nanriem),
                   'Means': 'Riemannian NaN-mean'}
    res_mdriem = {'n_corrupt': n_corrup_channels_max,
                  'dist': distance_riemann(C_ref, C_mdriem),
                  'Means': 'Riemannian mean after deletion'}
    df += [res_naneucl, res_nanriem, res_mdriem]
df = pd.DataFrame(df)

g = sns.catplot(data=df, x="n_corrupt", y="dist", hue="Means", kind="point",
                legend_out=False)
g.set(title="Influence of corrupted channels")
g.set_axis_labels("Maximum number of corrupted channels",
                  "Distance to reference")
plt.tight_layout()
plt.show()


###############################################################################
# References
# ----------
# .. [1] `Geodesically-convex optimization for averaging partially observed
#    covariance matrices
#    <https://hal.archives-ouvertes.fr/hal-02984423>`_
#    F. Yger, S. Chevallier, Q. Barthélemy, and S. Sra. Asian Conference on
#    Machine Learning (ACML), Nov 2020, Bangkok, Thailand. pp.417 - 432.
