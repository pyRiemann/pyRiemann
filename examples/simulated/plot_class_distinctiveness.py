"""
=====================================================================
Select best class-separated SPD matrix dataset on a manifold
=====================================================================
Selecting SPD matrix dataset with better separation
between two classes on the manifold.
This could be used as the metric for selecting the best frequency band
or time window among multiple choices.
"""
# Authors: Maria Sayu Yamamoto <maria-sayu.yamamoto@universite-paris-saclay.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

from pyriemann.embedding import SpectralEmbedding
from pyriemann.datasets import make_gaussian_blobs
from pyriemann.preprocessing import class_distinctiveness


print(__doc__)

###############################################################################
# Set parameters for sampling from the Riemannian Gaussian distribution

n_matrices = 100  # how many SPD matrices to generate
n_dim = 2  # number of dimensions of the SPD matrices
random_state = 42  # ensure reproducibility
epsilons_array = [.6, .6, .7]  # parameter for the distance between centers
sigmas_array = [.2, .7, .2]  # dispersion of the Gaussian distribution

###############################################################################
# Generate the samples on three different class separability conditions

# data0...small distance between class centroids
#         and small dispersion within class
data0_X, data0_y = make_gaussian_blobs(n_matrices=n_matrices, n_dim=n_dim,
                                       class_sep=epsilons_array[0],
                                       class_disp=sigmas_array[0],
                                       random_state=random_state, n_jobs=4)

# data1...small distance between class centroids
#         and large dispersion within class
data1_X, data1_y = make_gaussian_blobs(n_matrices=n_matrices, n_dim=n_dim,
                                       class_sep=epsilons_array[1],
                                       class_disp=sigmas_array[1],
                                       random_state=random_state, n_jobs=4)

# data2...large distance between class centroids
#         and small dispersion within class
data2_X, data2_y = make_gaussian_blobs(n_matrices=n_matrices, n_dim=n_dim,
                                       class_sep=epsilons_array[2],
                                       class_disp=sigmas_array[2],
                                       random_state=random_state, n_jobs=4)

datasets = [data0_X, data1_X, data2_X]
labels = [data0_y, data1_y, data2_y]

###############################################################################
# Apply classDis for each dataset

all_classDis = []
for data_ind in range(len(datasets)):
    classDis = class_distinctiveness(datasets[data_ind],
                                     labels[data_ind], nume_denomi=False)
    all_classDis.append(format(classDis, '.4f'))

###############################################################################
# Select the dataset with the highest classDis value

max_classDis_ind = np.argmax(all_classDis)
print('Best class-separated dataset is  dataset' + str(max_classDis_ind))

###############################################################################
# Plot sample distribution of each dataset

fig, ax = plt.subplots(figsize=(13.5, 4.4), ncols=3, sharey=True)
plt.subplots_adjust(wspace=0.10)
steps = [0, 1, 2]
titles = [r'dataset0($\varepsilon = 0.60, \sigma = 0.20$)',
          r'dataset1($\varepsilon = 0.60, \sigma = 0.70$)',
          r'dataset2($\varepsilon = 0.70, \sigma = 0.20$)']
for axi, step, title in zip(ax, steps, titles):
    X = datasets[step]
    y = labels[step]

    # Embed samples in 2D
    emb = SpectralEmbedding(n_components=2, metric='riemann')
    embedded_points = emb.fit_transform(X)

    # Plot the embedded points on plain
    axi.scatter(
        embedded_points[y == 0][:, 0],
        embedded_points[y == 0][:, 1],
        c='C0', s=50, alpha=0.50, label="class 0")
    axi.scatter(
        embedded_points[y == 1][:, 0],
        embedded_points[y == 1][:, 1],
        c='C1', s=50, alpha=0.50, label="class 1")
    axi.set_title(title, fontsize=14)
    axi.legend(loc="upper right")

ax[max_classDis_ind].set_title('best class-separated dataset\n'
                               + titles[max_classDis_ind], fontsize=14)
plt.show()
