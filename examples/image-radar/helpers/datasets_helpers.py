"""
=================================
Datasets Remote Sensing Helpers
=================================

This file contains helper functions for handling remote sensing datasets
"""

import logging
import os
from urllib.request import urlretrieve

from scipy.io import loadmat


logging.basicConfig(level=logging.INFO, force=True)


def download_salinas(data_path):
    """Download the Salinas dataset.

    Parameters
    ----------
    data_path : str
        Path to the destination folder for data download.
    """
    src_base = "https://zenodo.org/records/15771735/files/"
    srcs = [
        src_base + "Salinas.mat?download=1",
        src_base + "Salinas_corrected.mat?download=1",
        src_base + "Salinas_gt.mat?download=1",
    ]
    filenames = [os.path.basename(src).split("?")[0] for src in srcs]

    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    for src, filename in zip(srcs, filenames):
        dst = os.path.join(data_path, filename)
        if not os.path.exists(dst):
            logging.info(f"Downloading file {filename} from {src} to {dst}.")
            urlretrieve(src, dst)


def read_salinas(data_path, version="corrected"):
    """Read Salinas hyperspectral data.

    Parameters
    ----------
    data_path : str
        Path to the folder for data reading.
    version : {"corrected", "raw"}, default="corrected"
        Version of the data to read.

    Returns
    -------
    data : array-like, shape (512, 217, 204)
        Data.
    labels : array-like, shape (512, 217)
        Labels.
    labels_names : dict[int, str]
        Dictionary mapping labels to their names.
    """
    if version == "corrected":
        data_file = os.path.join(data_path, "Salinas_corrected.mat")
    else:
        data_file = os.path.join(data_path, "Salinas.mat")
    data = loadmat(data_file)["salinas_corrected"]
    labels = loadmat(os.path.join(data_path, "Salinas_gt.mat"))["salinas_gt"]
    labels_names = {
        0: "Undefined",
        1: "Brocoli_green_weeds_1",
        2: "Brocoli_green_weeds_2",
        3: "Fallow",
        4: "Fallow_rough_plow",
        5: "Fallow_smooth",
        6: "Stubble",
        7: "Celery",
        8: "Grapes_untrained",
        9: "Soil_vinyard_develop",
        10: "Corn_senesced_green_weeds",
        11: "Lettuce_romaine_4wk",
        12: "Lettuce_romaine_5wk",
        13: "Lettuce_romaine_6wk",
        14: "Lettuce_romaine_7wk",
        15: "Vinyard_untrained",
        16: "Vinyard_vertical_trellis",
    }
    return data, labels, labels_names


def download_uavsar(data_path, scene):
    """Download the UAVSAR dataset.

    Parameters
    ----------
    data_path : str
        Path to the destination folder for data download.
    scene : {1, 2}
        Scene index to download.
    """
    assert scene in [1, 2], f"Unknown scene {scene} for UAVSAR dataset"
    filename = f"scene{scene}.npy"
    src = f"https://zenodo.org/records/10625505/files/{filename}?download=1"

    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    dst = os.path.join(data_path, filename)
    if not os.path.exists(dst):
        logging.info(f"Downloading file {filename} from {src} to {dst}.")
        urlretrieve(src, dst)
