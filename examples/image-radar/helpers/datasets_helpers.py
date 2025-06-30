"""
=================================
Datasets Remote Sensing Helpers
=================================

This file contains helper functions for handling remote sensing datasets
"""

import os
from typing import Tuple, Dict
import urllib.request

from numpy.typing import ArrayLike
from scipy.io import loadmat


def download_salinas(data_path: str):
    """Download the Salinas dataset.

    Parameters
    ----------
    data_path : str
        Path to the data folder to download the data.
    """
    url_base = "https://zenodo.org/records/15771735/files/"
    urls = [
        url_base + "Salinas.mat?download=1",
        url_base + "Salinas_corrected.mat?download=1",
        url_base + "Salinas_gt.mat?download=1",
    ]
    filenames = [os.path.basename(url).split("?")[0] for url in urls]
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    if not all(
        [
            os.path.exists(os.path.join(data_path, filename))
            for filename in filenames
        ]
    ):
        print("Downloading Salinas dataset...")
        for url, filename in zip(urls, filenames):
            urllib.request.urlretrieve(url, os.path.join(data_path, filename))
        print("Done.")
    else:
        print("Salinas dataset already downloaded.")


def read_salinas(
    data_path: str, version: str = "corrected"
) -> Tuple[ArrayLike, ArrayLike, Dict[int, str]]:
    """Read Salinas hyperspectral data.

    Parameters
    ----------
    data_path : str
        Path to the data folder.
    version : str, default="corrected"
        Version of the data to read. Can be either "corrected" or "raw".

    Returns
    -------
    data : ArrayLike, shape (512, 217, 204)
        Data.
    labels : ArrayLike, shape (512, 217)
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


def download_uavsar(data_path: str, scene: int):
    """Download the UAVSAR dataset.

    Parameters
    ----------
    data_path : str
        Path to the data folder to download the data.
    scene : {1, 2}
        Scene to download.
    """
    assert scene in [1, 2], f"Unknown scene {scene} for UAVSAR dataset"
    if scene == 1:
        url = "https://zenodo.org/records/10625505/files/scene1.npy?download=1"
    else:
        url = "https://zenodo.org/records/10625505/files/scene2.npy?download=1"
    filename = f"scene{scene}.npy"
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    if not os.path.exists(os.path.join(data_path, filename)):
        print(f"Downloading UAVSAR dataset scene {scene}...")
        urllib.request.urlretrieve(url, os.path.join(data_path, filename))
        print("Download done.")
    else:
        print("UAVSAR dataset already downloaded.")
