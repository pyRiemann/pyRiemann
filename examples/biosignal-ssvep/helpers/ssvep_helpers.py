""" Local functions for SSVEP examples """

import os

from mne import create_info
from mne.datasets import fetch_dataset
from mne.io import RawArray
import numpy as np


SSVEP_HASHES = {
    "subject01_run1_raw.fif": "md5:e33a589abce842bf676dd881d9f4a9b0",
    "subject01_run2_raw.fif": "md5:2e75762abc30b832ad3d3d60f09accd4",
    "subject02_run1_raw.fif": "md5:f619d931dbb02b0fac32a77882d13388",
    "subject02_run2_raw.fif": "md5:dcf504b6a49fb10f4bf9094b6c765d6d",
    "subject03_run1_raw.fif": "md5:dae416ec161c3b46dd98d9d5f7cb7230",
    "subject03_run2_raw.fif": "md5:e7aed9d6ec5e03b284b368ff4a7c9862",
    "subject04_run1_raw.fif": "md5:6ac09c4d40f8cbb7b19aef38094d7e5a",
    "subject04_run2_raw.fif": "md5:216e2093c2533114e9f5aeb6687b56fd",
    "subject05_run1_raw.fif": "md5:18b44fafdfe582923d54d2fd06ed8538",
    "subject05_run2_raw.fif": "md5:632886d3ea209924b017a3e040f88e52",
    "subject06_run1_raw.fif": "md5:515e01550331d3782886f0d784b0b6dd",
    "subject06_run2_raw.fif": "md5:2c7db173e9d61d76d0ccfdd818f979fa",
    "subject06_run3_raw.fif": "md5:59df2e17c126d3bcb192ad6cd7378ff5",
    "subject07_run1_raw.fif": "md5:ef653c1c1f10eb3be909e2239a21c2df",
    "subject07_run2_raw.fif": "md5:d8baad6f61d5992f5716609f72f0b0b2",
    "subject07_run3_raw.fif": "md5:59df2e17c126d3bcb192ad6cd7378ff5",
    "subject08_run1_raw.fif": "md5:5d00b5fbc7979ee1f03348e0375fcf1e",
    "subject08_run2_raw.fif": "md5:5adefa32f17259c30f8be5e093df038f",
    "subject09_run1_raw.fif": "md5:1a067f96be7517daa180fb41eb75f593",
    "subject09_run2_raw.fif": "md5:58d7521ab9c70342de2eb740270eac59",
    "subject09_run3_raw.fif": "md5:3f77234f8bf24e7a4b53711161b72bae",
    "subject09_run4_raw.fif": "md5:f68c3173558c1321f380a9ff6145104e",
    "subject10_run1_raw.fif": "md5:6635bc4c7098751e6ee20e113b23e5ef",
    "subject10_run2_raw.fif": "md5:19b84659c8406a3840a3d90424578955",
    "subject10_run3_raw.fif": "md5:3f77234f8bf24e7a4b53711161b72bae",
    "subject10_run4_raw.fif": "md5:f68c3173558c1321f380a9ff6145104e",
    "subject11_run1_raw.fif": "md5:22248cdd15d98a7e3ddb9c6ec0cedf96",
    "subject11_run2_raw.fif": "md5:37f3021012f4ff01bedc03903bf60b66",
    "subject11_run3_raw.fif": "md5:92cf84dadb5e0f7ac833343637fd32b9",
    "subject11_run4_raw.fif": "md5:1d3e3f8cf2bb63f6815713f00f01b90c",
    "subject11_run5_raw.fif": "md5:19ee524adb69aa29b5ab04308bf5fbb6",
    "subject12_run1_raw.fif": "md5:3c40e01698b6edda06b696511155f4fc",
    "subject12_run2_raw.fif": "md5:ff7f0361a2d41f8df3fb53b9a9bc1220",
    "subject12_run3_raw.fif": "md5:92cf84dadb5e0f7ac833343637fd32b9",
    "subject12_run4_raw.fif": "md5:1d3e3f8cf2bb63f6815713f00f01b90c",
    "subject12_run5_raw.fif": "md5:19ee524adb69aa29b5ab04308bf5fbb6",
}


def download_data(subject=1, session=1):
    """Download data for SSVEP examples using MNE

    Parameters
    ----------
    subject : int, default=1
        Subject id
    session : int, default=1
        Session number

    Returns
    -------
    destination : str
        Path to downloaded data
    """
    DATASET_URL = "https://zenodo.org/record/2392979/files/"
    fname = f"subject{subject:02d}_run{session + 1}_raw.fif"
    url = f"{DATASET_URL}{fname}"
    dhash = SSVEP_HASHES[fname]
    archive_name = url.split("/")[-4:]
    dataset_params = {
        "dataset_name": "ssvep",
        "archive_name": "/".join(archive_name),
        "hash": dhash,
        "url": url,
        "folder_name": "MNE-ssvepexo-data",
        "config_key": "MNE_DATASETS_SSVEPEXO_PATH"
    }
    data_path = fetch_dataset(dataset_params, force_update=True)

    return os.path.join(data_path, *archive_name)


def bandpass_filter(raw, l_freq, h_freq, method="iir", verbose=False):
    """ Band-pass filter a signal using MNE """
    return raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method=method,
        verbose=verbose
    ).get_data()


def extend_signal(raw, frequencies, freq_band):
    """ Extend a signal with filter bank using MNE """
    raw_ext = np.vstack([
        bandpass_filter(raw, l_freq=f - freq_band, h_freq=f + freq_band)
        for f in frequencies
    ])

    info = create_info(
        ch_names=sum(
            list(map(lambda f: [
                ch + "-" + str(f) + "Hz"
                for ch in raw.ch_names
            ], frequencies)), []
        ),
        ch_types=["eeg"] * len(raw.ch_names) * len(frequencies),
        sfreq=int(raw.info["sfreq"])
    )

    return RawArray(raw_ext, info)
