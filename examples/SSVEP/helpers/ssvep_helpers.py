""" Local functions for SSVEP examples """

import os
import numpy as np

from mne import create_info
from mne.datasets import fetch_dataset
from mne.io import RawArray


def download_data(subject=1, session=1):
    """Download data for SSVEP examples using MNE

    Parameters
    ----------
    subject : int, default 1
        Subject id
    session : int, default 1
        Session number

    Returns
    -------
    destination : str
        Path to downloaded data
    """
    DATASET_URL = 'https://zenodo.org/record/2392979/files/'
    url = '{:s}subject{:02d}_run{:d}_raw.fif'.format(
        DATASET_URL, subject, session + 1)

    archive_name = url.split('/')[-4:]
    dataset_params = {
        'dataset_name': "ssvep",
        'archive_name': archive_name,
        'hash': 'md5:ff7f0361a2d41f8df3fb53b9a9bc1220',
        'url': url,
        'folder_name': 'MNE-ssvepexo-data',
        'config_key': 'MNE_DATASETS_SSVEPEXO_PATH'
    }
    data_path = fetch_dataset(dataset_params)

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
        for f in frequencies]
    )

    info = create_info(
        ch_names=sum(
            list(map(lambda f: [ch + '-' + str(f) + 'Hz'
                                for ch in raw.ch_names],
                     frequencies)), []),
        ch_types=['eeg'] * len(raw.ch_names) * len(frequencies),
        sfreq=int(raw.info['sfreq'])
    )

    return RawArray(raw_ext, info)
