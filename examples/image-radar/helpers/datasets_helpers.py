"""
=================================
Datasets Remote Sensing Helpers
=================================

This file contains helper functions for handling remote sensing datasets
"""

import os
import time
from typing import Tuple, Dict, Optional
import urllib.request
import urllib.error

from numpy.typing import ArrayLike
from scipy.io import loadmat


def download_with_resume(
    url: str,
    filepath: str,
    chunk_size: int = 8192,
    max_retries: int = 5,
    backoff_factor: float = 1.0,
) -> bool:
    """Download a file with resume capability and retry logic.

    Parameters
    ----------
    url : str
        URL to download from.
    filepath : str
        Local path to save the file.
    chunk_size : int, default=8192
        Size of chunks to download at a time.
    max_retries : int, default=5
        Maximum number of retry attempts.
    backoff_factor : float, default=1.0
        Factor for exponential backoff between retries.

    Returns
    -------
    bool
        True if download successful, False otherwise.
    """

    def get_browser_headers() -> dict:
        """Get realistic browser headers to avoid bot detection."""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                AppleWebKit/537.36 (KHTML, like Gecko) \
                Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,\
                image/avif,image/webp,image/apng,*/*;q=0.8,\
                application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }

    def get_file_size(url: str) -> Optional[int]:
        """Get the total file size from URL."""
        try:
            req = urllib.request.Request(
                url, method="HEAD", headers=get_browser_headers()
            )
            with urllib.request.urlopen(req) as response:
                content_length = response.headers.get("Content-Length")
                return int(content_length) if content_length else None
        except urllib.error:
            return None

    def get_existing_size(filepath: str) -> int:
        """Get size of existing partial file."""
        return os.path.getsize(filepath) if os.path.exists(filepath) else 0

    # Get total file size
    total_size = get_file_size(url)
    existing_size = get_existing_size(filepath)

    # Check if file is already completely downloaded
    if total_size and existing_size == total_size:
        print(f"File {os.path.basename(filepath)} already fully downloaded.")
        return True

    for attempt in range(max_retries):
        try:
            # Prepare request with range header for resuming and browser header
            headers = get_browser_headers()
            mode = "wb"

            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"
                mode = "ab"  # Append to existing file
                print(f"Resuming download from byte {existing_size}")

            # Create request
            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req) as response:
                # Check if server supports range requests
                if existing_size > 0 and response.status not in (206, 416):
                    print("Server doesn't support resume, starting over...")
                    existing_size = 0
                    mode = "wb"
                    req = urllib.request.Request(
                        url, headers=get_browser_headers()
                    )  # Create new request with browser headers
                    response = urllib.request.urlopen(req)

                # If we get 416 Range Not Satisfiable,
                # file might already be complete
                if response.status == 416:
                    print(
                        f"File {os.path.basename(filepath)} "
                        "appears to be complete."
                    )
                    return True

                # Download in chunks
                downloaded = existing_size
                with open(filepath, mode) as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Show progress for large files
                        if total_size:
                            progress = (downloaded / total_size) * 100
                            print(
                                f"\rDownloading {os.path.basename(filepath)}: "
                                f"{progress:.1f}%",
                                end="",
                                flush=True,
                            )

                print(
                    f"\nSuccessfully downloaded {os.path.basename(filepath)}"
                )
                return True

        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            ConnectionError,
            OSError,
        ) as e:
            print(f"\nDownload attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                wait_time = backoff_factor * (2**attempt)
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                existing_size = get_existing_size(filepath)
            else:
                print(
                    f"Failed to download {os.path.basename(filepath)} after "
                    f"{max_retries} attempts"
                )
                return False

        except Exception as e:
            print(f"\nUnexpected error during download: {e}")
            return False

    return False


def download_salinas(
    data_path: str,
    chunk_size: int = 8192,
    max_retries: int = 5,
):
    """Download Salinas dataset.

    Download the Salinas dataset with robust error handling and resume
    capability.
    Uses browser headers to avoid bot detection and server restrictions.

    Parameters
    ----------
    data_path : str
        Path to the data folder to download the data.
    chunk_size : int, default=8192
        Size of chunks to download at a time (in bytes).
    max_retries : int, default=5
        Maximum number of retry attempts per file.
    """
    urls = [
        "https://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat",
        "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
        "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
    ]

    # Create data directory if it doesn't exist
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        print(f"Created directory: {data_path}")

    print("Starting Salinas dataset download...")

    success_count = 0
    for url in urls:
        filename = os.path.basename(url)
        filepath = os.path.join(data_path, filename)

        print(f"\nDownloading {filename}...")

        if download_with_resume(url, filepath, chunk_size, max_retries):
            success_count += 1
        else:
            print(f"Failed to download {filename}")

    if success_count == len(urls):
        print(f"\nSuccessfully downloaded all {len(urls)} files!")
    else:
        print(f"\nDownloaded {success_count}/{len(urls)} files successfully")

    return success_count == len(urls)


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
