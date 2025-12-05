


"""
data_loader.py
Automatically downloads the dataset from kagglehub and returns a local path.

This will use the `kagglehub` import you requested and download the dataset path. 
It returns a path that other modules can use.

Usage:
    from data_loader import download_dataset
    path = download_dataset("arshadrahmanziban/traffic-video-dataset")
"""

import os
import kagglehub
from pathlib import Path

def download_dataset(dataset_id: str, output_dir: str = "datasets") -> str:
    """
    Download dataset via kagglehub.dataset_download and return path to files.
    If files already exist, it will not re-download.
    """
    os.makedirs(output_dir, exist_ok=True)
    # This call is blocking — kagglehub handles download & extraction
    print(f"Downloading dataset {dataset_id} into {output_dir} ...")
    path = kagglehub.dataset_download(dataset_id, path=output_dir)
    print("Downloaded dataset to:", path)
    return str(path)

if __name__ == "__main__":
    # quick test when run directly
    print(download_dataset("arshadrahmanziban/traffic-video-dataset"))
