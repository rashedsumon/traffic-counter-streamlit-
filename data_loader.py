"""
data_loader.py
Automatically downloads the dataset from kagglehub and returns a local path.
"""

import kagglehub

def download_dataset(dataset_id: str) -> str:
    """
    Download dataset via kagglehub.dataset_download and return path to files.
    KaggleHub chooses its own cache directory.
    """
    print(f"Downloading dataset: {dataset_id}")
    path = kagglehub.dataset_download(dataset_id)
    print("Downloaded dataset to:", path)
    return str(path)


if __name__ == "__main__":
    print(download_dataset("arshadrahmanziban/traffic-video-dataset"))
