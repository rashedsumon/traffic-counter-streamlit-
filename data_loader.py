"""
data_loader.py
Downloads dataset from KaggleHub and returns path.
"""

import kagglehub

def download_dataset(dataset_id: str) -> str:
    """
    Downloads dataset from kagglehub and returns local cached path.
    KaggleHub chooses its own cache folder; you cannot override it.
    """
    print(f"Downloading dataset: {dataset_id}")
    path = kagglehub.dataset_download(dataset_id)
    print("Downloaded dataset to:", path)
    return str(path)


if __name__ == "__main__":
    print(download_dataset("arshadrahmanziban/traffic-video-dataset"))
