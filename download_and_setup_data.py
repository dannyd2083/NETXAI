# download_and_setup_data.py
# This script downloads and sets up both the CTU-13 raw dataset and the CICFlowMeter-based preprocessed dataset from GitHub.

import os
import requests
import tarfile
import subprocess
from tqdm import tqdm

# Create the datasets directory if it doesn't exist
os.makedirs("datasets", exist_ok=True)

# Download a file from a given URL if it does not already exist
def download_file(url, output_path, chunk_size=1024):
    if os.path.exists(output_path):
        print("File already exists at {}, skipping download.".format(output_path))
        return
    print("Downloading from {}...".format(url))

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(output_path))

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))
    progress_bar.close()
    print("Downloaded to {}".format(output_path))

# Extract a .tar.gz archive if the target directory is empty
def extract_tar_bz2(tar_path, extract_to):
    if os.path.exists(extract_to) and os.listdir(extract_to):
        print("{} already exists and is not empty, skipping extraction.".format(extract_to))
        return
    print("Extracting {} to {}...".format(tar_path, extract_to))
    try:
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(path=extract_to)
    except:
        return
    print("Extraction complete.")


def main():

    # === Step 1: Download and extract CTU-13 raw dataset ===
    ctu13_url = "https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2"
    ctu13_tar_path = "datasets/ctu13.tar.bz2"
    ctu13_extract_path = "datasets/ctu13"
    if os.path.exists(ctu13_extract_path):
        print("File already exists at {}, skipping download and extract.".format(ctu13_extract_path))
    else:
        download_file(ctu13_url, ctu13_tar_path)
        extract_tar_bz2(ctu13_tar_path, ctu13_extract_path)

    # === Step 2: Clone CICFlowMeter-based feature dataset from GitHub ===
    github_repo_url = "https://github.com/imfaisalmalik/CTU13-CSV-Dataset"
    github_clone_path = "datasets/CTU13-CSV-Dataset"

    if not os.path.exists(github_clone_path):
        print("Cloning {} ...".format(github_repo_url))
        subprocess.run(["git", "clone", github_repo_url, github_clone_path])
        print("Cloning complete.")
    else:
        print("{} already exists, skipping clone.".format(github_clone_path))
