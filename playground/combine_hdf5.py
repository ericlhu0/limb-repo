"""Try combining hdf5s."""

import os
import sys
from datetime import datetime
from typing import List

import h5py
import numpy as np

from limb_repo.structs import Action, LimbRepoState

# Parameters
INITIAL_SIZE = 100  # Initial size for datasets (adjust as needed)
GROWTH_FACTOR = 2  # Factor to grow the dataset size when needed

temp_dir = "/Users/eric/Documents/lr-dir/limb-repo/playground/out/temp"
final_file_path = "/Users/eric/Documents/lr-dir/limb-repo/playground/out/final.hdf5"

import os


def find_all_files_recursively(directory):
    """Find all files recursively in a given directory."""
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


# Sort files based on numeric prefix
all_files = find_all_files_recursively(temp_dir)

print(all_files)
