"""Try combining hdf5s."""

import os
from typing import List

import h5py

# Parameters
INITIAL_SIZE = 100  # Initial size for datasets (adjust as needed)
GROWTH_FACTOR = 2  # Factor to grow the dataset size when needed

temp_dir = (
    "/Users/eric/Documents/lr-dir/limb-repo/playground/out/temp/01-05_13-04-52/0.hdf5"
)
final_file_path = "/Users/eric/Documents/lr-dir/limb-repo/playground/out/final.hdf5"


with h5py.File(temp_dir, "r") as f:
    print(f.keys())
    print(f["initial_state"].shape)
    print(f["torque_action"].shape[0])
    print(f["result_qdd"].shape[0])


def find_all_files_recursively(directory) -> List[str]:
    """Find all files recursively in a given directory."""
    all_files = []
    for root, dir, files in os.walk(directory):
        print(root, dir, files)
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


print(find_all_files_recursively(temp_dir))
