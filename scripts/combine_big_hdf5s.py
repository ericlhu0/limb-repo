"""Combine hdf5s that each have many datapoints"""

import h5py
import numpy as np

# List of input HDF5 files to combine
input_files = [
    "/home/eric/lr-dir/limb-repo/_the_good_stuff/2500000_01-14_21-53-47.hdf5",
    "/home/eric/lr-dir/limb-repo/_the_good_stuff/2500000_01-14_21-54-17.hdf5",
    "/home/eric/lr-dir/limb-repo/_the_good_stuff/2500000_01-14_21-56-17.hdf5",
    "/home/eric/lr-dir/limb-repo/_the_good_stuff/2500000_01-14_21-58-13.hdf5",
]
output_file = "/home/eric/lr-dir/limb-repo/_the_good_stuff/combined.hdf5"


def combine_hdf5_files(in_files: list[str], out_files: str):
    """Combine multiple HDF5 files into one."""
    # Open the output file in write mode
    with h5py.File(out_files, "w") as combined_file:
        for _, file_name in enumerate(in_files):
            with h5py.File(file_name, "r") as h5_file:
                for key in h5_file.keys(): # pylint: disable=consider-using-dict-items
                    # Check if dataset or group already exists
                    if key not in combined_file:
                        # If it's the first file, create datasets/groups
                        h5_file.copy(key, combined_file)
                    else:
                        # Append data if it's a dataset
                        if isinstance(h5_file[key], h5py.Dataset):
                            existing_data = combined_file[key]
                            new_data = h5_file[key][:]
                            combined_data = np.concatenate(
                                (existing_data[:], new_data), axis=0
                            )
                            del combined_file[key]  # Delete old dataset
                            combined_file.create_dataset(key, data=combined_data)
                        else:
                            print(f"Skipping non-dataset object: {key}")
        print(f"Combined HDF5 file created: {out_files}")


if __name__ == "__main__":
    combine_hdf5_files(input_files, output_file)
