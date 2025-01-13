"""File Utils."""

import os
import sys
from datetime import datetime
from typing import List

import h5py
import numpy as np

from limb_repo.structs import Action, JointState, LimbRepoState


def get_root_path() -> str:
    """Get the root path of the repository."""
    return os.path.abspath(os.path.join(__file__, "../../../.."))


def to_abs_path(input_path: str) -> str:
    """Get the absolute path w.r.t.

    the limb-repo repository root.
    """
    if input_path[0] == "/":
        return input_path

    return os.path.abspath(os.path.join(get_root_path(), input_path))


class HDF5Saver:
    """Class to save data to hdf5."""

    def __init__(
        self, final_file_dir: str, tmp_dir: str = "/tmp/dynamics_data/"
    ) -> None:
        self.datapoint_number = 0
        self.final_file_dir = final_file_dir

        self.timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        self.tmp_dir = tmp_dir
        self.trial_tmp_dir = os.path.join(tmp_dir, self.timestamp)

        if not os.path.exists(self.trial_tmp_dir):
            os.makedirs(self.trial_tmp_dir)

        if len(os.listdir(self.trial_tmp_dir)) > 0:
            delete_tmp = input(
                "The temp directory is not empty. Delete all files? (y/n): "
            )
            if delete_tmp == "y":
                print(f"Deleting all files in {self.trial_tmp_dir}")
                for file in os.listdir(self.trial_tmp_dir):
                    os.remove(os.path.join(self.trial_tmp_dir, file))
            else:
                print("Exiting...")
                sys.exit(1)

    def save_demo(
        self,
        initial_state: LimbRepoState,
        torque_action: Action,
        result_qdd_a: JointState,
        result_qdd_p: JointState,
    ) -> None:
        """Save each result as a separate hdf5."""

        path = os.path.join(self.trial_tmp_dir, f"{self.datapoint_number}.hdf5")
        with h5py.File(path, "w") as f:
            f.create_dataset("initial_state", data=initial_state)
            f.create_dataset("torque_action", data=torque_action)
            f.create_dataset(
                "result_qdd", data=np.concatenate([result_qdd_a, result_qdd_p])
            )

        self.datapoint_number += 1

    def find_hdf5_files(self, directory):
        """Recursively find all .hdf5 files in a given directory."""
        hdf5_files = []
        for root, dirs, files in os.walk(directory):
            print(root, dirs, files)
            for file in files:
                if file.endswith(".hdf5"):
                    hdf5_files.append(os.path.join(root, file))
        return hdf5_files

    def combine_temp_hdf5s(self, data_dirs=List[str]) -> None:
        """Combine all temp hdf5s into one."""
        self.datapoint_number = 0

        # hdf5_files = self.find_hdf5_files(self.tmp_dir)
        files_per_dir = {}
        keys = []
        data_shapes = {}

        find_step_size = 10000

        for data_dir in data_dirs:
            i = 0
            while True:
                try:
                    with h5py.File(
                        os.path.join(self.tmp_dir, data_dir, f"{i}.hdf5"), "r"
                    ) as _:
                        i += find_step_size
                except FileNotFoundError:
                    i -= find_step_size
                    break

            for j in range(i, i + find_step_size):
                try:
                    with h5py.File(
                        os.path.join(self.tmp_dir, data_dir, f"{j}.hdf5"), "r"
                    ) as _:
                        pass
                except FileNotFoundError:
                    break

            files_per_dir[data_dir] = j
            print(files_per_dir)

        print(files_per_dir)
        num_files = sum(files_per_dir.values())
        print("num_files", num_files)

        final_file_path = os.path.join(
            self.final_file_dir, f"{num_files}_{self.timestamp}.hdf5"
        )

        with h5py.File(os.path.join(self.tmp_dir, data_dirs[0], "0.hdf5"), "r") as f:
            for key in f.keys():  # pylint: disable=consider-using-dict-items
                data_shapes[key] = f[key].shape
                keys.append(key)

        print(keys, data_shapes)

        with h5py.File(final_file_path, "w") as f:
            for key in keys:
                f.create_dataset(key, shape=(num_files, *data_shapes[key]))

            for data_dir in data_dirs:
                for i in range(files_per_dir[data_dir]):
                    print("               i", i)
                    print("datapoint number", self.datapoint_number)
                    file = os.path.join(self.tmp_dir, data_dir, f"{i}.hdf5")
                    with h5py.File(file, "r") as temp_f:
                        for key in keys:
                            f[key][self.datapoint_number] = temp_f[key]

                    self.datapoint_number += 1

        print(f"Saved to {final_file_path}")