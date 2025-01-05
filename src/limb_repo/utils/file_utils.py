"""File Utils."""

import os
import sys
from datetime import datetime

import h5py

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
        self, final_file_path: str, tmp_dir: str = "/tmp/dynamics_data/"
    ) -> None:
        self.datapoint_number = 0
        self.final_file_path = final_file_path

        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        self.temp_dir = os.path.join(tmp_dir, timestamp)

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        if len(os.listdir(self.temp_dir)) > 0:
            delete_tmp = input(
                "The temp directory is not empty. Delete all files? (y/n): "
            )
            if delete_tmp == "y":
                print(f"Deleting all files in {self.temp_dir}")
                for file in os.listdir(self.temp_dir):
                    os.remove(os.path.join(self.temp_dir, file))
            else:
                print("Exiting...")
                sys.exit(1)

    # def process_limb_repo_state(
    #     self,
    #     state: LimbRepoState,
    #     active_joint_min: JointState,
    #     active_joint_max: JointState,
    # ) -> np.ndarray:
    #     """Normalize and circularize ."""
    #     q_a_cos = np.cos(state.active_q)
    #     q_a_sin = np.sin(state.active_q)
    #     q_p_cos = np.cos(state.passive_q)
    #     q_p_sin = np.sin(state.passive_q)

    # def decode_limb_repo_state(self, state: np.ndarray) -> LimbRepoState:
    #     return

    def save_demo(
        self,
        initial_state: LimbRepoState,
        torque_action: Action,
        result_qdd: JointState,
    ) -> None:
        """Save each result as a separate hdf5."""

        path = os.path.join(self.temp_dir, f"{self.datapoint_number}.hdf5")
        with h5py.File(path, "w") as f:
            f.create_dataset("initial_state", data=initial_state)
            f.create_dataset("torque_action", data=torque_action)
            f.create_dataset("result_qdd", data=result_qdd)

        self.datapoint_number += 1

    def combine_temp_hdf5s(self) -> None:
        """Combine all temp hdf5s into one."""
