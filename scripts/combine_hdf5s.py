"""Combine hdf5s that each have one datapoint."""

from limb_repo.utils import file_utils

if __name__ == "__main__":
    hdf5_saver = file_utils.HDF5Saver(
        file_utils.to_abs_path("_the_good_stuff/"),
        file_utils.to_abs_path("_out/temp/"),
    )

    hdf5_saver.combine_temp_hdf5s(
        data_dirs=["01-14_14-30-09", "01-14_14-30-27", "01-14_14-31-20"]
    )
