"""Check if end-effector pos, vel and orn of active and passive match."""

import numpy as np

from limb_repo.structs import LREEState


def check_ee_kinematics(
    lr_ee_state: LREEState,
    active_ee_to_passive_ee: np.ndarray,
    debug=False,
):
    """Check if end-effector pos, vel and orn of active and passive match."""
    active_ee_pos = lr_ee_state.active_ee_pos
    active_ee_vel = lr_ee_state.active_ee_vel
    active_ee_orn = lr_ee_state.active_ee_orn
    passive_ee_pos = lr_ee_state.passive_ee_pos
    passive_ee_vel = lr_ee_state.passive_ee_vel
    passive_ee_orn = lr_ee_state.passive_ee_orn

    position_check = np.allclose(active_ee_pos, passive_ee_pos, atol=0.01)
    velocity_check = np.allclose(active_ee_vel, passive_ee_vel, atol=0.01)
    orientaion_check = np.allclose(
        (np.linalg.inv(active_ee_orn) @ passive_ee_orn),
        active_ee_to_passive_ee,
        atol=0.01,
    )

    if debug:
        if not position_check:
            print("fail position")
            print(f"robot ee pos: {active_ee_pos}\nhuman ee pos: {passive_ee_pos}")
        if not velocity_check:
            print("fail velocity")
            print(f"robot ee vel: {active_ee_vel}\nhuman ee vel: {passive_ee_vel}")
        if not orientaion_check:
            print("fail orientation")
            print(
                f"grasp: {np.linalg.inv(active_ee_orn) @ passive_ee_orn}\n\
                    target orn: {active_ee_to_passive_ee}"
            )

    # if (not position_check) or (not orientaion_check) or not velocity_check:
    # input('break')
    # if (not position_check) or (not orientaion_check) or not velocity_check:
    # input('break')

    return position_check and velocity_check and orientaion_check
