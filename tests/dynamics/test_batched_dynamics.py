"""Testing batched dynamics subclasses."""

import time
from typing import Callable

import numpy as np
import torch

from limb_repo.dynamics.models.batched.batched_learned_dynamics import (
    BatchedLearnedDynamics,
    NeuralNetworkConfig,
)
from limb_repo.dynamics.models.batched.batched_math_dynamics import BatchedMathDynamics
from limb_repo.dynamics.models.math_dynamics import MathDynamics
from limb_repo.environments.limb_repo_pybullet_env import (
    LimbRepoPyBulletConfig,
)
from limb_repo.structs import LimbRepoState
from limb_repo.utils import utils

parsed_config = utils.parse_config(
    "assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig
)
parsed_config.pybullet_config.use_gui = False

nn_config = utils.parse_config(
    "tests/dynamics/assets/test-30-30-12.yaml", NeuralNetworkConfig
)

min_features = torch.tensor(
    [
        -0.9855,
        -0.8491,
        -0.8514,
        -0.9355,
        -0.9976,
        -0.9874,
        -2.1485,
        -1.6155,
        -3.0084,
        -2.5424,
        0.1075,
        -2.7035,
        -0.8594,
        -0.9784,
        -0.8397,
        -0.9688,
        -0.8783,
        -0.7717,
        -2.9354,
        -3.0956,
        -2.1496,
        -2.4043,
        -3.0636,
        -2.8582,
        -2.5928,
        -2.4859,
        -1.6618,
        -3.2744,
        -1.9645,
        -3.1509,
    ]
)

max_features = torch.tensor(
    [
        0.9833,
        0.8924,
        0.8715,
        0.7282,
        0.9600,
        0.8429,
        2.8072,
        1.1658,
        -0.0891,
        2.7595,
        3.7065,
        2.7978,
        0.9209,
        0.9712,
        0.9610,
        0.7764,
        0.9979,
        0.9983,
        3.1069,
        2.8172,
        2.4196,
        1.9789,
        2.6599,
        2.0019,
        2.2319,
        2.0167,
        2.3994,
        2.8685,
        1.6744,
        2.3665,
    ]
)


def normalize_fn_lin(
    min_values: torch.Tensor, max_values: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function that normalizes input data between -1 and 1."""
    range_value = max_values - min_values

    def _normalize_fn_lin(x: torch.Tensor) -> torch.Tensor:
        return 2 * (x - min_values) / range_value - 1

    return _normalize_fn_lin


def batched_denormalize_fn_tanh(scaling: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function that denormalizes input data using tanh."""

    def _denormalize_fn_tanh(x: torch.Tensor) -> torch.Tensor:
        x[x > 1] = 0.9999
        x[x < -1] = -0.9999
        out = scaling * torch.arctanh(x)
        return out

    return _denormalize_fn_tanh


batch_size = 25

dynamics = BatchedLearnedDynamics(
    parsed_config,
    nn_config,
    "tests/dynamics/assets/test-weights.weights",
    normalize_fn_lin(min_features, max_features),
    batched_denormalize_fn_tanh(8),
    batch_size=batch_size,
)

math_dynamics = BatchedMathDynamics(parsed_config, batch_size=batch_size)


def test_batched_size():
    """Test the batched size."""
    # torques = torch.tensor(
    #     np.random.uniform(-1, 1, (dynamics.batch_size, dynamics.active_n_dofs))
    # )

    torques = torch.rand((dynamics.batch_size, dynamics.active_n_dofs))

    learned_time = 0
    math_time = 0

    for _ in range(500):
        time1 = time.time()
        resulting_state = dynamics.step_batched(torques)
        time2 = time.time()
        resulting_state_math = math_dynamics.step_batched(torques)
        time3 = time.time()
        print(f"Learned: {time2 - time1}, Math: {time3 - time2}")
        learned_time += time2 - time1
        math_time += time3 - time2

    print(f"average learned time: {learned_time / 500}")
    print(f"average math time: {math_time / 500}")

    assert resulting_state.shape == (
        dynamics.batch_size,
        2 * dynamics.active_n_dofs + 2 * dynamics.passive_n_dofs,
    )

    assert resulting_state_math.shape == (
        dynamics.batch_size,
        2 * dynamics.active_n_dofs + 2 * dynamics.passive_n_dofs,
    )


def test_batched_vs_nonbatched():
    """Test that each batch element matches the non-batched model for 5
    steps."""
    # Parse configuration.

    # Instantiate the models.
    nonbatched_model = MathDynamics(parsed_config)
    batched_model = BatchedMathDynamics(parsed_config, batch_size=batch_size)

    # Get the common initial state.
    init_state = nonbatched_model.get_state()
    batched_init = torch.tensor(
        np.tile(init_state, (batch_size, 1)), dtype=torch.float32
    )
    batched_model.set_state_batched(batched_init)
    nonbatched_model.set_state(init_state)

    active_n_dofs = len(init_state.active_q)
    steps = 5

    # Generate a sequence of torques for 5 steps.
    # Shape: [steps, batch_size, active_n_dofs]
    torque_sequence = torch.rand((steps, batch_size, active_n_dofs))

    # Simulate batched model for 5 steps
    for step in range(steps):
        # For each step, apply the torques for all batch elements.
        batched_model.step_batched(torque_sequence[step])
    # Get the final batched state.
    final_batched_state = batched_model.get_state_batched()

    # For each batch element, simulate non-batched model with its own torque sequence
    for i in range(batch_size):
        # Reset non-batched model to the common initial state.
        nonbatched_model.set_state(LimbRepoState(init_state))
        for step in range(steps):
            torque_i = torque_sequence[step, i].numpy()
            nonbatched_model.step(torque_i)
        # Get the final state from the non-batched model.
        nonbatched_final_state = torch.tensor(
            nonbatched_model.get_state(), dtype=torch.float32
        )
        # Compare to the corresponding batch element.
        if not torch.allclose(
            final_batched_state[i], nonbatched_final_state, atol=1e-3
        ):
            raise AssertionError(
                f"Mismatch for batch index {i} after {steps} steps:\n"
                f"Batched state: {final_batched_state[i]}\n"
                f"Non-batched state: {nonbatched_final_state}"
            )
    print("All batch elements match the non-batched outputs after 5 steps.")
