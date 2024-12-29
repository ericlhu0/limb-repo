"""PyBullet environment for Limb Repositioning."""

from dataclasses import dataclass

import numpy as np
import omegaconf
import pybullet
import pybullet_utils.bullet_client as bc

from limb_repo.environments.base_env import BaseEnv


@dataclass
class PyBulletConfig:
    """Configuration for a PyBullet environment."""

    use_gui: bool
    real_time_simulation: bool
    gravity: np.ndarray
    dt: float


class PyBulletEnv(BaseEnv):
    """Pybullet environment for Limb Repositioning."""

    def __init__(self, config: omegaconf.DictConfig):
        # super().__init__()
        # Create pybullet sim
        if config.use_gui:
            self.p = bc.BulletClient(
                connection_mode=pybullet.GUI, options="--width=1000 --height=1000"
            )
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self.p.setGravity(*config.gravity)
        self.p.setRealTimeSimulation(1 if config.real_time_simulation else 0)
        self.p.setTimeStep(config.dt)
        self.p.setPhysicsEngineParameter(
            constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,
            globalCFM=0.000001,
        )

        self.dt = config.dt
