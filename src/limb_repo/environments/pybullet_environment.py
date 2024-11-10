from dataclasses import dataclass

from limb_repo.environments.base_environment import BaseEnvironment, Config
from limb_repo.structs import Pose, State

@dataclass
class PyBulletConfig(Config):
    """The configuration of the PyBullet environment."""
    active_pose: Pose
    active_config: State
    active_urdf: str
    passive_pose: Pose
    passive_config: State
    passive_urdf: str
    wheelchair_pose: Pose
    wheelchair_config: State
    wheelchair_urdf: str

class PyBulletEnvironment(BaseEnvironment):
    """Pybullet environment for Limb Repositioning."""

    def __init__(self):
        #super().__init__()
        pass

    def parse_config(self) -> Config:
        """Parse the configuration file."""
        return Config()