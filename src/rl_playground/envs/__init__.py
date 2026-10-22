"""Custom Gymnasium environments."""

from gymnasium.envs.registration import register

from rl_playground.envs.racing import RacingConfig, RacingEnv, VehicleConfig
from rl_playground.envs.track import Track

register(id="rl_playground/Racing-v0", entry_point="rl_playground.envs.racing:RacingEnv")

__all__ = ["RacingConfig", "RacingEnv", "Track", "VehicleConfig"]
