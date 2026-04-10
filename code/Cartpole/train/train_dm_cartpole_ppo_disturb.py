import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

import numpy as np
import gymnasium as gym

from dm_control import suite
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO


# Add random disturbances during training to improve robustness.
class DisturbanceWrapper(gym.ActionWrapper):
    def __init__(self, env, prob=0.02, force_min=-1.5, force_max=1.5, duration=30):
        super().__init__(env)
        self.prob = prob
        self.force_min = force_min
        self.force_max = force_max
        self.duration = duration

        self.current_disturbance = 0.0
        self.remaining_steps = 0

    def action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1).copy()

        if self.remaining_steps == 0 and np.random.rand() < self.prob:
            self.current_disturbance = float(
                np.random.uniform(self.force_min, self.force_max)
            )
            self.remaining_steps = self.duration

        if self.remaining_steps > 0:
            action[0] += self.current_disturbance
            self.remaining_steps -= 1

        action = np.clip(action, self.action_space.low, self.action_space.high)
        return np.asarray(action, dtype=np.float32)


def make_env(render_mode=None):
    dm_env = suite.load(
        domain_name="cartpole",
        task_name="swingup",
        task_kwargs={"time_limit": 30},
    )
    env = DmControlCompatibilityV0(dm_env, render_mode=render_mode)
    env = FlattenObservation(env)

    # Disturbances start at random times and persist for `duration` steps.
    env = DisturbanceWrapper(
        env, prob=0.03, force_min=-1.5, force_max=1.5, duration=30
    )
    return env


def main():
    env = make_env()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        n_epochs=10,
        ent_coef=0.001,
        tensorboard_log=os.path.join(BASE_DIR, "logs", "tb_dm_cartpole_disturb")
    )

    model.learn(total_timesteps=300_000)
    model.save(os.path.join(BASE_DIR, "models", "ppo_dm_cartpole_swingup_disturb"))


if __name__ == "__main__":
    main()
