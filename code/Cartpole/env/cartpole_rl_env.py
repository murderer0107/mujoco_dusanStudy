import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class CartpoleSwingupEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, xml_path="cartpole_swingup.xml", max_steps=3000):
        super().__init__()

        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.max_steps = max_steps
        self.step_count = 0

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        low = np.array([-np.inf, -1.0, -1.0, -np.inf, -np.inf], dtype=np.float32)
        high = np.array([np.inf, 1.0, 1.0, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_obs(self):
        x = float(self.data.qpos[0])
        theta = float(self.data.qpos[1])
        x_dot = float(self.data.qvel[0])
        theta_dot = float(self.data.qvel[1])

        return np.array(
            [x, np.cos(theta), np.sin(theta), x_dot, theta_dot],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Start near the downward-hanging configuration.
        self.data.qpos[0] = self.np_random.normal(0.0, 0.01)
        self.data.qpos[1] = self.np_random.normal(0.0, 0.05)
        self.data.qvel[0] = self.np_random.normal(0.0, 0.01)
        self.data.qvel[1] = self.np_random.normal(0.0, 0.01)

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        force = float(np.clip(action[0], -1.0, 1.0)) * 10.0
        self.data.ctrl[0] = force

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        x = float(self.data.qpos[0])
        theta = float(self.data.qpos[1])

        # Map downward to 0 and upright to 1.
        # cos(theta) is -1 when the pole points downward.
        upright = (1 - np.cos(theta)) / 2
        centered = np.exp(-0.5 * x**2)

        reward = (
            1.0 * upright
            + 0.5 * centered
            - 0.01 * theta_dot**2
        )
    

        terminated = bool(abs(x) > 1.8)
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "force": force,
            "upright": float(upright),
            "centered": float(centered),
        }

        return obs, float(reward), terminated, truncated, info
