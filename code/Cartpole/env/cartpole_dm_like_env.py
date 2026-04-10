import os
import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_THIS_DIR = os.path.dirname(__file__)
_WORKSPACE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
)
if sys.path and sys.path[0] == "" and os.getcwd() == _WORKSPACE_ROOT:
    sys.path.pop(0)
if _WORKSPACE_ROOT in sys.path:
    sys.path.remove(_WORKSPACE_ROOT)

import mujoco

if not hasattr(mujoco, "MjModel"):
    raise ImportError(
        "Official `mujoco` Python bindings were not found. "
        "The local folder `C:\\mujoco\\mujoco` is shadowing the package name, "
        "or the package is not installed in this Python environment."
    )


@dataclass
class DisturbanceConfig:
    enabled: bool = False
    probability: float = 0.03
    min_force: float = -0.15
    max_force: float = 0.15
    duration_steps: int = 30


class CartpoleDmLikeSwingupEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 100,
    }

    def __init__(
        self,
        xml_path=None,
        render_mode=None,
        max_steps=1000,
        frame_skip=1,
        action_scale=1.0,
        disturbance_config=None,
        width=640,
        height=480,
    ):
        super().__init__()

        if xml_path is None:
            xml_path = os.path.join(
                os.path.dirname(__file__), "cartpole_dm_like_swingup.xml"
            )

        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.action_scale = float(action_scale)
        self.width = int(width)
        self.height = int(height)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.disturbance = disturbance_config or DisturbanceConfig()
        self._active_disturbance = 0.0
        self._disturbance_steps_left = 0

        self._viewer = None
        self._renderer = None
        self._step_count = 0

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32,
        )

    def _pole_cosine(self):
        return float(np.cos(self.data.qpos[1]))

    def _pole_sine(self):
        return float(np.sin(self.data.qpos[1]))

    def _get_obs(self):
        return np.array(
            [
                float(self.data.qpos[0]),
                self._pole_cosine(),
                self._pole_sine(),
                float(self.data.qvel[0]),
                float(self.data.qvel[1]),
            ],
            dtype=np.float32,
        )

    def _tolerance_quadratic(self, value, margin, value_at_margin=0.1):
        value = np.asarray(value, dtype=np.float64)
        margin = float(margin)
        value_at_margin = float(value_at_margin)

        if margin <= 0:
            return (np.abs(value) <= 0).astype(np.float64)

        if value_at_margin <= 0.0:
            return np.where(np.abs(value) < margin, 1.0 - (np.abs(value) / margin) ** 2, 0.0)

        scaled = np.abs(value) / margin
        scale = np.sqrt(1.0 / value_at_margin) - 1.0
        reward = 1.0 / (1.0 + scale * scaled**2)
        reward = np.where(np.abs(value) <= 0.0, 1.0, reward)
        return reward

    def _sample_disturbance(self):
        if not self.disturbance.enabled:
            return

        if self._disturbance_steps_left == 0:
            if self.np_random.random() < self.disturbance.probability:
                self._active_disturbance = float(
                    self.np_random.uniform(
                        self.disturbance.min_force,
                        self.disturbance.max_force,
                    )
                )
                self._disturbance_steps_left = int(self.disturbance.duration_steps)

    def _apply_action(self, action):
        clipped = np.asarray(action, dtype=np.float32).reshape(1)
        clipped = np.clip(clipped, self.action_space.low, self.action_space.high)

        self._sample_disturbance()

        disturbance_force = 0.0
        if self._disturbance_steps_left > 0:
            disturbance_force = self._active_disturbance
            self._disturbance_steps_left -= 1

        applied = float(clipped[0]) * self.action_scale + disturbance_force
        self.data.ctrl[0] = applied
        return float(clipped[0]), float(applied), float(disturbance_force)

    def _get_reward(self, normalized_action):
        upright = (self._pole_cosine() + 1.0) / 2.0
        centered = self._tolerance_quadratic(self.data.qpos[0], margin=2.0)
        centered = (1.0 + float(centered)) / 2.0

        small_control = self._tolerance_quadratic(
            normalized_action, margin=1.0, value_at_margin=0.0
        )
        small_control = (4.0 + float(small_control)) / 5.0

        small_velocity = self._tolerance_quadratic(self.data.qvel[1], margin=5.0)
        small_velocity = (1.0 + float(small_velocity)) / 2.0

        return float(upright * centered * small_control * small_velocity)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0] = 0.01 * self.np_random.standard_normal()
        self.data.qpos[1] = np.pi + 0.01 * self.np_random.standard_normal()
        self.data.qvel[:] = 0.01 * self.np_random.standard_normal(self.model.nv)

        self._step_count = 0
        self._active_disturbance = 0.0
        self._disturbance_steps_left = 0

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {
            "upright": float((self._pole_cosine() + 1.0) / 2.0),
            "cart_position": float(self.data.qpos[0]),
        }

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        self._step_count += 1

        normalized_action, applied_action, disturbance_force = self._apply_action(action)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._get_reward(normalized_action)

        terminated = False
        truncated = self._step_count >= self.max_steps
        info = {
            "upright": float((self._pole_cosine() + 1.0) / 2.0),
            "centered": float((1.0 + self._tolerance_quadratic(self.data.qpos[0], 2.0)) / 2.0),
            "cart_position": float(self.data.qpos[0]),
            "angular_velocity": float(self.data.qvel[1]),
            "normalized_action": float(normalized_action),
            "applied_action": float(applied_action),
            "disturbance_force": float(disturbance_force),
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                    show_left_ui=True,
                    show_right_ui=True,
                )
            if self._viewer.is_running():
                self._viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model,
                    width=self.width,
                    height=self.height,
                )
            self._renderer.update_scene(self.data, camera="side")
            return self._renderer.render()

        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
