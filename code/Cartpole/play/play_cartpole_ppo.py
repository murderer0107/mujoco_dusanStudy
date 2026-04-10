import os
import time
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(BASE_DIR, "env"))

from cartpole_rl_env import CartpoleSwingupEnv

MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo_cartpole_model")
XML_PATH = os.path.join(BASE_DIR, "env", "cartpole_swingup.xml")


def main():
    env = CartpoleSwingupEnv(xml_path=XML_PATH, max_steps=1000)
    model = PPO.load(MODEL_PATH)

    obs, info = env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15

        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            viewer.sync()
            time.sleep(env.model.opt.timestep)

            if terminated or truncated:
                obs, info = env.reset()


if __name__ == "__main__":
    main()