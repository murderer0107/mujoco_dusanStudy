import argparse
import os
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_DIR = os.path.join(BASE_DIR, "env")
sys.path.append(ENV_DIR)

from stable_baselines3 import PPO

from cartpole_dm_like_env import CartpoleDmLikeSwingupEnv, DisturbanceConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--sleep", type=float, default=0.01)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--disturbance", action="store_true")
    parser.add_argument("--strong-disturbance", action="store_true")
    return parser.parse_args()


def make_env(max_steps, enable_disturbance, strong_disturbance):
    if strong_disturbance:
        disturbance = DisturbanceConfig(
            enabled=enable_disturbance,
            probability=0.05,
            min_force=-0.30,
            max_force=0.30,
            duration_steps=100,
        )
    else:
        disturbance = DisturbanceConfig(enabled=enable_disturbance)

    return CartpoleDmLikeSwingupEnv(
        xml_path=os.path.join(ENV_DIR, "cartpole_dm_like_swingup.xml"),
        render_mode="human",
        max_steps=max_steps,
        disturbance_config=disturbance,
    )


def default_model_path(enable_disturbance):
    suffix = "_disturb" if enable_disturbance else ""
    return os.path.join(
        BASE_DIR,
        "models",
        f"ppo_pure_mujoco_cartpole_swingup{suffix}",
    )


def main():
    args = parse_args()

    env = make_env(args.max_steps, args.disturbance, args.strong_disturbance)
    model = PPO.load(args.model_path or default_model_path(args.disturbance))

    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated

            if step_count % 200 == 0:
                print(
                    f"episode={episode + 1} step={step_count} "
                    f"reward={episode_reward:.3f} upright={info['upright']:.3f} "
                    f"disturbance={info['disturbance_force']:.3f}"
                )

            time.sleep(args.sleep)

        print(
            f"episode={episode + 1} finished "
            f"steps={step_count} total_reward={episode_reward:.3f}"
        )

    env.close()


if __name__ == "__main__":
    main()
