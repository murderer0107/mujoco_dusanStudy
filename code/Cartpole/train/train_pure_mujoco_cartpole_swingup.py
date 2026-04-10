import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_DIR = os.path.join(BASE_DIR, "env")
sys.path.append(ENV_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from cartpole_dm_like_env import CartpoleDmLikeSwingupEnv, DisturbanceConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--disturbance", action="store_true")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--log-name", type=str, default=None)
    return parser.parse_args()


def make_env(max_steps, enable_disturbance):
    disturbance = DisturbanceConfig(enabled=enable_disturbance)
    env = CartpoleDmLikeSwingupEnv(
        xml_path=os.path.join(ENV_DIR, "cartpole_dm_like_swingup.xml"),
        max_steps=max_steps,
        disturbance_config=disturbance,
    )
    return Monitor(env)


def main():
    args = parse_args()

    env = make_env(args.max_steps, args.disturbance)
    check_env(env.unwrapped, warn=True)

    suffix = "_disturb" if args.disturbance else ""
    model_name = args.model_name or f"ppo_pure_mujoco_cartpole_swingup{suffix}"
    log_name = args.log_name or f"tb_pure_mujoco_cartpole{suffix}"

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
        tensorboard_log=os.path.join(BASE_DIR, "logs", log_name),
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(BASE_DIR, "models", model_name))
    env.close()


if __name__ == "__main__":
    main()
