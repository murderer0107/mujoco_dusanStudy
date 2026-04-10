import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(BASE_DIR, "env"))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from cartpole_rl_env import CartpoleSwingupEnv


def main():
    xml_path = os.path.join(BASE_DIR, "env", "cartpole_swingup.xml")
    env = CartpoleSwingupEnv(xml_path=xml_path, max_steps=3000)

    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-5,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        n_epochs=10,
        ent_coef=0.001,
        tensorboard_log=os.path.join(BASE_DIR, "logs", "ppo_cartpole_tensorboard"),
    )

    model.learn(total_timesteps=300_000)
    model.save(os.path.join(BASE_DIR, "models", "ppo_cartpole_model"))


if __name__ == "__main__":
    main()