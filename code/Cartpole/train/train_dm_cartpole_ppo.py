import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

from dm_control import suite
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO


def main():
    dm_env = suite.load(domain_name="cartpole", task_name="swingup")
    env = DmControlCompatibilityV0(dm_env)
    env = FlattenObservation(env)

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
        tensorboard_log=os.path.join(BASE_DIR, "logs", "tb_dm_cartpole")
        )

    model.learn(total_timesteps=300_000)
    model.save(os.path.join(BASE_DIR, "models", "ppo_dm_cartpole_swingup"))

if __name__ == "__main__":
    main()
