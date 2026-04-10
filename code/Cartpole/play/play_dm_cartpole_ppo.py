import time
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
import numpy as np

from dm_control import suite
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO


def main():
    dm_env = suite.load(
        domain_name="cartpole",
        task_name="swingup",
        task_kwargs={"time_limit": 30},
    )
    env = DmControlCompatibilityV0(dm_env, render_mode="human")
    env = FlattenObservation(env)

    model_path = os.path.join(BASE_DIR, "models", "ppo_dm_cartpole_swingup_disturb")
    model = PPO.load(model_path)

    obs, info = env.reset()

    step_count = 0
    disturbance = 0.0
    disturb_steps = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = np.array(action, dtype=np.float32)

        # During evaluation, start a disturbance every 300 steps.
        if step_count % 300 == 0 and step_count > 0 and disturb_steps == 0:
            disturbance = np.random.uniform(-2.0, 2.0)
            disturb_steps = 20
            print(f"disturbance applied: {disturbance:.3f}")

        disturbed_action = action.copy()

        # Keep the same disturbance for a short window.
        if disturb_steps > 0:
            disturbed_action[0] += disturbance
            disturb_steps -= 1

        disturbed_action = np.clip(
            disturbed_action,
            env.action_space.low,
            env.action_space.high
        )

        obs, reward, terminated, truncated, info = env.step(disturbed_action)
        env.render()
        time.sleep(0.01)

        step_count += 1

        if terminated or truncated:
            print("reset")
            obs, info = env.reset()
            step_count = 0
            disturbance = 0.0
            disturb_steps = 0


if __name__ == "__main__":
    main()
