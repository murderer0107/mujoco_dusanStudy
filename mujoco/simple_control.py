from dm_control import suite
from dm_control.viewer import launch
import numpy as np

env = suite.load("cartpole", "swingup")

def policy(time_step):
    action = np.random.uniform(-1, 1, size=(1,))
    print("action:", action)  # ← 이거 핵심
    return action

launch(env, policy=policy)