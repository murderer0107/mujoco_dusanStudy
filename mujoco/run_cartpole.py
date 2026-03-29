from dm_control import suite
from dm_control.viewer import launch
import numpy as np

print("🔥 실행됨")

env = suite.load("cartpole", "swingup")

def policy(time_step):
    print("🔥 policy 호출됨")
    action = np.random.uniform(-1, 1, size=(1,))
    return action

launch(env, policy=policy)