from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
TRACE_PATH = BASE_DIR / "ee_trace.npy"
TARGET_PATH = BASE_DIR / "target_trace.npy"

trace = np.load(TRACE_PATH)
target = np.load(TARGET_PATH)

plt.figure(figsize=(6, 6))
plt.plot(target[:, 0], target[:, 1], label="target")
if len(trace) > 0:
    plt.plot(trace[:, 0], trace[:, 1], label="drawn")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()