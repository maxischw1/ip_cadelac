import dill as pickle
import numpy as np
from pathlib import Path
import shutil

p = Path("cadelac/learning/datasets/panda/exo_hip_knee_delan_2dof_left_all_trials.pkl")
backup = p.with_suffix(".before_time_fix.pkl")

shutil.copy2(p, backup)
print("Backup saved to:", backup)

with open(p, "rb") as f:
    data = pickle.load(f)

dt = 0.005  # 200 Hz

for i in range(len(data["labels"])):
    n = len(data["t"][i])

    # Force perfectly uniform time vector
    data["t"][i] = (np.arange(n, dtype=np.float64) * dt)

    # Recompute acceleration consistently from qv
    data["qa"][i] = np.gradient(data["qv"][i], dt, axis=0).astype(np.float64)

with open(p, "wb") as f:
    pickle.dump(data, f)

print("Fixed:", p)
print("Number of segments:", len(data["labels"]))
print("First label:", data["labels"][0])
print("First t dt var:", np.var(np.diff(data["t"][0])))
print("Last t dt var:", np.var(np.diff(data["t"][-1])))
