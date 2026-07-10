# Local repo import bootstrap for running this script from scripts/data/
from pathlib import Path as _Path
import sys as _sys
_REPO_DIR_BOOTSTRAP = _Path(__file__).resolve().parents[2]
if str(_REPO_DIR_BOOTSTRAP) not in _sys.path:
    _sys.path.insert(0, str(_REPO_DIR_BOOTSTRAP))

import dill as pickle
import numpy as np
from pathlib import Path
import shutil

p = Path("cadelac/learning/datasets/panda/exo_hip_knee_delan_2dof_left_all_trials.pkl")

# Backingup original file (safety-measure)
backup = p.with_suffix(".before_time_fix.pkl")
shutil.copy2(p, backup)
print("Backup saved to:", backup)


with open(p, "rb") as f:
    data = pickle.load(f)

# Set sampling time
dt = 0.005  # 200 Hz

# Correct uniform time-axis and acceleration computation 
for i in range(len(data["labels"])):
    n = len(data["t"][i])

    # Force perfectly uniform time vector
    data["t"][i] = (np.arange(n, dtype=np.float64) * dt)

    # Recompute acceleration consistently from qv
    data["qa"][i] = np.gradient(data["qv"][i], dt, axis=0).astype(np.float64)

with open(p, "wb") as f:
    pickle.dump(data, f)

# control output
print("Fixed:", p)
print("Number of segments:", len(data["labels"]))
print("First label:", data["labels"][0])
print("First t dt var:", np.var(np.diff(data["t"][0])))
print("Last t dt var:", np.var(np.diff(data["t"][-1])))
