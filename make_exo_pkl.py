from pathlib import Path
import numpy as np
import dill as pickle

repo = Path.cwd()

exo_path = Path.home() / "Downloads" / "Exo.csv"
mom_path = Path.home() / "Downloads" / "Joint_Moments_Filt.csv"

out_path = repo / "cadelac" / "learning" / "datasets" / "panda" / "exo_hip_knee_delan_2dof_left_only.pkl"
out_path.parent.mkdir(parents=True, exist_ok=True)

print("Loading CSV files...")
exo = np.genfromtxt(exo_path, delimiter=",", names=True)
mom = np.genfromtxt(mom_path, delimiter=",", names=True)

time = np.asarray(exo["time"], dtype=np.float64)
time_mom = np.asarray(mom["time"], dtype=np.float64)

if len(time) != len(time_mom):
    raise ValueError("Exo.csv and Joint_Moments_Filt.csv have different lengths.")

if not np.allclose(time, time_mom, atol=1e-9):
    raise ValueError("Time vectors do not match.")

dt = float(np.median(np.diff(time)))

# q in radians
q = np.column_stack([
    exo["hip_angle_l"],
    exo["knee_angle_l"],
])
q = np.deg2rad(q)

# qdot in rad/s
qv = np.column_stack([
    exo["hip_angle_l_velocity_filt"],
    exo["knee_angle_l_velocity_filt"],
])
qv = np.deg2rad(qv)

# qddot in rad/s^2
qa = np.gradient(qv, dt, axis=0)

# target torque / joint moment
tau = np.column_stack([
    mom["hip_flexion_l_moment"],
    mom["knee_angle_l_moment"],
])

all_values = np.column_stack([q, qv, qa, tau])
valid = np.isfinite(all_values).all(axis=1)

time = time[valid]
q = q[valid]
qv = qv[valid]
qa = qa[valid]
tau = tau[valid]

segment_len = 2000

data = {
    "labels": [],
    "t": [],
    "qp": [],
    "qv": [],
    "qa": [],
    "tau": [],
    "tau_m": [],
    "tau_c": [],
    "tau_g": [],
    "metadata": {
        "source": "Exo.csv + Joint_Moments_Filt.csv",
        "side": "left",
        "dof": ["hip_flexion_l", "knee_angle_l"],
        "dt": dt,
        "q_unit": "rad",
        "qv_unit": "rad/s",
        "qa_unit": "rad/s^2",
        "tau_source": ["hip_flexion_l_moment", "knee_angle_l_moment"],
    },
}

start = 0
seg_id = 0

while start < len(time):
    end = min(start + segment_len, len(time))

    if end - start < 100:
        break

    label = f"left_seg_{seg_id:02d}"

    t_seg = time[start:end] - time[start]

    data["labels"].append(label)
    data["t"].append(t_seg.astype(np.float64))
    data["qp"].append(q[start:end].astype(np.float64))
    data["qv"].append(qv[start:end].astype(np.float64))
    data["qa"].append(qa[start:end].astype(np.float64))
    data["tau"].append(tau[start:end].astype(np.float64))

    zeros = np.zeros_like(tau[start:end], dtype=np.float64)
    data["tau_m"].append(zeros.copy())
    data["tau_c"].append(zeros.copy())
    data["tau_g"].append(zeros.copy())

    start = end
    seg_id += 1

with open(out_path, "wb") as f:
    pickle.dump(data, f)

print("Saved:", out_path)
print("Segments:", len(data["labels"]))
print("First q shape:", data["qp"][0].shape)
print("First tau shape:", data["tau"][0].shape)
print("dt:", dt)
