# Local repo import bootstrap for running this script from scripts/data/
from pathlib import Path as _Path
import sys as _sys
_REPO_DIR_BOOTSTRAP = _Path(__file__).resolve().parents[2]
if str(_REPO_DIR_BOOTSTRAP) not in _sys.path:
    _sys.path.insert(0, str(_REPO_DIR_BOOTSTRAP))

from pathlib import Path
import numpy as np
import dill as pickle

ROOT = Path.home() / "Downloads" / "codeocean_exo_data"
OUT_DIR = Path.cwd() / "cadelac" / "learning" / "datasets" / "panda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_LEN = 2000       # 10 Sekunden bei 200 Hz
MIN_SEGMENT_LEN = 400    # mindestens 2 Sekunden

def load_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding=None)

def has_cols(arr, cols):
    names = set(arr.dtype.names or [])
    return all(c in names for c in cols)

# change tau_cols
def make_dataset(side):
    if side == "left":
        angle_cols = ["hip_angle_l", "knee_angle_l"]
        vel_cols = ["hip_angle_l_velocity_filt", "knee_angle_l_velocity_filt"]
        tau_cols = ["hip_flexion_l_moment", "knee_angle_l_moment"]
        out_name = "exo_hip_knee_delan_2dof_left_all_trials.pkl"

    elif side == "right":
        angle_cols = ["hip_angle_r", "knee_angle_r"]
        vel_cols = ["hip_angle_r_velocity_filt", "knee_angle_r_velocity_filt"]
        tau_cols = ["hip_flexion_r_moment", "knee_angle_r_moment"]
        out_name = "exo_hip_knee_delan_2dof_right_all_trials.pkl"

    else:
        raise ValueError("side must be left or right")

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
            "root": str(ROOT),
            "side": side,
            "segment_len": SEGMENT_LEN,
            "angle_cols": angle_cols,
            "vel_cols": vel_cols,
            "tau_cols": tau_cols,
            "q_unit": "rad",
            "qv_unit": "rad/s",
            "qa_unit": "rad/s^2",
        },
    }

    pairs = []
    for exo_path in ROOT.rglob("Exo.csv"):
        mom_path = exo_path.parent / "Joint_Moments_Filt.csv"
        if mom_path.exists():
            pairs.append((exo_path, mom_path))

    print(f"\nFound {len(pairs)} Exo/Moment pairs for {side}")

    global_seg_id = 0

    for exo_path, mom_path in pairs:
        try:
            exo = load_csv(exo_path)
            mom = load_csv(mom_path)

            required_exo = ["time"] + angle_cols + vel_cols
            required_mom = ["time"] + tau_cols

            if not has_cols(exo, required_exo):
                print("Skipping because Exo columns missing:", exo_path)
                print("Available Exo columns:", exo.dtype.names)
                continue

            if not has_cols(mom, required_mom):
                print("Skipping because Moment columns missing:", mom_path)
                print("Available Moment columns:", mom.dtype.names)
                continue

            time = np.asarray(exo["time"], dtype=np.float64)
            time_mom = np.asarray(mom["time"], dtype=np.float64)

            if len(time) != len(time_mom):
                print("Skipping length mismatch:", exo_path)
                continue

            if not np.allclose(time, time_mom, atol=1e-6):
                print("Skipping time mismatch:", exo_path)
                continue

            dt = float(np.median(np.diff(time)))

            q = np.column_stack([exo[c] for c in angle_cols])
            q = np.deg2rad(q)

            qv = np.column_stack([exo[c] for c in vel_cols])
            qv = np.deg2rad(qv)

            qa = np.gradient(qv, dt, axis=0)

            tau = np.column_stack([mom[c] for c in tau_cols])

            valid = np.isfinite(np.column_stack([q, qv, qa, tau])).all(axis=1)

            time = time[valid]
            q = q[valid]
            qv = qv[valid]
            qa = qa[valid]
            tau = tau[valid]

            trial_name = exo_path.parent.relative_to(ROOT).as_posix()
            trial_name = trial_name.replace("/", "__")

            start = 0
            local_seg_id = 0

            while start < len(time):
                end = min(start + SEGMENT_LEN, len(time))

                if end - start < MIN_SEGMENT_LEN:
                    break

                label = f"{side}_{global_seg_id:05d}_{trial_name}_seg_{local_seg_id:02d}"
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
                local_seg_id += 1
                global_seg_id += 1

        except Exception as e:
            print("Skipping due to error:", exo_path, e)

    out_path = OUT_DIR / out_name

    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print("Saved:", out_path)
    print("Number of segments:", len(data["labels"]))

    if data["labels"]:
        print("First label:", data["labels"][0])
        print("First q shape:", data["qp"][0].shape)
        print("First tau shape:", data["tau"][0].shape)

    # Context-Aware training dataset.
    # In the simplified exoskeleton setup, the measured torque is used directly
    # as the residual target: diff_tau = tau.
    context_data = dict(data)
    context_data["metadata"] = dict(data["metadata"])
    context_data["metadata"]["context_setup"] = "diff_tau = tau"

    context_data["diff_tau"] = [tau_seg.copy() for tau_seg in data["tau"]]
    context_data["diff_tau_m"] = [np.zeros_like(tau_seg) for tau_seg in data["tau"]]
    context_data["diff_tau_c"] = [np.zeros_like(tau_seg) for tau_seg in data["tau"]]
    context_data["diff_tau_g"] = [np.zeros_like(tau_seg) for tau_seg in data["tau"]]

    context_out_path = OUT_DIR / out_name.replace(".pkl", "_context.pkl")

    with open(context_out_path, "wb") as f:
        pickle.dump(context_data, f)

    print("Saved Context-Aware dataset:", context_out_path)

make_dataset("left")
make_dataset("right")
