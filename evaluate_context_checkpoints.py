from pathlib import Path
import csv
import re

import dill as pickle
import numpy as np
import torch

from cadelac.learning.models.context_aware_delan import ContextAwareDeLaN
from cadelac.learning.data_scripts.utils import load_dataset


REPO_DIR = Path(__file__).resolve().parent
LEARNING_DIR = REPO_DIR / "cadelac" / "learning"

DATASET_NAME = "exo_hip_knee_delan_2dof_left_all_trials_context"
DATASET_PATH = LEARNING_DIR / "datasets" / "panda" / f"{DATASET_NAME}.pkl"

MODEL_DIR = LEARNING_DIR / "trained_models" / "res_model" / "panda" / "ContextAware"
CHECKPOINT_DIR = MODEL_DIR / "checkpoint"

OUTPUT_CSV = REPO_DIR / "logs" / "exo_context_checkpoint_metrics.csv"

N_DOF = 2
SAMPLE_OFFSET = 1
HIST_LABELS = ["qp", "qv", "tau", "diff_tau"]


def parse_epoch(path: Path, state: dict) -> int:
    if "epoch" in state:
        return int(state["epoch"])

    match = re.match(r"(\d+)_", path.name)
    if match:
        return int(match.group(1))

    match = re.search(r"epochs_(\d+)", path.name)
    if match:
        return int(match.group(1))

    return -1


def find_models():
    models = []

    if CHECKPOINT_DIR.exists():
        models.extend(sorted(CHECKPOINT_DIR.glob(f"*{DATASET_NAME}.torch")))

    models.extend(sorted(MODEL_DIR.glob(f"epochs_*{DATASET_NAME}.torch")))

    if not models:
        raise FileNotFoundError(
            f"No trained models found in {MODEL_DIR} or {CHECKPOINT_DIR}."
        )

    return models


def load_test_data(hist_length: int):
    with open(DATASET_PATH, "rb") as f:
        raw_data = pickle.load(f)

    test_label = [label for label in raw_data["labels"] if "BT24" in label]

    _, test_data, _, _ = load_dataset(
        filename=str(DATASET_PATH),
        test_label=test_label,
        full_model=False,
        sample_offset=SAMPLE_OFFSET,
        dataset_use=1.0,
        n_dof=N_DOF,
        hist_length=hist_length,
        hist_labels=HIST_LABELS,
        add_noise=False,
    )

    (
        test_labels,
        test_qp,
        test_qv,
        test_qa,
        test_tau,
        _test_m,
        _test_c,
        _test_g,
        test_hist_qp,
        test_hist_qv,
        test_hist_tau,
        _test_hist_diff_tau,
    ) = test_data

    test_lstm_input = np.concatenate(
        (test_hist_qp, test_hist_qv, test_hist_tau),
        axis=-1,
    )

    return test_labels, test_qp, test_qv, test_qa, test_tau, test_lstm_input


def evaluate_model(model_path: Path):
    state = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    hyper = state["hyper"]
    hist_length = int(hyper["hist_length"])

    test_labels, test_qp, test_qv, test_qa, test_tau, test_lstm_input = load_test_data(hist_length)

    model = ContextAwareDeLaN(N_DOF, **hyper)
    model.load_state_dict(state["state_dict"])
    model.cpu()
    model.eval()

    q = torch.from_numpy(test_qp).float()
    qd = torch.from_numpy(test_qv).float()
    qdd = torch.from_numpy(test_qa).float()
    lstm_input = torch.from_numpy(test_lstm_input).float()

    with torch.no_grad():
        tau_pred = model(q, qd, qdd, lstm_input)[0].cpu().numpy()

    error = tau_pred - test_tau

    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    joint_mse = np.mean(error ** 2, axis=0)
    joint_rmse = np.sqrt(joint_mse)

    return {
        "epoch": parse_epoch(model_path, state),
        "model_path": str(model_path),
        "n_samples": int(test_tau.shape[0]),
        "torque_mse": mse,
        "torque_rmse": rmse,
        "joint_0_mse": float(joint_mse[0]),
        "joint_1_mse": float(joint_mse[1]),
        "joint_0_rmse": float(joint_rmse[0]),
        "joint_1_rmse": float(joint_rmse[1]),
    }


def main():
    models = find_models()
    rows = []

    for model_path in models:
        print(f"Evaluating: {model_path}")
        rows.append(evaluate_model(model_path))

    rows = sorted(rows, key=lambda row: row["epoch"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved metrics to:", OUTPUT_CSV)
    print("\nEpoch | Torque MSE | Torque RMSE")
    print("--------------------------------")
    for row in rows:
        print(f"{row['epoch']:5d} | {row['torque_mse']:.6e} | {row['torque_rmse']:.6e}")


if __name__ == "__main__":
    main()
