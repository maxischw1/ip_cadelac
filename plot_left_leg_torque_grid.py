from argparse import ArgumentParser
from pathlib import Path
import csv
import re

import dill as pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

N_DOF = 2
SAMPLE_OFFSET = 1
HIST_LABELS = ["qp", "qv", "tau", "diff_tau"]


def sanitize(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def parse_epoch(path):
    match = re.match(r"(\d+)_", path.name)
    if match:
        return int(match.group(1))

    match = re.search(r"epochs_(\d+)", path.name)
    if match:
        return int(match.group(1))

    return -1


def find_latest_model():
    candidates = []

    candidates.extend(MODEL_DIR.glob(f"epochs_*{DATASET_NAME}.torch"))

    if CHECKPOINT_DIR.exists():
        candidates.extend(CHECKPOINT_DIR.glob(f"*{DATASET_NAME}.torch"))

    if not candidates:
        raise FileNotFoundError(f"No model found in {MODEL_DIR} or {CHECKPOINT_DIR}")

    return max(candidates, key=parse_epoch)


def get_left_bt24_labels(movement, max_segments=None):
    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)

    labels = [
        label for label in data["labels"]
        if "left_" in label and "BT24" in label and movement in label
    ]

    if not labels:
        raise ValueError(f"No left BT24 labels found for movement: {movement}")

    if max_segments is not None:
        labels = labels[:max_segments]

    return labels


def predict(model_path, labels):
    state = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    hyper = state["hyper"]
    hist_length = int(hyper["hist_length"])

    _, test_data, divider, dt_mean = load_dataset(
        filename=str(DATASET_PATH),
        test_label=labels,
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

    return test_labels, test_tau, tau_pred, divider, dt_mean


def segment_metrics(tau_true, tau_pred):
    error = tau_pred - tau_true
    mse = np.mean(error ** 2, axis=0)
    rmse = np.sqrt(mse)

    return mse, rmse


def plot_grid(test_labels, tau_true, tau_pred, divider, dt_mean, movement, output_path, metrics_path):
    n_segments = len(test_labels)

    fig, axes = plt.subplots(
        2,
        n_segments,
        figsize=(3.0 * n_segments, 6.0),
        sharey="row",
        squeeze=False,
    )

    metric_rows = []

    for seg_idx, label in enumerate(test_labels):
        start = divider[seg_idx]
        end = divider[seg_idx + 1]

        true_seg = tau_true[start:end]
        pred_seg = tau_pred[start:end]

        t = np.arange(start, end) * dt_mean

        mse, rmse = segment_metrics(true_seg, pred_seg)

        for joint_idx in range(2):
            ax = axes[joint_idx, seg_idx]

            ax.plot(t, true_seg[:, joint_idx], label="Ground Truth", linewidth=1.0)
            ax.plot(t, pred_seg[:, joint_idx], label="Prediction", linewidth=1.0)

            ax.grid(True, alpha=0.3)

            if joint_idx == 0:
                ax.set_title(f"{t[0]:.1f}-{t[-1]:.1f}s", fontsize=8)

            if seg_idx == 0:
                ax.set_ylabel(f"Joint {joint_idx} Torque [Nm]")

            if joint_idx == 1:
                ax.set_xlabel("Time [s]")

            ax.text(
                0.02,
                0.95,
                f"RMSE={rmse[joint_idx]:.3e}",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
            )

            if seg_idx == 0 and joint_idx == 0:
                ax.legend(fontsize=7)

            metric_rows.append(
                {
                    "label": label,
                    "joint": joint_idx,
                    "mse": float(mse[joint_idx]),
                    "rmse": float(rmse[joint_idx]),
                    "samples": int(true_seg.shape[0]),
                }
            )

    total_error = tau_pred - tau_true
    total_mse = np.mean(total_error ** 2, axis=0)
    total_rmse = np.sqrt(total_mse)

    fig.suptitle(
        f"BT24 left {movement} - actual vs predicted torque\n"
        f"Joint 0 RMSE={total_rmse[0]:.3e} Nm, Joint 1 RMSE={total_rmse[1]:.3e} Nm",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "joint", "mse", "rmse", "samples"])
        writer.writeheader()
        writer.writerows(metric_rows)

    return total_mse, total_rmse


def main():
    parser = ArgumentParser()
    parser.add_argument("--movement", type=str, required=True, help="Example: incline_walk or ball_toss")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="logs/left_leg_torque_grid")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else find_latest_model()
    labels = get_left_bt24_labels(args.movement, args.max_segments)

    test_labels, tau_true, tau_pred, divider, dt_mean = predict(model_path, labels)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"BT24_left_{sanitize(args.movement)}_torque_grid.png"
    metrics_path = output_dir / f"BT24_left_{sanitize(args.movement)}_torque_grid_metrics.csv"

    total_mse, total_rmse = plot_grid(
        test_labels,
        tau_true,
        tau_pred,
        divider,
        dt_mean,
        args.movement,
        output_path,
        metrics_path,
    )

    print("Model:", model_path)
    print("Movement:", args.movement)
    print("Labels:", test_labels)
    print("Saved plot:", output_path)
    print("Saved metrics:", metrics_path)
    print(f"Joint 0 MSE/RMSE: {total_mse[0]:.6e} / {total_rmse[0]:.6e} Nm")
    print(f"Joint 1 MSE/RMSE: {total_mse[1]:.6e} / {total_rmse[1]:.6e} Nm")


if __name__ == "__main__":
    main()
