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

# Define repository-relative paths for the dataset, trained models,
# checkpoints, and output directory.
REPO_DIR = Path(__file__).resolve().parent
LEARNING_DIR = REPO_DIR / "cadelac" / "learning"
DATASET_NAME = "exo_hip_knee_delan_2dof_left_all_trials_context"
DATASET_PATH = LEARNING_DIR / "datasets" / "panda" / f"{DATASET_NAME}.pkl"
MODEL_DIR = LEARNING_DIR / "trained_models" / "res_model" / "panda" / "ContextAware"
CHECKPOINT_DIR = MODEL_DIR / "checkpoint"
OUTPUT_DIR = REPO_DIR / "logs" / "torque_zoom"

# Exoskeleton setup:
# 2 DoF means hip and knee.
N_DOF = 2
SAMPLE_OFFSET = 1 # Skipped first samples for hist
HIST_LABELS = ["qp", "qv", "tau", "diff_tau"]


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)

# Helper: Find the latest available trained model
def find_latest_model():
    candidates = []

    candidates.extend(sorted(MODEL_DIR.glob(f"epochs_*{DATASET_NAME}.torch")))

    # If a checkpoint directory exists, also add checkpoint models
    if CHECKPOINT_DIR.exists():
        candidates.extend(sorted(CHECKPOINT_DIR.glob(f"*{DATASET_NAME}.torch")))

    if not candidates:
        raise FileNotFoundError(
            f"No model found in {MODEL_DIR} or {CHECKPOINT_DIR}."
        )

    return candidates[-1]

# Helper: Select test labels
def get_test_labels(segment: str | None):
    with open(DATASET_PATH, "rb") as f:
        raw_data = pickle.load(f)

    if segment:
        labels = [label for label in raw_data["labels"] if segment in label]
        if not labels:
            raise ValueError(f"No label contains segment string: {segment}")
        return labels

    return [label for label in raw_data["labels"] if "BT24" in label]

# Load test data and build LSTM input
def load_prediction_data(hist_length: int, segment: str | None):
    test_label = get_test_labels(segment)

    _, test_data, _, dt_mean = load_dataset(
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

    # Build the LSTM history input from position, velocity, and torque history.
    test_lstm_input = np.concatenate(
        (test_hist_qp, test_hist_qv, test_hist_tau),
        axis=-1,
    )

    return test_labels, test_qp, test_qv, test_qa, test_tau, test_lstm_input, dt_mean

# Load model and generate torque predictions
def predict(model_path: Path, segment: str | None):
    state = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    hyper = state["hyper"]
    hist_length = int(hyper["hist_length"])

    test_labels, test_qp, test_qv, test_qa, test_tau, test_lstm_input, dt_mean = load_prediction_data(
        hist_length,
        segment,
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

    return test_labels, test_tau, tau_pred, dt_mean

# Computes total MSE/RMSE over both joints and individual MSE/RMSE per joint.
def compute_metrics(tau_true, tau_pred):
    error = tau_pred - tau_true

    joint_mse = np.mean(error ** 2, axis=0)
    joint_rmse = np.sqrt(joint_mse)

    torque_mse = float(np.mean(error ** 2))
    torque_rmse = float(np.sqrt(torque_mse))

    return {
        "n_samples": int(tau_true.shape[0]),
        "torque_mse": torque_mse,
        "torque_rmse": torque_rmse,
        "joint_0_mse": float(joint_mse[0]),
        "joint_1_mse": float(joint_mse[1]),
        "joint_0_rmse": float(joint_rmse[0]),
        "joint_1_rmse": float(joint_rmse[1]),
    }

# Stores the used model path, evaluated labels, and all computed error metrics as CSV
def save_metrics(output_path, model_path, test_labels, metrics):
    metrics_path = output_path.with_name(output_path.stem + "_metrics.csv")

    row = {
        "model_path": str(model_path),
        "labels": ";".join(test_labels),
        **metrics,
    }

    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    return metrics_path

# Creates a two-row plot:
# One subplot for hip torque and one subplot for knee torque
# Each subplot shows ground truth and Context-Aware DeLaN prediction
def plot_torque(test_labels, tau_true, tau_pred, dt_mean, output_path, max_samples=None):
    if max_samples is not None:
        tau_true = tau_true[:max_samples]
        tau_pred = tau_pred[:max_samples]

    metrics = compute_metrics(tau_true, tau_pred)

    t = np.arange(tau_true.shape[0]) * dt_mean

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for joint_idx, ax in enumerate(axes):
        ax.plot(t, tau_true[:, joint_idx], label="Ground Truth")
        ax.plot(t, tau_pred[:, joint_idx], label="Context-Aware Prediction")
        ax.set_ylabel(f"Joint {joint_idx} Torque [Nm]")
        ax.grid(True)
        ax.legend()

        metric_text = (
            f"MSE = {metrics[f'joint_{joint_idx}_mse']:.3e}\n"
            f"RMSE = {metrics[f'joint_{joint_idx}_rmse']:.3e} Nm"
        )
        ax.text(
            0.01,
            0.95,
            metric_text,
            transform=ax.transAxes,
            verticalalignment="top",
        )

    axes[-1].set_xlabel("Time [s]")

    title = "Torque Prediction vs Ground Truth"
    if len(test_labels) == 1:
        title += f" - {test_labels[0]}"
    else:
        title += " - BT24 Test Split"

    title += (
        f"\nTotal MSE = {metrics['torque_mse']:.3e}, "
        f"Total RMSE = {metrics['torque_rmse']:.3e} Nm"
    )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return metrics
# Main logic
# Parses command-line arguments, loads the selected model,
# generates predictions, creates the plot, saves metrics,
# and prints a short summary to the terminal.
def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--segment", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else find_latest_model()

    test_labels, tau_true, tau_pred, dt_mean = predict(model_path, args.segment)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.segment:
        name = f"exo_context_zoomed_torque_{sanitize(args.segment)}.png"
    else:
        name = "exo_context_zoomed_torque_full_BT24.png"

    output_path = output_dir / name
    metrics = plot_torque(test_labels, tau_true, tau_pred, dt_mean, output_path, args.max_samples)
    metrics_path = save_metrics(output_path, model_path, test_labels, metrics)

    print("Model:", model_path)
    print("Labels:", test_labels)
    print("Saved:", output_path)
    print("Saved metrics:", metrics_path)
    print(f"Torque MSE:  {metrics['torque_mse']:.6e}")
    print(f"Torque RMSE: {metrics['torque_rmse']:.6e} Nm")
    print(f"Joint 0 RMSE: {metrics['joint_0_rmse']:.6e} Nm")
    print(f"Joint 1 RMSE: {metrics['joint_1_rmse']:.6e} Nm")


if __name__ == "__main__":
    main()
