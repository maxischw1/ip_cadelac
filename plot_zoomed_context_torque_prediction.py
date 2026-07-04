from argparse import ArgumentParser
from pathlib import Path
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

OUTPUT_DIR = REPO_DIR / "logs" / "torque_zoom"

N_DOF = 2
SAMPLE_OFFSET = 1
HIST_LABELS = ["qp", "qv", "tau", "diff_tau"]


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def find_latest_model():
    candidates = []

    candidates.extend(sorted(MODEL_DIR.glob(f"epochs_*{DATASET_NAME}.torch")))

    if CHECKPOINT_DIR.exists():
        candidates.extend(sorted(CHECKPOINT_DIR.glob(f"*{DATASET_NAME}.torch")))

    if not candidates:
        raise FileNotFoundError(
            f"No model found in {MODEL_DIR} or {CHECKPOINT_DIR}."
        )

    return candidates[-1]


def get_test_labels(segment: str | None):
    with open(DATASET_PATH, "rb") as f:
        raw_data = pickle.load(f)

    if segment:
        labels = [label for label in raw_data["labels"] if segment in label]
        if not labels:
            raise ValueError(f"No label contains segment string: {segment}")
        return labels

    return [label for label in raw_data["labels"] if "BT24" in label]


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

    test_lstm_input = np.concatenate(
        (test_hist_qp, test_hist_qv, test_hist_tau),
        axis=-1,
    )

    return test_labels, test_qp, test_qv, test_qa, test_tau, test_lstm_input, dt_mean


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


def plot_torque(test_labels, tau_true, tau_pred, dt_mean, output_path, max_samples=None):
    if max_samples is not None:
        tau_true = tau_true[:max_samples]
        tau_pred = tau_pred[:max_samples]

    t = np.arange(tau_true.shape[0]) * dt_mean

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for joint_idx, ax in enumerate(axes):
        ax.plot(t, tau_true[:, joint_idx], label="Ground Truth")
        ax.plot(t, tau_pred[:, joint_idx], label="Context-Aware Prediction")
        ax.set_ylabel(f"Joint {joint_idx} Torque [Nm]")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time [s]")

    title = "Torque Prediction vs Ground Truth"
    if len(test_labels) == 1:
        title += f" - {test_labels[0]}"
    else:
        title += " - BT24 Test Split"

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


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
    plot_torque(test_labels, tau_true, tau_pred, dt_mean, output_path, args.max_samples)

    print("Model:", model_path)
    print("Labels:", test_labels)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
