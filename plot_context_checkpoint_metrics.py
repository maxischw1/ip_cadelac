from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_DIR = Path(__file__).resolve().parent
INPUT_CSV = REPO_DIR / "logs" / "exo_context_checkpoint_metrics.csv"
OUTPUT_DIR = REPO_DIR / "logs"


def load_metrics():
    # Open csv
    with open(INPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Sort all rows by epoch.
    # CSV values are read as strings, so epoch must be converted to int
    rows = sorted(rows, key=lambda row: int(row["epoch"]))

    epochs = [int(row["epoch"]) for row in rows]
    mse = [float(row["torque_mse"]) for row in rows]
    rmse = [float(row["torque_rmse"]) for row in rows]

    # Output as lists for plotting
    return epochs, mse, rmse

# Function for plotting
def save_plot(x, y, ylabel, title, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get csv contents
    epochs, mse, rmse = load_metrics()

    mse_path = OUTPUT_DIR / "exo_context_torque_mse_over_epochs.png"
    rmse_path = OUTPUT_DIR / "exo_context_torque_rmse_over_epochs.png"

    # Plot list contents
    # Save the torque MSE plot
    save_plot(
        epochs,
        mse,
        "Torque MSE",
        "Context-Aware Torque MSE over Epochs",
        mse_path,
    )

    save_plot(
        epochs,
        rmse,
        "Torque RMSE [Nm]",
        "Context-Aware Torque RMSE over Epochs",
        rmse_path,
    )

    print("Saved:", mse_path)
    print("Saved:", rmse_path)


if __name__ == "__main__":
    main()
