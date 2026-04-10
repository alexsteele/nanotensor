#!/usr/bin/env python3
import csv
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path):
    epochs = []
    losses = []

    rows = []
    saw_header = False
    with open(path, newline="", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not saw_header:
                if stripped.startswith("epoch,"):
                    saw_header = True
                    rows.append(line)
                continue
            rows.append(line)

    reader = csv.DictReader(rows)
    for row in reader:
        epochs.append(int(row["epoch"]))
        losses.append(float(row["train_loss"]))

    if not epochs:
        raise ValueError(f"no metric rows found in {path}")
    return epochs, losses


def plot_loss(metrics_path, output_path):
    epochs, losses = load_metrics(metrics_path)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=140)
    fig.patch.set_facecolor("#f7f4ec")
    ax.set_facecolor("#fffdf8")

    ax.plot(
        epochs,
        losses,
        color="#d96f28",
        linewidth=2.6,
        marker="o",
        markersize=4.5,
    )

    ax.set_title(f"Training Loss: {os.path.basename(metrics_path)}", loc="left")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.grid(True, color="#ddd6c8", linewidth=0.8)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#6a6a6a")
    ax.spines["bottom"].set_color("#6a6a6a")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) not in (2, 3):
        print("usage: plot_training_loss.py <metrics.csv> [output.png]", file=sys.stderr)
        return 1

    metrics_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) == 3 else os.path.splitext(metrics_path)[0] + "_loss.png"
    plot_loss(metrics_path, output_path)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
