#!/usr/bin/env python3
import csv
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROWS = 28
COLS = 28
PIXELS = ROWS * COLS


def load_recon_csv(path):
    images = {"input": {}, "recon": {}}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kind = row["kind"]
            idx = int(row["index"])
            pixel = int(row["pixel"])
            value = float(row["value"])
            if idx not in images[kind]:
                images[kind][idx] = [0.0] * PIXELS
            images[kind][idx][pixel] = value

    keys = sorted(images["input"].keys())
    if not keys:
        raise ValueError(f"no reconstruction rows found in {path}")
    return keys, images


def render_grid(csv_path, output_path):
    keys, images = load_recon_csv(csv_path)
    n = len(keys)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.8), dpi=150)
    fig.patch.set_facecolor("#f6f2e8")

    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    for col, idx in enumerate(keys):
        inp = images["input"][idx]
        rec = images["recon"][idx]
        ax0 = axes[0][col]
        ax1 = axes[1][col]

        ax0.imshow([inp[r * COLS : (r + 1) * COLS] for r in range(ROWS)], cmap="bone", vmin=0.0, vmax=1.0)
        ax1.imshow([rec[r * COLS : (r + 1) * COLS] for r in range(ROWS)], cmap="bone", vmin=0.0, vmax=1.0)
        ax0.set_title(f"{idx}", fontsize=9)
        ax0.axis("off")
        ax1.axis("off")

    axes[0][0].set_ylabel("input", fontsize=10)
    axes[1][0].set_ylabel("recon", fontsize=10)
    fig.suptitle(f"MNIST Autoencoder Reconstructions: {os.path.basename(csv_path)}", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) not in (2, 3):
        print("usage: render_autoencoder_recon.py <recon.csv> [output.png]", file=sys.stderr)
        return 1

    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) == 3 else os.path.splitext(csv_path)[0] + ".png"
    render_grid(csv_path, output_path)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
