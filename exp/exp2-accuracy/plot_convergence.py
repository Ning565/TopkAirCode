#!/usr/bin/env python3
"""Plot selected experiment-2 convergence curves from real evaluation points."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROFILES = (
    ("TOPK", "topk", "#1f77b4"),
    ("RANDK", "randk", "#ff7f0e"),
    ("FULL", "full", "#2ca02c"),
)


def ratio_tag(ratio: float) -> str:
    return f"{int(round(ratio * 1000)):03d}"


def read_eval_points(path: Path, eval_every: int) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            round_i = int(row["round"])
            if round_i == 1 or round_i % eval_every == 0:
                points.append((round_i, float(row["test_accuracy"])))
    if not points:
        raise ValueError(f"No evaluation points found in {path}")
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output")
    parser.add_argument("--topk-ratio", type=float, default=0.12)
    parser.add_argument("--randk-ratio", type=float, default=0.50)
    parser.add_argument("--eval-every", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.input_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ratios = {"topk": args.topk_ratio, "randk": args.randk_ratio, "full": 1.0}
    all_rows: list[dict[str, object]] = []

    plt.figure(figsize=(9.0, 5.6))
    for label, method, color in PROFILES:
        ratio = ratios[method]
        path = root / "runs" / f"{method}_r{ratio_tag(ratio)}" / "metrics_rounds.csv"
        points = read_eval_points(path, args.eval_every)
        plt.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=color,
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            label=f"{label} (k/d={ratio:.2f})",
        )
        all_rows.extend(
            {"method": method, "ratio": ratio, "round": round_i, "test_accuracy": accuracy}
            for round_i, accuracy in points
        )

    plt.xlabel("Communication rounds")
    plt.ylabel("Test accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()

    if args.csv_output:
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["method", "ratio", "round", "test_accuracy"])
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    main()
