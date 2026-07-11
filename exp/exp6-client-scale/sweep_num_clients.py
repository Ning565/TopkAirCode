#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 6: scalability with the number of participating clients.

This sweep keeps the validated experiment-2 system profile and changes only the
number of participating FEMNIST clients.  Each plotted point is produced by the
same end-to-end FL/AirComp/ADC simulator used in the previous experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROFILES = (
    ("topk_optimized", "topk", 0.15),
    ("topk_adc_unaware", "topk", 0.35),
    ("randk_optimized", "randk", 0.65),
    ("full", "full", 1.0),
)
COLORS = {
    "topk_optimized": "#1f77b4",
    "topk_adc_unaware": "#9467bd",
    "randk_optimized": "#ff7f0e",
    "full": "#2ca02c",
}
LABELS = {
    "topk_optimized": "ADC-aware TOPK",
    "topk_adc_unaware": "ADC-unaware TOPK",
    "randk_optimized": "RANDK",
    "full": "FULL",
}
MARKERS = {
    "topk_optimized": "o",
    "topk_adc_unaware": "s",
    "randk_optimized": "^",
    "full": "D",
}


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def run_dir(root: Path, label: str, num_clients: int) -> Path:
    return root / "runs" / f"N{num_clients:03d}_{label}"


def build_command(args, label: str, method: str, ratio: float, num_clients: int, out_dir: Path) -> list[str]:
    return [
        args.python,
        "exp/common/full_system.py",
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--datasets",
        "femnist",
        "--methods",
        method,
        "--num-clients",
        str(num_clients),
        "--rounds",
        str(args.rounds),
        "--eval-every",
        str(args.eval_every),
        "--lr-femnist",
        str(args.lr_femnist),
        "--lr-decay",
        str(args.lr_decay),
        "--min-lr",
        str(args.min_lr),
        "--optimizer-momentum",
        str(args.optimizer_momentum),
        "--optimizer-weight-decay",
        str(args.optimizer_weight_decay),
        "--ratio-overrides",
        f"femnist:{method}={ratio}",
        "--epsilon",
        str(args.epsilon),
        "--sigma0",
        str(args.sigma0),
        "--p-max",
        str(args.p_max),
        "--adc-backoff-gamma",
        str(args.adc_backoff_gamma),
        "--element-clip",
        str(args.element_clip),
        "--error-feedback-methods",
        args.error_feedback_methods,
        "--randk-mask-mode",
        args.randk_mask_mode,
        "--output-dir",
        str(out_dir),
    ]


def execute_jobs(args, jobs: list[dict[str, Any]]) -> None:
    for job in jobs:
        out_dir = job["out_dir"]
        if (out_dir / "summary.json").exists() and not args.force:
            print(f"[skip] {out_dir}", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(
            args,
            job["label"],
            job["method"],
            job["ratio"],
            job["num_clients"],
            out_dir,
        )
        print("[run] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def read_metrics(path: Path, eval_every: int, rounds: int, tail_window: int) -> dict[str, Any]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["round_i"] = int(row["round"])
            row["acc_f"] = float(row["test_accuracy"])
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty metrics file: {path}")

    ordered = sorted(rows, key=lambda row: row["round_i"])
    eval_rows = [
        row
        for row in ordered
        if row["round_i"] == 1 or row["round_i"] == rounds or row["round_i"] % eval_every == 0
    ]
    if not eval_rows:
        eval_rows = ordered
    last = eval_rows[-1]
    best = max(eval_rows, key=lambda row: row["acc_f"])
    tail = eval_rows[-max(1, min(tail_window, len(eval_rows))) :]
    active_resource_ratio = float(last["active_resource_ratio"])
    u_active_mean = float(last["u_active_mean"])
    num_clients = max(1.0, float(last.get("num_clients", 1.0)))
    effective_tx_ratio = active_resource_ratio * u_active_mean / num_clients
    return {
        "final_accuracy": float(last["test_accuracy"]),
        "tail_mean_accuracy": sum(row["acc_f"] for row in tail) / len(tail),
        "best_accuracy": float(best["test_accuracy"]),
        "best_round": int(best["round"]),
        "b_star": float(last["b_star"]),
        "regime": last["regime"],
        "clip_probability": float(last["clip_sample_ratio"]),
        "clip_energy": float(last["normalized_clip_energy"]),
        "resource_weighted_clip_stress": float(last["normalized_clip_energy"]) * effective_tx_ratio,
        "effective_noise_std": float(last["effective_noise_std"]),
        "papr_p99_db": float(last["papr_p99_db"]),
        "active_resource_ratio": active_resource_ratio,
        "u_active_mean": u_active_mean,
        "effective_tx_ratio": effective_tx_ratio,
    }


def collect(jobs: list[dict[str, Any]], eval_every: int, rounds: int, tail_window: int) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        metrics_path = job["out_dir"] / "metrics_rounds.csv"
        if not metrics_path.exists():
            print(f"[missing] {metrics_path}", flush=True)
            continue
        row = read_metrics(metrics_path, eval_every, rounds, tail_window)
        row.update(
            {
                "label": job["label"],
                "method": job["method"],
                "ratio": job["ratio"],
                "num_clients": job["num_clients"],
                "run_dir": str(job["out_dir"]),
            }
        )
        rows.append(row)
    order = {label: idx for idx, (label, _method, _ratio) in enumerate(PROFILES)}
    return sorted(rows, key=lambda row: (row["num_clients"], order[row["label"]]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict[str, Any]], metric: str, ylabel: str, filename: str, out_dir: Path, logy: bool = False) -> None:
    plt.figure(figsize=(7.2, 4.6))
    for label, _method, _ratio in PROFILES:
        sub = sorted([row for row in rows if row["label"] == label], key=lambda row: row["num_clients"])
        if not sub:
            continue
        x = [row["num_clients"] for row in sub]
        y = [row[metric] for row in sub]
        plotter = plt.semilogy if logy else plt.plot
        plotter(x, y, marker=MARKERS[label], linewidth=2.2, color=COLORS[label], label=LABELS[label])
    plt.xlabel("Number of participating clients N")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=260)
    plt.close()


def plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    specs = [
        ("final_accuracy", "Final test accuracy (%)", "exp6_num_clients_vs_final_accuracy.png", False),
        ("tail_mean_accuracy", "Tail mean test accuracy (%)", "exp6_num_clients_vs_tail_mean_accuracy.png", False),
        ("clip_probability", "Clipped sample ratio", "exp6_num_clients_vs_clip_probability.png", False),
        ("clip_energy", "Normalized clipping residual energy", "exp6_num_clients_vs_clip_energy.png", True),
        ("resource_weighted_clip_stress", "Resource-weighted clipping stress", "exp6_num_clients_vs_resource_weighted_clip_stress.png", True),
        ("effective_noise_std", "Effective aggregation noise std", "exp6_num_clients_vs_effective_noise.png", False),
    ]
    for metric, ylabel, filename, logy in specs:
        plot_metric(rows, metric, ylabel, filename, out_dir, logy)


def write_markdown(path: Path, rows: list[dict[str, Any]], args) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 实验六 参与客户端数量扩展性\n\n")
        handle.write("本实验固定实验二/三/五使用的在线工作点与训练配置，只改变参与客户端数量 N，验证系统扩展性和 ADC clipping/stress 随设备规模变化的趋势。\n\n")
        handle.write("## 设置\n\n")
        handle.write(f"- seed: `{args.seed}`\n")
        handle.write(f"- rounds: `{args.rounds}`\n")
        handle.write(f"- num clients: `{args.num_clients}`\n")
        handle.write(f"- epsilon: `{args.epsilon}`\n")
        handle.write(f"- sigma0: `{args.sigma0}`\n")
        handle.write(f"- Pmax: `{args.p_max}`\n")
        handle.write(f"- ADC gamma: `{args.adc_backoff_gamma}`\n")
        handle.write(f"- element clip: `{args.element_clip}`\n")
        handle.write("- workpoints: `ADC-aware Top-k=0.15`, `ADC-unaware Top-k=0.35`, `Rand-k=0.65`, `Full=1.0`\n\n")
        handle.write("## 结果\n\n")
        handle.write("| N | profile | k/d | final | tail | best | clip prob | clip energy | resource clip stress | eff noise | PAPR p99 | b* | regime |\n")
        handle.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            handle.write(
                f"| {row['num_clients']} | {row['label']} | {row['ratio']:.2f} | "
                f"{row['final_accuracy']:.2f} | {row['tail_mean_accuracy']:.2f} | {row['best_accuracy']:.2f} | "
                f"{row['clip_probability']:.3e} | {row['clip_energy']:.3e} | {row['resource_weighted_clip_stress']:.3e} | "
                f"{row['effective_noise_std']:.3e} | {row['papr_p99_db']:.2f} | {row['b_star']:.3e} | {row['regime']} |\n"
            )
        handle.write(
            "\n写作时不要声称 Top-k 在相同 k 下必然降低 PAPR；本实验强调在 ADC-aware 工作点下，Top-k 以更小的有效传输比例和更低的 resource-weighted clipping stress 获得更好的规模鲁棒性。\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--num-clients", default="8,16,32,64,96,128")
    parser.add_argument("--epsilon", type=float, default=3e8)
    parser.add_argument("--sigma0", type=float, default=0.03)
    parser.add_argument("--p-max", type=float, default=2.5e3)
    parser.add_argument("--adc-backoff-gamma", type=float, default=1.0)
    parser.add_argument("--element-clip", type=float, default=0.02)
    parser.add_argument("--error-feedback-methods", default="topk")
    parser.add_argument("--randk-mask-mode", choices=("common", "independent"), default="common")
    parser.add_argument("--lr-femnist", type=float, default=0.05)
    parser.add_argument("--lr-decay", type=float, default=0.992)
    parser.add_argument("--min-lr", type=float, default=0.005)
    parser.add_argument("--optimizer-momentum", type=float, default=0.9)
    parser.add_argument("--optimizer-weight-decay", type=float, default=1e-4)
    parser.add_argument("--tail-window", type=int, default=3)
    parser.add_argument("--output-dir", default="logs/experiments/exp6-client-scale/client_scale")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for num_clients in parse_ints(args.num_clients):
        for label, method, ratio in PROFILES:
            jobs.append(
                {
                    "label": label,
                    "method": method,
                    "ratio": ratio,
                    "num_clients": num_clients,
                    "out_dir": run_dir(out_dir, label, num_clients),
                }
            )
    with (out_dir / "jobs.json").open("w", encoding="utf-8") as handle:
        json.dump(jobs, handle, indent=2, ensure_ascii=False, default=str)
    if args.execute:
        execute_jobs(args, jobs)
    rows = collect(jobs, args.eval_every, args.rounds, args.tail_window)
    write_csv(out_dir / "client_scale_results.csv", rows)
    with (out_dir / "client_scale_results.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    plot(rows, out_dir)
    write_markdown(out_dir / "summary.md", rows, args)


if __name__ == "__main__":
    main()
