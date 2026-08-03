#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 5: ADC-aware sparsity-selection ablation.

This sweep answers a specific reviewer question: does the ADC term in the
one-dimensional k-search actually matter?  We compare the online performance of
the ADC-aware Top-k workpoint against a Top-k workpoint selected without the ADC
term, while also reporting Rand-k and Full baselines under the same gamma sweep.
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
    ("topk_adc_aware", "topk", 0.15),
    ("topk_adc_unaware", "topk", 0.35),
    ("randk", "randk", 0.65),
    ("full", "full", 1.0),
)
COLORS = {
    "topk_adc_aware": "#1f77b4",
    "topk_adc_unaware": "#9467bd",
    "randk": "#ff7f0e",
    "full": "#2ca02c",
}
LABELS = {
    "topk_adc_aware": "ADC-aware TOPK",
    "topk_adc_unaware": "ADC-unaware TOPK",
    "randk": "RANDK",
    "full": "FULL",
}


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def gamma_tag(gamma: float) -> str:
    return str(gamma).replace(".", "p")


def run_dir(root: Path, label: str, gamma: float) -> Path:
    return root / "runs" / f"gamma{gamma_tag(gamma)}_{label}"


def build_command(args, label: str, method: str, ratio: float, gamma: float, out_dir: Path) -> list[str]:
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
        str(gamma),
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
        cmd = build_command(args, job["label"], job["method"], job["ratio"], job["gamma"], out_dir)
        print("[run] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def read_metrics(path: Path, tail_window: int) -> dict[str, Any]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["round_i"] = int(row["round"])
            row["acc_f"] = float(row["test_accuracy"])
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty metrics file: {path}")
    ordered = sorted(rows, key=lambda row: row["round_i"])
    last = ordered[-1]
    best = max(ordered, key=lambda row: row["acc_f"])
    tail = ordered[-max(1, min(tail_window, len(ordered))) :]
    return {
        "final_accuracy": float(last["test_accuracy"]),
        "tail_mean_accuracy": sum(row["acc_f"] for row in tail) / len(tail),
        "best_accuracy": float(best["test_accuracy"]),
        "best_round": int(best["round"]),
        "b_star": float(last["b_star"]),
        "regime": last["regime"],
        "papr_p99_last": float(last["papr_p99_db"]),
        "clip_energy_last": float(last["normalized_clip_energy"]),
        "clip_sample_ratio_last": float(last["clip_sample_ratio"]),
        "effective_noise_std": float(last["effective_noise_std"]),
    }


def collect(jobs: list[dict[str, Any]], tail_window: int) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        metrics_path = job["out_dir"] / "metrics_rounds.csv"
        if not metrics_path.exists():
            print(f"[missing] {metrics_path}", flush=True)
            continue
        row = read_metrics(metrics_path, tail_window)
        row.update(
            {
                "label": job["label"],
                "method": job["method"],
                "ratio": job["ratio"],
                "gamma": job["gamma"],
                "run_dir": str(job["out_dir"]),
            }
        )
        rows.append(row)
    order = {label: idx for idx, (label, _method, _ratio) in enumerate(PROFILES)}
    return sorted(rows, key=lambda row: (row["gamma"], order[row["label"]]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    for metric, ylabel, filename, logy in [
        ("final_accuracy", "Final test accuracy (%)", "exp5_gamma_vs_final_accuracy.png", False),
        ("tail_mean_accuracy", "Tail mean test accuracy (%)", "exp5_gamma_vs_tail_mean_accuracy.png", False),
        ("clip_energy_last", "Normalized clipping residual energy", "exp5_gamma_vs_clip_energy.png", True),
        ("effective_noise_std", "Effective aggregation noise std", "exp5_gamma_vs_effective_noise.png", False),
    ]:
        plt.figure(figsize=(7.2, 4.6))
        for label, _method, _ratio in PROFILES:
            sub = sorted([row for row in rows if row["label"] == label], key=lambda row: row["gamma"])
            if not sub:
                continue
            if logy:
                plt.semilogy([row["gamma"] for row in sub], [row[metric] for row in sub], marker="o", linewidth=2.2, color=COLORS[label], label=LABELS[label])
            else:
                plt.plot([row["gamma"] for row in sub], [row[metric] for row in sub], marker="o", linewidth=2.2, color=COLORS[label], label=LABELS[label])
        plt.xlabel("ADC backoff threshold gamma")
        plt.ylabel(ylabel)
        plt.grid(True, which="both", linestyle="--", alpha=0.45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=260)
        plt.close()


def write_markdown(path: Path, rows: list[dict[str, Any]], args) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 实验五 ADC-aware 稀疏率选择消融\n\n")
        handle.write("本实验检验 ADC 项是否真的影响 Top-k 工作点选择。ADC-aware Top-k 使用实验二真实在线工作点，ADC-unaware Top-k 使用实验一去掉 ADC 项后的搜索点。\n\n")
        handle.write("## 设置\n\n")
        handle.write(f"- seed: `{args.seed}`\n")
        handle.write(f"- rounds: `{args.rounds}`\n")
        handle.write(f"- gammas: `{args.gammas}`\n")
        handle.write(f"- epsilon: `{args.epsilon}`\n")
        handle.write(f"- sigma0: `{args.sigma0}`\n")
        handle.write(f"- Pmax: `{args.p_max}`\n")
        handle.write("- workpoints: `ADC-aware Top-k=0.15`, `ADC-unaware Top-k=0.35`, `Rand-k=0.65`, `Full=1.0`\n\n")
        handle.write("## 结果\n\n")
        handle.write("| gamma | profile | k/d | final | tail | best | b* | clip energy | PAPR p99 | regime |\n")
        handle.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            handle.write(
                f"| {row['gamma']:.2f} | {row['label']} | {row['ratio']:.2f} | "
                f"{row['final_accuracy']:.2f} | {row['tail_mean_accuracy']:.2f} | {row['best_accuracy']:.2f} | "
                f"{row['b_star']:.3e} | {row['clip_energy_last']:.3e} | {row['papr_p99_last']:.2f} | {row['regime']} |\n"
            )
        handle.write("\n主文重点观察小 gamma 区域：若 ADC-aware Top-k 在更强 clipping 下更稳，则说明 ADC-aware k 选择有必要。\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--gammas", default="0.8,1.0,1.2,1.3,1.5,2.0")
    parser.add_argument("--epsilon", type=float, default=3e8)
    parser.add_argument("--sigma0", type=float, default=0.03)
    parser.add_argument("--p-max", type=float, default=2.5e3)
    parser.add_argument("--element-clip", type=float, default=0.02)
    parser.add_argument("--error-feedback-methods", default="topk")
    parser.add_argument("--randk-mask-mode", choices=("common", "independent"), default="common")
    parser.add_argument("--lr-femnist", type=float, default=0.05)
    parser.add_argument("--lr-decay", type=float, default=0.992)
    parser.add_argument("--min-lr", type=float, default=0.005)
    parser.add_argument("--optimizer-momentum", type=float, default=0.9)
    parser.add_argument("--optimizer-weight-decay", type=float, default=1e-4)
    parser.add_argument("--tail-window", type=int, default=25)
    parser.add_argument("--output-dir", default="logs/experiments/exp5-adc-aware/adc_robustness")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for gamma in parse_floats(args.gammas):
        for label, method, ratio in PROFILES:
            jobs.append({"label": label, "method": method, "ratio": ratio, "gamma": gamma, "out_dir": run_dir(out_dir, label, gamma)})
    if args.execute:
        execute_jobs(args, jobs)
    rows = collect(jobs, args.tail_window)
    write_csv(out_dir / "adc_robustness_results.csv", rows)
    with (out_dir / "adc_robustness_results.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    plot(rows, out_dir)
    write_markdown(out_dir / "summary.md", rows, args)


if __name__ == "__main__":
    main()
