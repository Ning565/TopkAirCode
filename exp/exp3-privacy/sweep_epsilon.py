#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run and aggregate experiment-3 privacy-budget sweeps.

This experiment fixes the working ratios selected by the online compression
sweep, then varies epsilon.  It is meant to show the transition from privacy
bottleneck to convergence/power bottleneck without hand-picking weaker
baselines.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_ORDER = ("topk", "randk", "full")


def parse_values(text: str) -> List[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def value_tag(value: float) -> str:
    return f"{value:.0e}".replace("+", "").replace(".", "p")


def run_dir(root: Path, method: str, epsilon: float) -> Path:
    return root / "runs" / f"eps{value_tag(epsilon)}_{method}"


def ratio_for(args, method: str) -> float:
    if method == "topk":
        return args.topk_ratio
    if method == "randk":
        return args.randk_ratio
    return 1.0


def build_command(args, method: str, epsilon: float, out_dir: Path) -> List[str]:
    ratio = ratio_for(args, method)
    override = f"femnist:{method}={ratio}"
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
        override,
        "--epsilon",
        str(epsilon),
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
        "--eval-every",
        str(args.eval_every),
    ]


def execute_sweep(args, jobs: Iterable[Dict]) -> None:
    for job in jobs:
        out_dir = job["out_dir"]
        summary_path = out_dir / "summary.json"
        if summary_path.exists() and not args.force:
            print(f"[skip] {out_dir} already has summary.json", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(args, job["method"], job["epsilon"], out_dir)
        print("[run] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def read_metrics(path: Path, target_accuracy: float, tail_window: int) -> Dict:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["round_i"] = int(row["round"])
            row["acc_f"] = float(row["test_accuracy"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows in {path}")
    ordered = sorted(rows, key=lambda row: row["round_i"])
    last = ordered[-1]
    best = max(ordered, key=lambda row: row["acc_f"])
    tail = ordered[-max(1, min(tail_window, len(ordered))) :]
    tail_mean = sum(row["acc_f"] for row in tail) / len(tail)
    auc_mean = sum(row["acc_f"] for row in ordered) / len(ordered)
    hit_round: Optional[int] = None
    for row in ordered:
        if row["acc_f"] >= target_accuracy:
            hit_round = row["round_i"]
            break
    return {
        "dataset": last["dataset"],
        "method": last["method"],
        "ratio": float(last["ratio"]),
        "k": int(last["k"]),
        "rounds": int(last["round"]),
        "final_accuracy": float(last["test_accuracy"]),
        "tail_mean_accuracy": float(tail_mean),
        "auc_mean_accuracy": float(auc_mean),
        "best_accuracy": float(best["test_accuracy"]),
        "best_round": int(best["round"]),
        "round_to_target": hit_round,
        "target_accuracy": target_accuracy,
        "b_power": float(last["b_power"]),
        "b_privacy": float(last["b_privacy"]),
        "b_star": float(last["b_star"]),
        "effective_noise_std": float(last.get("effective_noise_std", 0.0)),
        "papr_p99_last": float(last["papr_p99_db"]),
        "clip_energy_last": float(last["normalized_clip_energy"]),
        "regime": last["regime"],
    }


def collect(jobs: Iterable[Dict], target_accuracy: float, tail_window: int) -> List[Dict]:
    rows = []
    for job in jobs:
        metrics_path = job["out_dir"] / "metrics_rounds.csv"
        if not metrics_path.exists():
            print(f"[missing] {metrics_path}", flush=True)
            continue
        row = read_metrics(metrics_path, target_accuracy, tail_window)
        row["epsilon"] = float(job["epsilon"])
        row["run_dir"] = str(job["out_dir"])
        rows.append(row)
    return sorted(rows, key=lambda row: (row["epsilon"], METHOD_ORDER.index(row["method"])))


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def transition_epsilon(args) -> float:
    log_delta = math.log(1.0 / args.delta)
    margin = 2.0 * args.h_th * math.sqrt(args.p_max) * math.sqrt(args.rounds) / args.sigma0
    return max(0.0, (margin + math.sqrt(log_delta)) ** 2 - log_delta)


def write_markdown(path: Path, rows: List[Dict], args) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 实验三隐私强度 sweep\n\n")
        handle.write("本 sweep 固定实验二在线搜索得到的工作点，只改变隐私预算 epsilon。\n\n")
        handle.write("## 设置\n\n")
        handle.write(f"- seed: `{args.seed}`\n")
        handle.write(f"- rounds: `{args.rounds}`\n")
        handle.write(f"- ratios: `topk={args.topk_ratio}, randk={args.randk_ratio}, full=1.0`\n")
        handle.write(
            f"- optimizer: `lr={args.lr_femnist}, decay={args.lr_decay}, min_lr={args.min_lr}, "
            f"momentum={args.optimizer_momentum}, weight_decay={args.optimizer_weight_decay}`\n"
        )
        handle.write(f"- sigma0: `{args.sigma0}`\n")
        handle.write(f"- Pmax: `{args.p_max}`\n")
        handle.write(f"- ADC gamma: `{args.adc_backoff_gamma}`\n")
        handle.write(f"- element clip: `{args.element_clip}`\n")
        handle.write(f"- error feedback methods: `{args.error_feedback_methods}`\n")
        handle.write(f"- Rand-k mask mode: `{args.randk_mask_mode}`\n")
        handle.write(f"- estimated privacy/power transition epsilon: `{transition_epsilon(args):.3e}`\n\n")

        handle.write("## 结果\n\n")
        handle.write("| epsilon | 方法 | k/d | final | tail | AUC均值 | best | best轮次 | 到达target轮次 | b* | eff-noise | 约束 |\n")
        handle.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            hit = "-" if row["round_to_target"] is None else str(row["round_to_target"])
            handle.write(
                f"| {row['epsilon']:.3e} | {row['method']} | {row['ratio']:.2f} | "
                f"{row['final_accuracy']:.2f} | {row['tail_mean_accuracy']:.2f} | "
                f"{row['auc_mean_accuracy']:.2f} | {row['best_accuracy']:.2f} | "
                f"{row['best_round']} | {hit} | {row['b_star']:.3e} | "
                f"{row['effective_noise_std']:.2e} | {row['regime']} |\n"
            )

        handle.write("\n结论必须按同一 epsilon 下的真实排序书写，不手工替换 baseline。\n")


def plot(rows: List[Dict], out_dir: Path) -> None:
    if not rows:
        return
    for metric, ylabel, filename in [
        ("final_accuracy", "Final test accuracy (%)", "epsilon_vs_final_accuracy.png"),
        ("tail_mean_accuracy", "Tail mean test accuracy (%)", "epsilon_vs_tail_mean_accuracy.png"),
        ("auc_mean_accuracy", "Mean accuracy over rounds (%)", "epsilon_vs_auc_mean_accuracy.png"),
        ("best_accuracy", "Best test accuracy (%)", "epsilon_vs_best_accuracy.png"),
        ("effective_noise_std", "Effective aggregation noise std", "epsilon_vs_effective_noise.png"),
    ]:
        plt.figure(figsize=(7.2, 4.5))
        for method in METHOD_ORDER:
            sub = [row for row in rows if row["method"] == method]
            if not sub:
                continue
            plt.semilogx(
                [row["epsilon"] for row in sub],
                [row[metric] for row in sub],
                marker="o",
                linewidth=2.0,
                label=method.upper(),
            )
        plt.xlabel("Privacy budget epsilon")
        plt.ylabel(ylabel)
        plt.grid(True, which="both", linestyle="--", alpha=0.45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr-femnist", type=float, default=0.05)
    parser.add_argument("--lr-decay", type=float, default=0.992)
    parser.add_argument("--min-lr", type=float, default=0.005)
    parser.add_argument("--optimizer-momentum", type=float, default=0.9)
    parser.add_argument("--optimizer-weight-decay", type=float, default=1e-4)
    parser.add_argument("--topk-ratio", type=float, default=0.15)
    parser.add_argument("--randk-ratio", type=float, default=0.65)
    parser.add_argument("--epsilons", default="1e5,3e5,1e6,3e6,1e7,3e7,1e8,3e8")
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--sigma0", type=float, default=0.03)
    parser.add_argument("--h-th", type=float, default=0.1)
    parser.add_argument("--p-max", type=float, default=2.5e3)
    parser.add_argument("--adc-backoff-gamma", type=float, default=1.2)
    parser.add_argument("--element-clip", type=float, default=0.02)
    parser.add_argument("--error-feedback-methods", default="topk")
    parser.add_argument("--randk-mask-mode", choices=("common", "independent"), default="common")
    parser.add_argument("--target-accuracy", type=float, default=75.0)
    parser.add_argument("--tail-window", type=int, default=25)
    parser.add_argument("--output-dir", default="logs/experiments/exp3-privacy/privacy_sweep")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict] = []
    for epsilon in parse_values(args.epsilons):
        for method in METHOD_ORDER:
            jobs.append({"method": method, "epsilon": epsilon, "out_dir": run_dir(out_dir, method, epsilon)})

    if args.execute:
        execute_sweep(args, jobs)

    rows = collect(jobs, args.target_accuracy, args.tail_window)
    with (out_dir / "privacy_sweep_results.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    write_csv(out_dir / "privacy_sweep_results.csv", rows)
    write_markdown(out_dir / "summary.md", rows, args)
    plot(rows, out_dir)


if __name__ == "__main__":
    main()
