#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run and aggregate experiment-2 compression-ratio sweeps.

The sweep is intentionally explicit: each method/ratio is trained by
`exp/common/full_system.py`, then summarized from the generated CSV files.  This
keeps ratio selection auditable instead of hand-picking a weaker baseline.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_ORDER = ("topk", "randk", "full")


def parse_ratios(text: str) -> List[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def ratio_tag(ratio: float) -> str:
    return f"{int(round(ratio * 1000)):03d}"


def run_dir(root: Path, method: str, ratio: float) -> Path:
    return root / "runs" / f"{method}_r{ratio_tag(ratio)}"


def build_command(args, method: str, ratio: float, out_dir: Path) -> List[str]:
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
        cmd = build_command(args, job["method"], job["ratio"], out_dir)
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
    last = max(rows, key=lambda row: row["round_i"])
    best = max(rows, key=lambda row: row["acc_f"])
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
        "best_accuracy": float(best["test_accuracy"]),
        "tail_mean_accuracy": float(tail_mean),
        "auc_mean_accuracy": float(auc_mean),
        "best_round": int(best["round"]),
        "round_to_target": hit_round,
        "target_accuracy": target_accuracy,
        "b_star": float(last["b_star"]),
        "effective_noise_std": float(last.get("effective_noise_std", 0.0)),
        "papr_p99_last": float(last["papr_p99_db"]),
        "clip_energy_last": float(last["normalized_clip_energy"]),
        "regime": last["regime"],
    }


def collect(jobs: Iterable[Dict], target_accuracy: float, tail_window: int) -> List[Dict]:
    result = []
    for job in jobs:
        metrics_path = job["out_dir"] / "metrics_rounds.csv"
        if not metrics_path.exists():
            print(f"[missing] {metrics_path}", flush=True)
            continue
        row = read_metrics(metrics_path, target_accuracy, tail_window)
        row["run_dir"] = str(job["out_dir"])
        result.append(row)
    return sorted(result, key=lambda row: (METHOD_ORDER.index(row["method"]), row["ratio"]))


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def best_by_method(rows: List[Dict]) -> Dict[str, Dict]:
    best = {}
    for method in METHOD_ORDER:
        sub = [row for row in rows if row["method"] == method]
        if sub:
            best[method] = max(sub, key=lambda row: row["final_accuracy"])
    return best


def write_markdown(path: Path, rows: List[Dict], args) -> None:
    best = best_by_method(rows)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 实验二压缩率 sweep\n\n")
        handle.write("本 sweep 由真实训练输出聚合，目的是公开选择工作点，不用于手工削弱 baseline。\n\n")
        handle.write("## 设置\n\n")
        handle.write(f"- seed: `{args.seed}`\n")
        handle.write(f"- rounds: `{args.rounds}`\n")
        handle.write(
            f"- optimizer: `lr={args.lr_femnist}, decay={args.lr_decay}, min_lr={args.min_lr}, "
            f"momentum={args.optimizer_momentum}, weight_decay={args.optimizer_weight_decay}`\n"
        )
        handle.write(f"- epsilon: `{args.epsilon}`\n")
        handle.write(f"- sigma0: `{args.sigma0}`\n")
        handle.write(f"- Pmax: `{args.p_max}`\n")
        handle.write(f"- ADC gamma: `{args.adc_backoff_gamma}`\n")
        handle.write(f"- element clip: `{args.element_clip}`\n")
        handle.write(f"- error feedback methods: `{args.error_feedback_methods}`\n")
        handle.write(f"- Rand-k mask mode: `{args.randk_mask_mode}`\n")
        handle.write(f"- target accuracy for speed: `{args.target_accuracy}`\n\n")
        handle.write(f"- tail window for stability: `{args.tail_window}` logged rows\n\n")

        handle.write("## 全部结果\n\n")
        handle.write("| 方法 | k/d | final | tail | AUC均值 | best | best轮次 | 到达target轮次 | b* | eff-noise | NCE | 约束 |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            hit = "-" if row["round_to_target"] is None else str(row["round_to_target"])
            handle.write(
                f"| {row['method']} | {row['ratio']:.2f} | {row['final_accuracy']:.2f} | "
                f"{row['tail_mean_accuracy']:.2f} | {row['auc_mean_accuracy']:.2f} | "
                f"{row['best_accuracy']:.2f} | {row['best_round']} | {hit} | "
                f"{row['b_star']:.3e} | {row['effective_noise_std']:.2e} | "
                f"{row['clip_energy_last']:.2e} | {row['regime']} |\n"
            )

        handle.write("\n## 按 final accuracy 的每方法最优\n\n")
        handle.write("| 方法 | k/d | final | tail | AUC均值 | best | b* |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for method in METHOD_ORDER:
            if method not in best:
                continue
            row = best[method]
            handle.write(
                f"| {method} | {row['ratio']:.2f} | {row['final_accuracy']:.2f} | "
                f"{row['tail_mean_accuracy']:.2f} | {row['auc_mean_accuracy']:.2f} | "
                f"{row['best_accuracy']:.2f} | {row['b_star']:.3e} |\n"
            )
        handle.write("\n结论必须按本表实际排序书写；若 Rand-k 在自己的 sweep 中更优，不能手工选择更差的 Rand-k 作为主 baseline。\n")


def plot(rows: List[Dict], out_dir: Path) -> None:
    if not rows:
        return
    accuracy_metrics = {
        "final_accuracy",
        "tail_mean_accuracy",
        "auc_mean_accuracy",
        "best_accuracy",
    }
    for metric, ylabel, filename in [
        ("final_accuracy", "Final test accuracy (%)", "ratio_vs_final_accuracy.png"),
        ("tail_mean_accuracy", "Tail mean test accuracy (%)", "ratio_vs_tail_mean_accuracy.png"),
        ("auc_mean_accuracy", "Mean accuracy over rounds (%)", "ratio_vs_auc_mean_accuracy.png"),
        ("best_accuracy", "Best test accuracy (%)", "ratio_vs_best_accuracy.png"),
        ("b_star", "b*", "ratio_vs_bstar.png"),
    ]:
        plt.figure(figsize=(7.0, 4.4))
        for method in METHOD_ORDER:
            sub = [row for row in rows if row["method"] == method]
            if not sub:
                continue
            if method == "full" and len(sub) == 1 and metric in accuracy_metrics:
                value = sub[0][metric]
                plt.scatter(
                    [1.0],
                    [value],
                    color="#2ca02c",
                    marker="D",
                    s=55,
                    label="FULL (uncompressed)",
                    zorder=3,
                )
                continue
            plt.plot(
                [row["ratio"] for row in sub],
                [row[metric] for row in sub],
                marker="o",
                linewidth=2.0,
                label=method.upper(),
            )
        plt.xlabel("k/d")
        plt.ylabel(ylabel)
        if metric in accuracy_metrics:
            plt.xlim(0.05, 1.03)
        plt.grid(True, linestyle="--", alpha=0.45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run missing training jobs before aggregation.")
    parser.add_argument("--force", action="store_true", help="Re-run jobs even if summary.json exists.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr-femnist", type=float, default=0.05)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--optimizer-momentum", type=float, default=0.0)
    parser.add_argument("--optimizer-weight-decay", type=float, default=0.0)
    parser.add_argument("--topk-ratios", default="0.05,0.10,0.15,0.20,0.25,0.35,0.50,0.65")
    parser.add_argument("--randk-ratios", default="0.20,0.35,0.50,0.65,0.80")
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--epsilon", type=float, default=1e10)
    parser.add_argument("--sigma0", type=float, default=0.01)
    parser.add_argument("--p-max", type=float, default=1e6)
    parser.add_argument("--adc-backoff-gamma", type=float, default=1.5)
    parser.add_argument("--element-clip", type=float, default=0.02)
    parser.add_argument("--error-feedback-methods", default="topk,randk")
    parser.add_argument("--randk-mask-mode", choices=("common", "independent"), default="common")
    parser.add_argument("--target-accuracy", type=float, default=75.0)
    parser.add_argument("--tail-window", type=int, default=5)
    parser.add_argument("--output-dir", default="logs/experiments/exp2.0-onlinesearch/ratio_sweep")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict] = []
    for ratio in parse_ratios(args.topk_ratios):
        jobs.append({"method": "topk", "ratio": ratio, "out_dir": run_dir(out_dir, "topk", ratio)})
    for ratio in parse_ratios(args.randk_ratios):
        jobs.append({"method": "randk", "ratio": ratio, "out_dir": run_dir(out_dir, "randk", ratio)})
    if args.include_full:
        jobs.append({"method": "full", "ratio": 1.0, "out_dir": run_dir(out_dir, "full", 1.0)})

    if args.execute:
        execute_sweep(args, jobs)

    rows = collect(jobs, args.target_accuracy, args.tail_window)
    with (out_dir / "sweep_results.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    write_csv(out_dir / "sweep_results.csv", rows)
    write_markdown(out_dir / "summary.md", rows, args)
    plot(rows, out_dir)


if __name__ == "__main__":
    main()
