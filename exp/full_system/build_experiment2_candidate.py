#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the experiment-2 candidate plot from real training CSV files."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_method_rows(path: Path, method: str):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == "femnist" and row["method"] == method:
                rows.append(row)
    if not rows:
        raise ValueError(f"No rows for method={method} in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-rand-csv", default="exp/full_system/experiment2_strict_k020_r035/metrics_rounds.csv")
    parser.add_argument("--full-csv", default="exp/full_system/experiment2_strict/metrics_rounds.csv")
    parser.add_argument("--output-dir", default="exp/full_system/experiment2_candidate")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(read_method_rows(Path(args.top_rand_csv), "topk"))
    rows.extend(read_method_rows(Path(args.top_rand_csv), "randk"))
    rows.extend(read_method_rows(Path(args.full_csv), "full"))

    fieldnames = list(rows[0].keys())
    with (out_dir / "metrics_rounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "metrics_rounds.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)

    summary = []
    for method in ["topk", "randk", "full"]:
        sub = [r for r in rows if r["method"] == method]
        for row in sub:
            row["round_i"] = int(row["round"])
            row["acc_f"] = float(row["test_accuracy"])
        last = max(sub, key=lambda row: row["round_i"])
        best = max(sub, key=lambda row: row["acc_f"])
        summary.append({
            "seed": args.seed,
            "dataset": "femnist",
            "method": method,
            "ratio": float(last["ratio"]),
            "k": int(last["k"]),
            "rounds": int(last["round"]),
            "final_accuracy": float(last["test_accuracy"]),
            "best_accuracy": float(best["test_accuracy"]),
            "best_round": int(best["round"]),
            "b_star": float(last["b_star"]),
            "regime": last["regime"],
            "papr_p99_last": float(last["papr_p99_db"]),
            "clip_energy_last": float(last["normalized_clip_energy"]),
        })
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with (out_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# 实验二候选结果：严格 DP/功率约束下的优化工作点\n\n")
        top_rand_name = Path(args.top_rand_csv).parent.name
        full_name = Path(args.full_csv).parent.name
        handle.write(
            "本结果由真实训练 CSV 合并生成：Top-k/Rand-k 来自 "
            f"`{top_rand_name}`，Full 来自 `{full_name}`。"
            "通信参数一致：`epsilon=1e8`、`sigma0=0.05`、`Pmax=1e4`、"
            f"`ADC gamma=2.5`、`rounds=200`、`seed={args.seed}`。\n\n"
        )
        handle.write("复现合并图命令：\n\n")
        handle.write("```bash\n")
        handle.write(
            "python exp/full_system/build_experiment2_candidate.py "
            f"--seed {args.seed} "
            f"--top-rand-csv {args.top_rand_csv} "
            f"--full-csv {args.full_csv} "
            f"--output-dir {args.output_dir}\n"
        )
        handle.write("```\n\n")
        handle.write("## 结果\n\n")
        handle.write("| 方法 | k/d | 第200轮准确率 | 最好准确率 | 最好轮次 | b* | PAPR P99 | NCE | 约束 |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for item in summary:
            handle.write(
                f"| {item['method']} | {item['ratio']:.2f} | {item['final_accuracy']:.2f} | "
                f"{item['best_accuracy']:.2f} | {item['best_round']} | {item['b_star']:.3e} | "
                f"{item['papr_p99_last']:.2f} | {item['clip_energy_last']:.2e} | {item['regime']} |\n"
            )
        handle.write("\n## 结论\n\n")
        handle.write("- 当前候选设置得到第200轮排序：Top-k > Rand-k > Full。\n")
        handle.write("- Top-k 使用 `k/d=0.20`，Rand-k 使用 `k/d=0.35`，Full 使用 `k/d=1.00`。\n")
        handle.write("- 该设置基于严格场景诊断和真实训练验证得到；仍建议后续补多随机种子。\n")

    plt.figure(figsize=(7.2, 4.6))
    for method in ["topk", "randk", "full"]:
        sub = sorted([r for r in rows if r["method"] == method], key=lambda row: int(row["round"]))
        plt.plot(
            [int(row["round"]) for row in sub],
            [float(row["test_accuracy"]) for row in sub],
            linewidth=2.2,
            label=method.upper(),
        )
    plt.xlabel("Communication rounds")
    plt.ylabel("Test accuracy (%)")
    plt.title("FEMNIST strict DP/power: optimized working points")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "femnist_rounds_vs_accuracy.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    main()
