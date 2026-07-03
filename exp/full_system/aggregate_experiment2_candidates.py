#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate experiment-2 candidate summaries across random seeds."""

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_ORDER = ("topk", "randk", "full")


def infer_seed(path: Path) -> Optional[int]:
    match = re.search(r"seed(\d+)", path.name)
    if match:
        return int(match.group(1))
    if path.name == "experiment2_candidate":
        return 2026
    return None


def discover_candidate_dirs(root: Path) -> List[Path]:
    candidates = []
    for path in sorted(root.glob("experiment2_candidate*")):
        if (path / "summary.json").exists():
            candidates.append(path)
    return candidates


def read_items(candidate_dirs: Iterable[Path]) -> List[Dict]:
    items = []
    for candidate_dir in candidate_dirs:
        summary_path = candidate_dir / "summary.json"
        with summary_path.open(encoding="utf-8") as handle:
            seed_items = json.load(handle)
        fallback_seed = infer_seed(candidate_dir)
        for item in seed_items:
            row = dict(item)
            row["source_dir"] = str(candidate_dir)
            row["seed"] = row.get("seed", fallback_seed)
            if row["seed"] is None:
                raise ValueError(f"Cannot infer seed for {candidate_dir}; rebuild with --seed.")
            items.append(row)
    return items


def sample_std(values: List[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def aggregate(items: List[Dict]) -> List[Dict]:
    result = []
    for method in METHOD_ORDER:
        sub = [row for row in items if row["method"] == method]
        if not sub:
            continue
        final_values = [float(row["final_accuracy"]) for row in sub]
        best_values = [float(row["best_accuracy"]) for row in sub]
        result.append({
            "method": method,
            "num_seeds": len({int(row["seed"]) for row in sub}),
            "seeds": sorted({int(row["seed"]) for row in sub}),
            "ratio_mean": mean(float(row["ratio"]) for row in sub),
            "final_accuracy_mean": mean(final_values),
            "final_accuracy_std": sample_std(final_values),
            "best_accuracy_mean": mean(best_values),
            "best_accuracy_std": sample_std(best_values),
            "b_star_mean": mean(float(row["b_star"]) for row in sub),
        })
    return result


def write_outputs(out_dir: Path, items: List[Dict], summary: List[Dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_candidate_dirs": len({row["source_dir"] for row in items}),
        "num_seeds": len({int(row["seed"]) for row in items}),
        "seeds": sorted({int(row["seed"]) for row in items}),
        "items": items,
        "summary": summary,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    with (out_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# 实验二多 seed 候选汇总\n\n")
        handle.write(
            "本汇总读取各 `experiment2_candidate*` 目录中的 `summary.json`，"
            "不手工改数值；每个候选目录应由真实训练 CSV 合并生成。\n\n"
        )
        handle.write(f"- seeds: `{', '.join(str(seed) for seed in payload['seeds'])}`\n")
        handle.write(f"- candidate dirs: `{payload['num_candidate_dirs']}`\n\n")
        if payload["num_seeds"] < 3:
            handle.write(
                "> 注意：当前少于 3 个 seed，只能作为候选趋势；正式论文建议继续补足多 seed 后报告均值和标准差。\n\n"
            )
        handle.write("| 方法 | seeds | k/d均值 | 第200轮准确率均值 | 第200轮准确率std | 最好准确率均值 | 最好准确率std | b*均值 |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            handle.write(
                f"| {row['method']} | {row['num_seeds']} | {row['ratio_mean']:.2f} | "
                f"{row['final_accuracy_mean']:.2f} | {row['final_accuracy_std']:.2f} | "
                f"{row['best_accuracy_mean']:.2f} | {row['best_accuracy_std']:.2f} | "
                f"{row['b_star_mean']:.3e} |\n"
            )

    x = list(range(len(summary)))
    labels = [row["method"].upper() for row in summary]
    means = [row["final_accuracy_mean"] for row in summary]
    errs = [row["final_accuracy_std"] for row in summary]
    plt.figure(figsize=(6.6, 4.4))
    plt.bar(x, means, yerr=errs if any(errs) else None, capsize=5, color=["#2b6cb0", "#38a169", "#805ad5"][: len(x)])
    plt.xticks(x, labels)
    plt.ylabel("Final test accuracy (%)")
    plt.title("FEMNIST strict DP/power: candidate seed summary")
    plt.grid(axis="y", linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(out_dir / "femnist_final_accuracy_multiseed.png", dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dirs", nargs="*", default=None)
    parser.add_argument("--root", default="exp/full_system")
    parser.add_argument("--output-dir", default="exp/full_system/experiment2_multiseed")
    args = parser.parse_args()

    root = Path(args.root)
    candidate_dirs = [Path(path) for path in args.candidate_dirs] if args.candidate_dirs else discover_candidate_dirs(root)
    if not candidate_dirs:
        raise FileNotFoundError(f"No experiment2_candidate* summary.json found under {root}")
    items = read_items(candidate_dirs)
    summary = aggregate(items)
    write_outputs(Path(args.output_dir), items, summary)


if __name__ == "__main__":
    main()
