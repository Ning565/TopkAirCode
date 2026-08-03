#!/usr/bin/env python3
"""Summarize baseline and one-factor Experiment 1 applicability scans."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_run(run_dir: Path, family: str, value: str) -> list[dict[str, Any]]:
    rows = read_rows(run_dir / "bound_terms.csv")
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    output = []
    for method in ("topk", "randk", "full"):
        subset = [row for row in rows if row["method"] == method]
        best = min(subset, key=lambda row: float(row["bound_total"]))
        strict = min(subset, key=lambda row: float(row["strict_bound_total"]))
        output.append({
            "family": family,
            "value": value,
            "method": method,
            "best_design_ratio": float(best["ratio"]),
            "best_design_bound": float(best["bound_total"]),
            "best_strict_ratio": float(strict["ratio"]),
            "best_strict_bound": float(strict["strict_bound_total"]),
            "dominant_term": best["dominant_term"],
            "dominant_share": float(best["dominant_share"]),
            "eta_condition_satisfied": strict["strict_eta_condition_satisfied"],
            "learning_rate": cfg["lr_femnist"],
            "tx_coordinate_clip": cfg["element_clip"],
            "snr_max_db": cfg["snr_max_db"],
            "epsilon_total": cfg["epsilon"],
            "adc_backoff_db": cfg["adc_backoff_db"],
            "num_clients": cfg["num_clients"],
            "rounds": cfg["rounds"],
            "local_steps": cfg["local_steps_femnist"],
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = summarize_run(args.baseline, "baseline", "baseline")
    for csv_path in sorted(args.scan_root.glob("*/*/bound_terms.csv")):
        run_dir = csv_path.parent
        rows.extend(summarize_run(run_dir, run_dir.parent.name, run_dir.name))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "applicability_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (args.output_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write("# 实验一系统适用范围汇总\n\n")
        handle.write("所有环境均采用同一方法公平协议；环境扫描不用于为不同方法选择不同参数。`best_design_ratio` 是校准代理候选，`best_strict_ratio` 使用通用严格收缩系数。\n\n")
        handle.write("| 扫描族 | 数值 | 方法 | 校准候选 $p$ | 严格候选 $p$ | 主导项 | 主导占比 |\n")
        handle.write("|---|---:|---|---:|---:|---|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['family']} | {row['value']} | {row['method'].upper()} | "
                f"{row['best_design_ratio']:.2f} | {row['best_strict_ratio']:.2f} | "
                f"{row['dominant_term']} | {row['dominant_share']:.3f} |\n"
            )


if __name__ == "__main__":
    main()

