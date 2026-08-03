#!/usr/bin/env python3
"""Summarize per-round client-DP power/privacy transition scans."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(run_dir: Path, family: str, value: str) -> list[dict[str, Any]]:
    rows = read_csv(run_dir / "bound_terms.csv")
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    output = []
    for method in ("topk", "randk", "full"):
        subset = sorted((row for row in rows if row["method"] == method), key=lambda row: float(row["ratio"]))
        best = min(subset, key=lambda row: float(row["bound_total"]))
        first_power = next((row for row in subset if row["active_constraint"] == "power"), None)
        output.append({
            "family": family,
            "value": value,
            "epsilon_per_round": cfg["epsilon"],
            "snr_max_db": cfg["snr_max_db"],
            "method": method,
            "best_ratio": float(best["ratio"]),
            "best_bound": float(best["bound_total"]),
            "first_power_limited_ratio": float(first_power["ratio"]) if first_power else "",
            "best_active_constraint": best["active_constraint"],
            "dominant_term": best["dominant_term"],
            "eta_condition_satisfied": best["eta_condition_satisfied"],
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(args.baseline, "baseline", "baseline")
    for path in sorted(args.scan_root.glob("*/*/bound_terms.csv")):
        summary.extend(summarize(path.parent, path.parent.parent.name, path.parent.name))

    with (args.output_dir / "transition_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    baseline_rows = [row for row in read_csv(args.baseline / "bound_terms.csv") if row["method"] == "topk"]
    x = [float(row["ratio"]) for row in baseline_rows]
    plt.figure(figsize=(7.2, 4.7))
    plt.semilogy(x, [float(row["b_power"]) for row in baseline_rows], label=r"$b_{power}(p)$", linewidth=2.2)
    plt.semilogy(x, [float(row["b_privacy"]) for row in baseline_rows], label=r"$b_{privacy}$", linewidth=2.2)
    plt.semilogy(x, [float(row["b_star"]) for row in baseline_rows], label=r"$b^*(p)$", linewidth=2.5, linestyle="--")
    plt.xlabel("Sparsity retention k/d")
    plt.ylabel("Feasible aggregation gain")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "power_privacy_transition.png", dpi=240)
    plt.close()

    epsilon_rows = [
        row for row in summary
        if row["family"] in ("epsilon_per_round", "baseline") and row["method"] == "topk"
    ]
    epsilon_rows.sort(key=lambda row: float(row["epsilon_per_round"]))
    plt.figure(figsize=(7.2, 4.7))
    plt.plot(
        [float(row["epsilon_per_round"]) for row in epsilon_rows],
        [float(row["first_power_limited_ratio"]) if row["first_power_limited_ratio"] != "" else 1.05 for row in epsilon_rows],
        marker="o",
        linewidth=2.2,
    )
    plt.axhline(1.0, color="black", linestyle=":", linewidth=1.2)
    plt.xlabel(r"Per-round client privacy budget $\epsilon$")
    plt.ylabel("First power-limited k/d")
    plt.ylim(0.0, 1.1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.output_dir / "epsilon_vs_transition_ratio.png", dpi=240)
    plt.close()

    with (args.output_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write("# 单轮客户端级隐私约束过渡实验\n\n")
        handle.write("本实验采用 add/remove-client 单轮隐私、公共更新范数裁剪和固定完整 OFDM 资源栅格。单轮预算不声明为多轮组合隐私保证。\n\n")
        handle.write("| 扫描 | 数值 | 方法 | 最优 $p$ | 首个功率约束 $p$ | 最优点约束 |\n")
        handle.write("|---|---:|---|---:|---:|---|\n")
        for row in summary:
            crossing = row["first_power_limited_ratio"] if row["first_power_limited_ratio"] != "" else "无"
            handle.write(
                f"| {row['family']} | {row['value']} | {row['method'].upper()} | "
                f"{row['best_ratio']:.2f} | {crossing} | {row['best_active_constraint']} |\n"
            )


if __name__ == "__main__":
    main()

