#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验一：真实 calibration 版目标函数扫描。

与 `run.py` 的 proxy 版本不同，本脚本用 FEMNIST/CNN 的真实 client update
估计论文优化式中的关键量：

  J_A(k)=Phi_A(k;bar_omega_A)+lambda_ch*a_A(k)/(b*(k))^2
         +lambda_adc*D_ADC,A(k;gamma)

其中：
- bar_omega_A(k) 来自真实 Top-k/Rand-k 保留能量；
- support overlap/rho/Omega 来自真实 mask；
- D_ADC,A(k;gamma) 来自 OFDM IFFT -> AGC/RMS -> clipping residual；
- b*(k)=min{B_P(k),B_epsilon(k)} 为闭式功率/隐私可行解。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = ROOT / "exp" / "common"
sys.path.insert(0, str(COMMON_DIR))

from full_system import (  # noqa: E402
    Config as BaseConfig,
    Experiment,
    OFDMAirCompADC,
    compress_update,
    elementwise_clip,
    load_femnist,
    load_mnist,
    make_model,
    power_privacy_limits,
)


@dataclass
class ObjectiveWeights:
    lambda_channel: float = 0.15
    lambda_adc: float = 0.50


def ratio_grid() -> list[float]:
    return [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.80, 1.00]


def normalize_by_method(rows: list[dict[str, Any]], key: str) -> None:
    for method in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == method]
        values = np.array([r[key] for r in sub], dtype=float)
        if key == "learning_cost":
            values = np.log1p(values)
        lo, hi = float(values.min()), float(values.max())
        for i, row in enumerate(sub):
            row[f"{key}_norm"] = 0.0 if hi - lo < 1e-12 else float((values[i] - lo) / (hi - lo))


def support_stats(masks: list[torch.Tensor], k: int) -> dict[str, float]:
    n = len(masks)
    u = torch.stack(masks, dim=0).float().sum(dim=0)
    active = u[u > 0]
    omega = float(u.pow(2).sum().item())
    denom = max(1.0, float(n * (n - 1) * max(1, k)))
    rho = max(0.0, (omega - n * k) / denom)
    return {
        "rho": rho,
        "omega_support": omega,
        "active_resource_ratio": float((u > 0).float().mean().item()),
        "u_active_mean": float(active.mean().item()) if active.numel() else 0.0,
        "u_active_p99": float(torch.quantile(active, 0.99).item()) if active.numel() else 0.0,
    }


def learning_cost(bar_omega: float) -> float:
    bar_omega = min(0.9999, max(1e-4, bar_omega))
    return 2.0 * (1.0 - bar_omega) * (2.0 - bar_omega) / (bar_omega * bar_omega)


def load_problem(cfg: BaseConfig, dataset: str):
    if dataset == "mnist":
        return load_mnist(cfg)
    if dataset == "femnist":
        return load_femnist(cfg)
    raise ValueError(dataset)


def collect_rows(cfg: BaseConfig, dataset: str, ratios: list[float], methods: list[str], calib_rounds: int) -> list[dict[str, Any]]:
    train, test, clients, num_classes = load_problem(cfg, dataset)
    exp = Experiment(cfg, dataset, train, test, clients, num_classes)
    model = make_model(dataset, num_classes).to(cfg.device)
    d = sum(p.numel() for p in model.parameters())
    channel = OFDMAirCompADC(cfg, d)
    lr = cfg.lr_femnist if dataset == "femnist" else cfg.lr_mnist
    local_steps = cfg.local_steps_femnist if dataset == "femnist" else cfg.local_steps_mnist

    method_ratios = {
        method: ([1.0] if method == "full" else ratios)
        for method in methods
    }
    raw_stats: dict[tuple[str, float], list[dict[str, float]]] = {
        (method, ratio): [] for method in methods for ratio in method_ratios[method]
    }
    for round_idx in range(calib_rounds):
        raw_updates = []
        for cid in range(cfg.num_clients):
            raw, _ = exp.client_delta(model, cid, lr, local_steps)
            raw_updates.append(raw)

        for method in methods:
            for ratio in method_ratios[method]:
                k = d if method == "full" else max(1, int(d * ratio))
                b_power, b_privacy, b_star, regime = power_privacy_limits(cfg, k, cfg.rounds)
                gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 10007 * round_idx + int(ratio * 1_000_000))
                common_idx = torch.randperm(d, generator=gen)[:k] if method == "randk" and k < d else None
                signals, masks, retained = [], [], []
                for raw in raw_updates:
                    sparse, mask, ret, _ = compress_update(raw, method, ratio, gen, common_idx)
                    sparse = elementwise_clip(sparse, cfg.element_clip)
                    signals.append(sparse)
                    masks.append(mask)
                    retained.append(ret)
                _, comm = channel.aggregate(signals, masks, b_star, round_idx)
                overlap = support_stats(masks, k)
                raw_stats[(method, ratio)].append({
                    "retained_energy": float(np.mean(retained)),
                    **overlap,
                    **comm,
                    "b_power": b_power,
                    "b_privacy": b_privacy,
                    "b_star": b_star,
                    "regime_is_power": 1.0 if regime == "power" else 0.0,
                })
        print(f"[real-calibration] {dataset} round {round_idx + 1}/{calib_rounds}", flush=True)

    rows: list[dict[str, Any]] = []
    for method in methods:
        for ratio in method_ratios[method]:
            k = d if method == "full" else max(1, int(d * ratio))
            b_power, b_privacy, b_star, regime = power_privacy_limits(cfg, k, cfg.rounds)
            sub = raw_stats[(method, ratio)]
            bar_omega = float(np.mean([x["retained_energy"] for x in sub]))
            active = float(np.mean([x["active_resource_ratio"] for x in sub]))
            adc = float(np.mean([x["normalized_clip_energy"] for x in sub]))
            rows.append({
                "dataset": dataset,
                "method": method,
                "ratio": ratio,
                "k": k,
                "d": d,
                "bar_omega": bar_omega,
                "learning_cost": learning_cost(bar_omega),
                "rho": float(np.mean([x["rho"] for x in sub])),
                "omega_support": float(np.mean([x["omega_support"] for x in sub])),
                "active_resource_ratio": active,
                "u_active_mean": float(np.mean([x["u_active_mean"] for x in sub])),
                "u_active_p99": float(np.mean([x["u_active_p99"] for x in sub])),
                "papr_mean_db": float(np.mean([x["papr_mean_db"] for x in sub])),
                "papr_p99_db": float(np.mean([x["papr_p99_db"] for x in sub])),
                "papr_max_db": float(np.mean([x["papr_max_db"] for x in sub])),
                "adc_cost": adc,
                "clip_sample_ratio": float(np.mean([x["clip_sample_ratio"] for x in sub])),
                "channel_cost": active * k / (b_star * b_star + 1e-18),
                "b_power": b_power,
                "b_privacy": b_privacy,
                "b_star": b_star,
                "regime": regime,
            })
    return rows


def compute_objective(rows: list[dict[str, Any]], weights: ObjectiveWeights) -> None:
    for key in ["learning_cost", "channel_cost", "adc_cost"]:
        normalize_by_method(rows, key)
    for row in rows:
        row["J_adc_aware"] = (
            row["learning_cost_norm"]
            + weights.lambda_channel * row["channel_cost_norm"]
            + weights.lambda_adc * row["adc_cost_norm"]
        )
        row["J_adc_unaware"] = row["learning_cost_norm"] + weights.lambda_channel * row["channel_cost_norm"]


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for method in ["topk", "randk"]:
        sub = [r for r in rows if r["method"] == method]
        best[f"{method}_adc_aware"] = min(sub, key=lambda r: r["J_adc_aware"])
    topk_sub = [r for r in rows if r["method"] == "topk"]
    best["topk_adc_unaware"] = min(topk_sub, key=lambda r: r["J_adc_unaware"])
    full = [r for r in rows if r["method"] == "full"]
    if full:
        best["full"] = full[0]
    return best


def trend_messages(rows: list[dict[str, Any]], best: dict[str, Any]) -> list[str]:
    messages = []
    for method in ["topk", "randk"]:
        sub = sorted([r for r in rows if r["method"] == method], key=lambda r: r["ratio"])
        learning = [r["learning_cost"] for r in sub]
        channel = [r["channel_cost"] for r in sub]
        adc = [r["adc_cost"] for r in sub]
        messages.append(f"{method}: learning下降={all(a >= b for a, b in zip(learning, learning[1:]))}")
        messages.append(f"{method}: channel上升={all(a <= b for a, b in zip(channel, channel[1:]))}")
        messages.append(f"{method}: ADC上升={all(a <= b for a, b in zip(adc, adc[1:]))}")
    messages.append(
        "k*: "
        f"Top-k={best['topk_adc_aware']['ratio']:.2f}, "
        f"Rand-k={best['randk_adc_aware']['ratio']:.2f}, "
        f"Top-k ADC-unaware={best['topk_adc_unaware']['ratio']:.2f}"
    )
    return messages


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_objective(rows: list[dict[str, Any]], best: dict[str, Any], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    for ax, method in zip(axes, ["topk", "randk"]):
        sub = sorted([r for r in rows if r["method"] == method], key=lambda r: r["ratio"])
        x = [r["ratio"] for r in sub]
        ax.plot(x, [r["learning_cost_norm"] for r in sub], marker="o", label="Learning")
        ax.plot(x, [r["channel_cost_norm"] for r in sub], marker="s", label="Channel noise")
        ax.plot(x, [r["adc_cost_norm"] for r in sub], marker="^", label="Measured ADC")
        ax.plot(x, [r["J_adc_aware"] for r in sub], marker="D", linewidth=2.3, label="Total objective")
        k_star = best[f"{method}_adc_aware"]["ratio"]
        ax.axvline(k_star, color="black", linestyle="--", linewidth=1.1)
        ax.set_title(f"{method.upper()} real-calibration objective")
        ax.set_xlabel("Compression ratio k/d")
        ax.grid(True, linestyle="--", alpha=0.38)
    axes[0].set_ylabel("Normalized value")
    axes[1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "real_experiment1_objective_decomposition.png", dpi=260)
    plt.close(fig)


def plot_topk_ablation(rows: list[dict[str, Any]], best: dict[str, Any], out_dir: Path) -> None:
    sub = sorted([r for r in rows if r["method"] == "topk"], key=lambda r: r["ratio"])
    x = [r["ratio"] for r in sub]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, [r["J_adc_aware"] for r in sub], marker="o", linewidth=2.2, label="ADC-aware Top-k")
    ax.plot(x, [r["J_adc_unaware"] for r in sub], marker="s", linewidth=2.0, label="ADC-unaware Top-k")
    for key, label in [("topk_adc_aware", "aware k*"), ("topk_adc_unaware", "unaware k*")]:
        ratio = best[key]["ratio"]
        ax.axvline(ratio, linestyle="--", linewidth=1.2)
        ax.text(ratio, ax.get_ylim()[1] * 0.95, label, ha="center", va="top", fontsize=9)
    ax.set_xlabel("Compression ratio k/d")
    ax.set_ylabel("Objective")
    ax.grid(True, linestyle="--", alpha=0.38)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "real_experiment1_topk_adc_ablation.png", dpi=260)
    plt.close(fig)


def write_summary(out_dir: Path, cfg: BaseConfig, weights: ObjectiveWeights, rows: list[dict[str, Any]], best: dict[str, Any]) -> None:
    payload = {
        "config": asdict(cfg),
        "weights": asdict(weights),
        "best": best,
        "trend_messages": trend_messages(rows, best),
    }
    (out_dir / "summary_real.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "summary_real.md").open("w", encoding="utf-8") as handle:
        handle.write("# 实验一真实 calibration 扫描\n\n")
        handle.write("本结果使用真实 client update、support mask 和 OFDM/IFFT/ADC 流程估计目标函数项。\n\n")
        handle.write("## 趋势检查\n\n")
        for msg in payload["trend_messages"]:
            handle.write(f"- {msg}\n")
        handle.write("\n## 搜索结果\n\n")
        handle.write("| 方法 | 搜索目标 | k*/d | b* | bar_omega | rho | PAPR P99 | ADC cost | J |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for key, label in [
            ("topk_adc_aware", "Top-k ADC-aware"),
            ("topk_adc_unaware", "Top-k ADC-unaware"),
            ("randk_adc_aware", "Rand-k ADC-aware"),
            ("full", "Full update"),
        ]:
            if key not in best:
                continue
            row = best[key]
            j_key = "J_adc_unaware" if key == "topk_adc_unaware" else "J_adc_aware"
            j_val = row.get(j_key, row.get("J_adc_aware", 0.0))
            handle.write(
                f"| {row['method']} | {label} | {row['ratio']:.2f} | {row['b_star']:.3e} | "
                f"{row['bar_omega']:.3f} | {row['rho']:.3f} | {row['papr_p99_db']:.2f} | "
                f"{row['adc_cost']:.2e} | {j_val:.3f} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="exp/exp1-ksearch/final")
    parser.add_argument("--dataset", default="femnist", choices=["mnist", "femnist"])
    parser.add_argument("--methods", default="topk,randk,full")
    parser.add_argument("--ratios", default="0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.35,0.5,0.65,0.8,1.0")
    parser.add_argument("--calib-rounds", type=int, default=8)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--lambda-channel", type=float, default=ObjectiveWeights.lambda_channel)
    parser.add_argument("--lambda-adc", type=float, default=ObjectiveWeights.lambda_adc)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=BaseConfig.epsilon)
    parser.add_argument("--delta", type=float, default=BaseConfig.delta)
    parser.add_argument("--sigma0", type=float, default=BaseConfig.sigma0)
    parser.add_argument("--h-th", type=float, default=BaseConfig.h_th)
    parser.add_argument("--p-max", type=float, default=BaseConfig.p_max)
    parser.add_argument("--eta-tau-c", type=float, default=BaseConfig.eta_tau_C)
    parser.add_argument("--adc-backoff-gamma", type=float, default=BaseConfig.adc_backoff_gamma)
    parser.add_argument("--ofdm-subcarriers", type=int, default=2000)
    args = parser.parse_args()

    cfg = BaseConfig(
        device=args.device,
        output_dir=args.output_dir,
        datasets=(args.dataset,),
        methods=tuple(x.strip() for x in args.methods.split(",") if x.strip()),
        num_clients=args.num_clients,
        rounds=args.rounds,
        epsilon=args.epsilon,
        delta=args.delta,
        sigma0=args.sigma0,
        h_th=args.h_th,
        p_max=args.p_max,
        eta_tau_C=args.eta_tau_c,
        adc_backoff_gamma=args.adc_backoff_gamma,
        ofdm_subcarriers=args.ofdm_subcarriers,
    )
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        cfg.device = "cpu"
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    methods = list(cfg.methods)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = ObjectiveWeights(lambda_channel=args.lambda_channel, lambda_adc=args.lambda_adc)

    rows = collect_rows(cfg, args.dataset, ratios, methods, args.calib_rounds)
    compute_objective(rows, weights)
    best = select_best(rows)
    write_csv(out_dir / "objective_terms_real.csv", rows)
    plot_objective(rows, best, out_dir)
    plot_topk_ablation(rows, best, out_dir)
    write_summary(out_dir, cfg, weights, rows, best)
    print(json.dumps({"output_dir": str(out_dir), "best": best, "messages": trend_messages(rows, best)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
