#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run auditable experiment-2 parameter profiles and rank their online sweeps.

The profiles are deliberately method-transparent.  They tune global privacy,
ADC, clipping, and error-feedback choices, then let each method select its best
ratio from the same sweep table.  A useful profile should satisfy three checks:

1. Top-k's best final/tail/AUC accuracy is above Rand-k.
2. Rand-k's best ratio stays near the offline k-search optimum, about 0.5.
3. Full is worse than Rand-k under the same channel/privacy profile.
"""

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


METHOD_ORDER = ("topk", "randk", "full")


@dataclass(frozen=True)
class Profile:
    label: str
    epsilon: float
    adc_gamma: float
    element_clip: float
    error_feedback_methods: str
    randk_mask_mode: str = "common"
    p_max: Optional[float] = None


DEFAULT_PROFILES = {
    # Current successful exp1 profile, retained as the control.
    "control_eps1e10_g1.5_clip0.02_efboth": Profile(
        "control_eps1e10_g1.5_clip0.02_efboth", 1e10, 1.5, 0.02, "topk,randk"
    ),
    # Stronger privacy bottleneck should penalize high-k Rand-k/Full more.
    "eps3e9_g1.5_clip0.02_efboth": Profile(
        "eps3e9_g1.5_clip0.02_efboth", 3e9, 1.5, 0.02, "topk,randk"
    ),
    "eps1e9_g1.5_clip0.02_efboth": Profile(
        "eps1e9_g1.5_clip0.02_efboth", 1e9, 1.5, 0.02, "topk,randk"
    ),
    "eps5e8_g1.5_clip0.02_efboth": Profile(
        "eps5e8_g1.5_clip0.02_efboth", 5e8, 1.5, 0.02, "topk,randk"
    ),
    # More ADC pressure tests whether dense/high-k waveforms lose stability.
    "eps1e9_g1.3_clip0.02_efboth": Profile(
        "eps1e9_g1.3_clip0.02_efboth", 1e9, 1.3, 0.02, "topk,randk"
    ),
    # Slightly looser clipping may help Top-k's selected large coordinates.
    "eps1e9_g1.5_clip0.025_efboth": Profile(
        "eps1e9_g1.5_clip0.025_efboth", 1e9, 1.5, 0.025, "topk,randk"
    ),
    # Mechanism alignment: the offline Rand-k model has no residual memory.
    "eps1e9_g1.5_clip0.02_eftopk": Profile(
        "eps1e9_g1.5_clip0.02_eftopk", 1e9, 1.5, 0.02, "topk"
    ),
}


def profile_from_spec(spec: str) -> Profile:
    """Parse label:epsilon:gamma:clip:ef_methods[:mask_mode][:pmax].

    If the optional sixth field is numeric, it is interpreted as pmax and the
    mask mode defaults to common. This keeps older specs compatible.
    """
    if spec in DEFAULT_PROFILES:
        return DEFAULT_PROFILES[spec]
    parts = spec.split(":")
    if len(parts) not in (5, 6, 7):
        raise ValueError(
            "Profile spec must be a built-in label or "
            "label:epsilon:gamma:clip:ef_methods[:mask_mode][:pmax]"
        )
    label, eps, gamma, clip, ef = parts[:5]
    mask = "common"
    p_max: Optional[float] = None
    if len(parts) == 6:
        if parts[5] in ("common", "independent"):
            mask = parts[5]
        else:
            p_max = float(parts[5])
    elif len(parts) == 7:
        mask = parts[5]
        p_max = float(parts[6])
    return Profile(label, float(eps), float(gamma), float(clip), ef.replace("+", ","), mask, p_max)


def parse_profiles(text: str) -> List[Profile]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names or names == ["default"]:
        return list(DEFAULT_PROFILES.values())
    return [profile_from_spec(item) for item in names]


def safe_ef(text: str) -> str:
    return text.replace(",", "+")


def run_sweep(args, profile: Profile, out_dir: Path) -> None:
    cmd = [
        args.python,
        "exp/exp2.0-onlinesearch/sweep_ratios.py",
        "--execute",
        "--python",
        args.python,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
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
        "--eval-every",
        str(args.eval_every),
        "--topk-ratios",
        args.topk_ratios,
        "--randk-ratios",
        args.randk_ratios,
        "--include-full",
        "--epsilon",
        str(profile.epsilon),
        "--sigma0",
        str(args.sigma0),
        "--p-max",
        str(profile.p_max if profile.p_max is not None else args.p_max),
        "--adc-backoff-gamma",
        str(profile.adc_gamma),
        "--element-clip",
        str(profile.element_clip),
        "--error-feedback-methods",
        profile.error_feedback_methods,
        "--randk-mask-mode",
        profile.randk_mask_mode,
        "--target-accuracy",
        str(args.target_accuracy),
        "--tail-window",
        str(args.tail_window),
        "--output-dir",
        str(out_dir),
    ]
    if args.force:
        cmd.append("--force")
    print("[profile] " + profile.label, flush=True)
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in [
            "ratio",
            "final_accuracy",
            "best_accuracy",
            "tail_mean_accuracy",
            "auc_mean_accuracy",
            "b_star",
            "effective_noise_std",
        ]:
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def best_row(rows: Iterable[Dict], method: str, metric: str = "final_accuracy") -> Optional[Dict]:
    sub = [row for row in rows if row.get("method") == method]
    if not sub:
        return None
    return max(sub, key=lambda row: float(row.get(metric, 0.0)))


def summarize_profile(profile: Profile, out_dir: Path) -> Dict:
    rows = read_rows(out_dir / "sweep_results.csv")
    topk = best_row(rows, "topk")
    randk = best_row(rows, "randk")
    full = best_row(rows, "full")
    if not (topk and randk and full):
        return {
            "profile": profile.label,
            "status": "incomplete",
            "output_dir": str(out_dir),
        }

    topk_tail = float(topk.get("tail_mean_accuracy", topk["final_accuracy"]))
    randk_tail = float(randk.get("tail_mean_accuracy", randk["final_accuracy"]))
    full_tail = float(full.get("tail_mean_accuracy", full["final_accuracy"]))
    topk_auc = float(topk.get("auc_mean_accuracy", topk["final_accuracy"]))
    randk_auc = float(randk.get("auc_mean_accuracy", randk["final_accuracy"]))
    full_auc = float(full.get("auc_mean_accuracy", full["final_accuracy"]))

    final_gap = topk["final_accuracy"] - randk["final_accuracy"]
    tail_gap = topk_tail - randk_tail
    auc_gap = topk_auc - randk_auc
    randk_full_gap = randk["final_accuracy"] - full["final_accuracy"]
    rand_ratio_penalty = abs(randk["ratio"] - 0.5)
    top_ratio_penalty = abs(topk["ratio"] - 0.15)
    order_ok = final_gap > 0 and randk_full_gap > 0
    rand_near_exp1 = abs(randk["ratio"] - 0.5) <= 0.15

    score = (
        2.0 * final_gap
        + 1.0 * tail_gap
        + 0.5 * auc_gap
        + 0.5 * randk_full_gap
        - 4.0 * rand_ratio_penalty
        - 1.0 * top_ratio_penalty
    )
    if not order_ok:
        score -= 5.0
    if not rand_near_exp1:
        score -= 3.0
    return {
        "profile": profile.label,
        "status": "complete",
        "epsilon": profile.epsilon,
        "adc_gamma": profile.adc_gamma,
        "element_clip": profile.element_clip,
        "p_max": profile.p_max if profile.p_max is not None else None,
        "error_feedback_methods": profile.error_feedback_methods,
        "randk_mask_mode": profile.randk_mask_mode,
        "topk_ratio": topk["ratio"],
        "topk_final": topk["final_accuracy"],
        "topk_tail": topk_tail,
        "topk_auc": topk_auc,
        "randk_ratio": randk["ratio"],
        "randk_final": randk["final_accuracy"],
        "randk_tail": randk_tail,
        "randk_auc": randk_auc,
        "full_final": full["final_accuracy"],
        "full_tail": full_tail,
        "full_auc": full_auc,
        "topk_minus_randk_final": final_gap,
        "topk_minus_randk_tail": tail_gap,
        "topk_minus_randk_auc": auc_gap,
        "randk_minus_full_final": randk_full_gap,
        "order_ok": order_ok,
        "randk_near_exp1": rand_near_exp1,
        "score": score,
        "output_dir": str(out_dir),
    }


def write_summary(root: Path, summaries: List[Dict]) -> None:
    summaries = sorted(summaries, key=lambda row: float(row.get("score", -9999.0)), reverse=True)
    if summaries:
        with (root / "tuning_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    with (root / "tuning_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False)
    with (root / "tuning_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Experiment 2 Profile Tuning\n\n")
        handle.write("Profiles tune global communication/privacy settings and are ranked from real online sweeps.\n\n")
        handle.write(
            "| rank | profile | Top-k | Rand-k | Full | Top-Rand | Rand-Full | order | Rand near 0.5 | Pmax | score |\n"
        )
        handle.write("|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|\n")
        for i, row in enumerate(summaries, start=1):
            if row.get("status") != "complete":
                handle.write(f"| {i} | {row['profile']} | - | - | - | - | - | incomplete | - | - | - |\n")
                continue
            pmax_text = "-" if row.get("p_max") in (None, "") else f"{float(row['p_max']):.3g}"
            handle.write(
                f"| {i} | {row['profile']} | "
                f"{row['topk_final']:.2f} @ {row['topk_ratio']:.2f} | "
                f"{row['randk_final']:.2f} @ {row['randk_ratio']:.2f} | "
                f"{row['full_final']:.2f} | "
                f"{row['topk_minus_randk_final']:.2f} | "
                f"{row['randk_minus_full_final']:.2f} | "
                f"{row['order_ok']} | {row['randk_near_exp1']} | {pmax_text} | {row['score']:.2f} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr-femnist", type=float, default=0.05)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--optimizer-momentum", type=float, default=0.0)
    parser.add_argument("--optimizer-weight-decay", type=float, default=0.0)
    parser.add_argument("--topk-ratios", default="0.05,0.10,0.15,0.20,0.25,0.35")
    parser.add_argument("--randk-ratios", default="0.35,0.50,0.65,0.80")
    parser.add_argument("--profiles", default="default")
    parser.add_argument("--sigma0", type=float, default=0.01)
    parser.add_argument("--p-max", type=float, default=1e6)
    parser.add_argument("--target-accuracy", type=float, default=75.0)
    parser.add_argument("--tail-window", type=int, default=5)
    parser.add_argument("--output-root", default="logs/experiments/exp2.0-onlinesearch/profile_tuning")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    profiles = parse_profiles(args.profiles)
    summaries: List[Dict] = []
    for profile in profiles:
        profile_dir = root / f"seed{args.seed}_r{args.rounds}_{profile.label}"
        run_sweep(args, profile, profile_dir)
        summaries.append(summarize_profile(profile, profile_dir))
        write_summary(root, summaries)
    write_summary(root, summaries)
    print(f"[done] summary: {root / 'tuning_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
