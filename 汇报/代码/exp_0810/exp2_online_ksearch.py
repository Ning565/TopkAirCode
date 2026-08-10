#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 2 (new scenario, 2026-08-10): ONLINE optimal-compression search.

Whereas experiment 1 selects k*/d offline from the calibrated convergence
bound, experiment 2 searches the compression ratio ONLINE: every candidate
(method, ratio) is trained end-to-end through the full physical chain
(fixed participation, per-round b_t^*(k), physical sigma_sc, oversampled
waveform, ideal per-round RMS-AGC, radial clipping), and the working point
is selected from the measured accuracy itself.

The sweep is auditable by construction: each run writes its own per-round
metrics CSV under runs/<method>_rXXX/, and the aggregation step only reads
those files.  Finished runs are skipped unless --force is given, so the
sweep can be resumed after interruption.

If --exp1-summary points to the experiment-1 summary.json, the offline k*
of each method is drawn as a dashed vertical line on the ratio-accuracy
curves, which closes the "offline bound search vs online accuracy search"
comparison required by the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from full_system_0810 import FLSystem, LearnConfig, PhyConfig  # noqa: E402


METHOD_ORDER = ("topk", "randk", "full")


def ratio_tag(ratio: float) -> str:
    return f"{int(round(ratio * 1000)):04d}"


def parse_ratios(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def run_one(system: FLSystem, method: str, ratio: float, out_dir: Path, force: bool) -> Path:
    """Train one (method, ratio) candidate and persist per-round metrics."""
    metrics_path = out_dir / "metrics_rounds.csv"
    if metrics_path.exists() and not force:
        print(f"[skip] {out_dir} already finished", flush=True)
        return metrics_path
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = system.run_training(method, ratio, log_prefix="[exp2] ")
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    last = rows[-1]
    (out_dir / "run_summary.json").write_text(json.dumps({
        "method": method,
        "ratio": ratio,
        "final_accuracy": last["test_accuracy"],
        "rounds": last["round"],
        "b_star_last": last["b_star"],
        "regime_last": last["regime"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics_path


def read_run(metrics_path: Path, target_accuracy: float, tail_window: int) -> Dict:
    rows = []
    with metrics_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["round_i"] = int(row["round"])
            row["acc_f"] = float(row["test_accuracy"])
            rows.append(row)
    ordered = sorted(rows, key=lambda r: r["round_i"])
    last = ordered[-1]
    best = max(ordered, key=lambda r: r["acc_f"])
    tail = ordered[-max(1, min(tail_window, len(ordered))):]
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
        "rounds": last["round_i"],
        "final_accuracy": float(last["acc_f"]),
        "best_accuracy": float(best["acc_f"]),
        "best_round": best["round_i"],
        "tail_mean_accuracy": sum(r["acc_f"] for r in tail) / len(tail),
        "auc_mean_accuracy": sum(r["acc_f"] for r in ordered) / len(ordered),
        "round_to_target": hit_round,
        "target_accuracy": target_accuracy,
        "b_star_mean": sum(float(r["b_star"]) for r in ordered) / len(ordered),
        "regime_last": last["regime"],
        "papr_p99_last_db": float(last["papr_p99_db"]),
        "rho_clip_last": float(last["rho_clip"]),
        "nmse_total_last": float(last["nmse_total"]),
        "eff_noise_std_last": float(last["eff_noise_std"]),
    }


def load_exp1_kstar(path: str) -> Dict[str, float]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[warn] exp1 summary not found: {p}", flush=True)
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    best = data.get("best_bound", {})
    return {m: float(best[m]["ratio"]) for m in best if m in ("topk", "randk")}


def best_by_method(rows: List[Dict], key: str = "final_accuracy") -> Dict[str, Dict]:
    best = {}
    for method in METHOD_ORDER:
        sub = [r for r in rows if r["method"] == method]
        if sub:
            best[method] = max(sub, key=lambda r: r[key])
    return best


def plot_sweep(rows: List[Dict], kstar: Dict[str, float], out_dir: Path) -> None:
    specs = [
        ("final_accuracy", "Final test accuracy (%)", "exp2_ratio_vs_final_accuracy.png"),
        ("tail_mean_accuracy", "Tail mean test accuracy (%)", "exp2_ratio_vs_tail_accuracy.png"),
        ("auc_mean_accuracy", "Mean accuracy over rounds (%)", "exp2_ratio_vs_auc_accuracy.png"),
        ("b_star_mean", "Mean $b_t^*$", "exp2_ratio_vs_bstar.png"),
        ("nmse_total_last", "NMSE_total (last round)", "exp2_ratio_vs_nmse.png"),
    ]
    for metric, ylabel, filename in specs:
        plt.figure(figsize=(7.2, 4.6))
        for method in METHOD_ORDER:
            sub = sorted([r for r in rows if r["method"] == method], key=lambda r: r["ratio"])
            if not sub:
                continue
            if method == "full" and len(sub) == 1:
                plt.scatter([1.0], [sub[0][metric]], marker="D", s=60, color="#2ca02c",
                            zorder=3, label="FULL")
                continue
            plt.plot([r["ratio"] for r in sub], [r[metric] for r in sub],
                     marker="o", linewidth=2.0, label=method.upper())
        for method, ratio in kstar.items():
            plt.axvline(ratio, linestyle="--", linewidth=1.2, alpha=0.8)
            plt.text(ratio, plt.ylim()[1], f" {method} offline k*", rotation=90,
                     va="top", fontsize=8)
        if metric in ("b_star_mean", "nmse_total_last"):
            plt.yscale("log")
        plt.xlabel("Compression ratio k/d")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.45)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=220)
        plt.close()


def write_markdown(path: Path, rows: List[Dict], kstar: Dict[str, float], args) -> None:
    best = best_by_method(rows)
    with path.open("w", encoding="utf-8") as f:
        f.write("# 实验二（新场景 0810）：在线最优压缩率搜索\n\n")
        f.write("每个候选 (method, ratio) 均在完整物理链路（固定参与、逐轮 b_t*(k)、"
                "物理噪声、理想 RMS-AGC、径向限幅）下端到端训练，工作点由实测精度选出。\n\n")
        f.write("## 设置\n\n")
        f.write(f"- dataset: `{args.dataset}`, seed: `{args.seed}`, rounds: `{args.rounds}`, "
                f"eval_every: `{args.eval_every}`\n")
        f.write(f"- epsilon: `{args.epsilon}`, delta: `{args.delta}`, P_cap: `{args.p_cap_dbm}` dBm, "
                f"B_clip: `{args.adc_backoff_db}` dB, c_tx: `{args.c_tx}`\n")
        f.write(f"- lr: `{args.lr}`, local_steps: `{args.local_steps}`, N: `{args.num_clients}`\n")
        if kstar:
            f.write(f"- 实验一离线 k*: `{kstar}`\n")
        f.write("\n## 全部结果\n\n")
        f.write("| 方法 | k/d | final | tail | AUC | best(轮) | 到target轮 | b*均值 | "
                "rho_clip末 | NMSE_total末 | 末轮约束 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in sorted(rows, key=lambda r: (METHOD_ORDER.index(r["method"]), r["ratio"])):
            hit = "-" if row["round_to_target"] is None else str(row["round_to_target"])
            f.write(
                f"| {row['method']} | {row['ratio']:.3f} | {row['final_accuracy']:.2f} | "
                f"{row['tail_mean_accuracy']:.2f} | {row['auc_mean_accuracy']:.2f} | "
                f"{row['best_accuracy']:.2f}({row['best_round']}) | {hit} | "
                f"{row['b_star_mean']:.3e} | {row['rho_clip_last']:.2e} | "
                f"{row['nmse_total_last']:.2e} | {row['regime_last']} |\n"
            )
        f.write("\n## 每方法在线最优（按 final accuracy）\n\n")
        f.write("| 方法 | 在线 k*/d | final | tail | 离线 k*/d(实验一) | 是否一致 |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for method in METHOD_ORDER:
            if method not in best:
                continue
            row = best[method]
            offline = kstar.get(method)
            if offline is None:
                agree = "-"
                offline_txt = "-"
            else:
                agree = "是" if abs(offline - row["ratio"]) < 1e-9 else "否（相邻档内属正常）"
                offline_txt = f"{offline:.3f}"
            f.write(f"| {method} | {row['ratio']:.3f} | {row['final_accuracy']:.2f} | "
                    f"{row['tail_mean_accuracy']:.2f} | {offline_txt} | {agree} |\n")
        f.write("\n结论必须按本表实际排序书写；若 Rand-k 在自己的 sweep 中更优，"
                "不能手工选择更差的 Rand-k 作为主 baseline。\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="exp_0810/results/exp2_online")
    parser.add_argument("--dataset", default="femnist", choices=["mnist", "femnist"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--topk-ratios", default="0.01,0.02,0.05,0.10,0.15,0.20,0.30,0.50")
    parser.add_argument("--randk-ratios", default="0.01,0.02,0.05,0.10,0.20,0.35,0.50,0.80")
    parser.add_argument("--include-full", action="store_true", default=True)
    parser.add_argument("--no-full", dest="include_full", action="store_false")
    parser.add_argument("--force", action="store_true", help="re-run finished candidates")
    # Physical scenario knobs (keep identical to experiment 1 for comparison).
    parser.add_argument("--epsilon", type=float, default=5.0)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--p-cap-dbm", type=float, default=20.0)
    parser.add_argument("--adc-backoff-db", type=float, default=6.0, help="use inf to disable clipping")
    parser.add_argument("--c-tx", type=float, default=0.02)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--oversampling", type=int, default=4)
    # Learning knobs.
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--local-steps", type=int, default=5)
    # Aggregation knobs.
    parser.add_argument("--target-accuracy", type=float, default=75.0)
    parser.add_argument("--tail-window", type=int, default=5)
    parser.add_argument("--exp1-summary", default="exp_0810/results/exp1_offline/summary.json",
                        help="overlay offline k* if the file exists; empty to disable")
    parser.add_argument("--mnist-root", default="data/MNIST/raw")
    parser.add_argument("--femnist-path", default="data/femnist/femnist_train.pkl")
    parser.add_argument("--femnist-test-path", default="data/femnist/femnist_test.pkl")
    args = parser.parse_args()

    backoff = float("inf") if str(args.adc_backoff_db).lower() in ("inf", "infinity") else float(args.adc_backoff_db)
    phy = PhyConfig(
        num_clients=args.num_clients,
        epsilon=args.epsilon,
        delta=args.delta,
        p_cap_dbm=args.p_cap_dbm,
        adc_backoff_db=backoff,
        c_tx=args.c_tx,
        oversampling=args.oversampling,
    )
    cfg = LearnConfig(
        seed=args.seed,
        device=args.device,
        rounds=args.rounds,
        eval_every=args.eval_every,
        lr=args.lr,
        local_steps=args.local_steps,
        mnist_root=args.mnist_root,
        femnist_path=args.femnist_path,
        femnist_test_path=args.femnist_test_path,
    )
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        cfg.device = "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "phy": {k: (str(v) if isinstance(v, float) and math.isinf(v) else v)
                for k, v in vars(phy).items()},
        "learn": vars(cfg),
        "args": vars(args),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    jobs: List[Dict] = []
    for ratio in parse_ratios(args.topk_ratios):
        jobs.append({"method": "topk", "ratio": ratio})
    for ratio in parse_ratios(args.randk_ratios):
        jobs.append({"method": "randk", "ratio": ratio})
    if args.include_full:
        jobs.append({"method": "full", "ratio": 1.0})

    system = FLSystem(phy, cfg, args.dataset)
    results: List[Dict] = []
    for job in jobs:
        run_dir = out_dir / "runs" / f"{job['method']}_r{ratio_tag(job['ratio'])}"
        metrics_path = run_one(system, job["method"], job["ratio"], run_dir, args.force)
        results.append(read_run(metrics_path, args.target_accuracy, args.tail_window))

    kstar = load_exp1_kstar(args.exp1_summary)
    results.sort(key=lambda r: (METHOD_ORDER.index(r["method"]), r["ratio"]))
    with (out_dir / "sweep_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    (out_dir / "sweep_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_sweep(results, kstar, out_dir)
    write_markdown(out_dir / "summary.md", results, kstar, args)

    best = best_by_method(results)
    print(json.dumps({
        "output_dir": str(out_dir),
        "online_k_star": {m: best[m]["ratio"] for m in best},
        "offline_k_star": kstar,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
