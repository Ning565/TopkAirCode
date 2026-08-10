#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 1 (new scenario, 2026-08-10): OFFLINE optimal-compression search.

Idea (paper core, convergence_analysis.tex Theorem A.1)
-------------------------------------------------------
The dynamic-range-aware convergence bound decomposes into three groups:

  J_A(k) =  Phi_A(k)                                  # learning side
          + (8*Gamma_A(k)*L*sigma_sc^2*d)/(eta*tau*N^2) * avg_t 1/b_t^2
                                                       # recovered channel noise
          + (16*Gamma_A(k)*C_dr)/(eta*tau*N^2) * avg_t E_clip^t(k,b_t)/b_t^2
                                                       # receiver dynamic range

with  mu_A(k)   = 2(1-w)(2-w)/w^2,  w = certified retention  bar_omega_A(k),
      Gamma_A(k)= 2 + 16 L^2 mu_A(k) eta^2 tau^2 G^2,
      C_dr      = 4/(eta*tau) + L,
      Phi_A(k)  = 16*Gamma*Df/(T*eta*tau) + 16*Gamma*E_cal(k)/(eta*tau)
                + 16*Gamma*eta*tau*L*Psi + 16*Gamma*Lambda*R(k)/(eta*tau) + R(k),
      E_cal(k)  = (eta*tau + 2L eta^2 tau^2) * beta_A^2(k),
      R(k)      = 8 L^2 mu eta^2 tau^2 (2 kappa^2 + zeta^2),
      Psi, Lambda as in the theorem.

This script CALIBRATES every plug-in quantity from real client updates under
the new physical scenario (fixed participation, per-round b_t^*(k), physical
sigma_sc, oversampled waveform + ideal RMS-AGC + radial clipping):

  bar_omega_A(k) : low quantile (default q10) of per-(client,round) retained
                   energy -> certified lower envelope (Assumption A.6).
  beta_A^2(k)    : median of ||xi||^2/(eta^2 tau^2), xi = pre-transmission
                   element-wise clipping bias of the sparsified update.
  L              : secant estimate along the (ideally advanced) trajectory,
                   median of ||g_{t+1}-g_t|| / ||theta_{t+1}-theta_t||.
  G^2, kappa^2   : least squares of (1/N)sum||g_i||^2 = G^2||g||^2 + kappa^2
                   with G^2 >= 1, kappa^2 >= 0.
  zeta^2         : median of 0.5*||g_a-g_b||^2 over paired mini-batch grads.
  avg 1/b_t^2, avg E_clip/b_t^2 : measured per round on the real waveform.

Quantile-based (median / low-quantile) plug-in statistics are mandatory: the
earlier max-based calibration inflated the learning coefficient by ~3 orders
of magnitude and destroyed the Rand-k interior optimum (project lesson).

Outputs
-------
  objective_terms.csv                    raw + normalized per-(method,ratio)
  summary.json / summary.md              constants, k*, trend checks
  exp1_objective_bound.png               raw bound terms (log scale) + total
  exp1_objective_normalized.png          normalized 3-component view
  exp1_topk_adc_ablation.png             ADC-aware vs ADC-unaware Top-k
  exp1_papr_ccdf.png                     structure-line PAPR CCDF at k*
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from full_system_0810 import (  # noqa: E402
    FLSystem,
    LearnConfig,
    OFDMAirCompChannel,
    PhyConfig,
    apply_update,
    compress_update,
    draw_round_channel,
    draw_topology,
    elementwise_clip,
    flatten_params,
    make_model,
    scaling_limits,
    set_seed,
)


# ---------------------------------------------------------------------------
# Bound-term algebra (Theorem A.1 constants)
# ---------------------------------------------------------------------------


def mu_of_omega(omega: float) -> float:
    omega = min(1.0, max(1e-4, omega))
    if omega >= 1.0 - 1e-12:
        return 0.0
    return 2.0 * (1.0 - omega) * (2.0 - omega) / (omega * omega)


def bound_terms(
    *,
    omega: float,
    beta2: float,
    mean_noise_over_b2: float,
    mean_eclip_over_b2: float,
    consts: Dict[str, float],
) -> Dict[str, float]:
    L = consts["L"]
    G2 = consts["G2"]
    kappa2 = consts["kappa2"]
    zeta2 = consts["zeta2"]
    df = consts["delta_f"]
    eta = consts["eta"]
    tau = consts["tau"]
    T = consts["T"]
    N = consts["N"]
    d = consts["d"]

    et = eta * tau
    mu = mu_of_omega(omega)
    gamma_a = 2.0 + 16.0 * L * L * mu * et * et * G2
    c_dr = 4.0 / et + L
    e_cal = (et + 2.0 * L * et * et) * beta2
    lam = et * (0.5 + 4.0 * L * et) * (1.0 + 32.0 * et * et * G2 * L * L)
    r_a = 8.0 * L * L * mu * et * et * (2.0 * kappa2 + zeta2)
    psi = 2.0 * (2.0 * kappa2 + zeta2) + eta * L * (0.5 + 4.0 * L * et) * (32.0 * tau * kappa2 + 8.0 * zeta2)

    phi = (
        16.0 * gamma_a * df / (T * et)
        + 16.0 * gamma_a * e_cal / et
        + 16.0 * gamma_a * et * L * psi
        + 16.0 * gamma_a * lam * r_a / et
        + r_a
    )
    # Channel + DP noise term: sigma_eff,t^2 = sigma_sc^2 + 2 b_t^2 N sigma_a,t^2
    # (artificial top-up included), collected per round as sigma_eff^2 / b^2.
    j_ch = 8.0 * gamma_a * L * d / (et * N * N) * mean_noise_over_b2
    j_dr = 16.0 * gamma_a * c_dr / (et * N * N) * mean_eclip_over_b2
    return {
        "mu": mu,
        "gamma_a": gamma_a,
        "learning_term": phi,
        "channel_term": j_ch,
        "dr_term": j_dr,
        "J_bound": phi + j_ch + j_dr,
        "J_bound_unaware": phi + j_ch,
    }


def fit_g2_kappa2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """LS fit y = G2*x + kappa2 with G2 >= 1, kappa2 >= 0."""
    if len(x) >= 2 and float(np.var(x)) > 0:
        g2 = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
        k2 = float(np.mean(y) - g2 * np.mean(x))
    else:
        g2, k2 = 1.0, float(np.median(y - x)) if len(x) else 0.0
    g2 = max(1.0, g2)
    k2 = max(0.0, float(np.median(y) - g2 * np.median(x))) if k2 < 0 else max(0.0, k2)
    return g2, k2


# ---------------------------------------------------------------------------
# Calibration pass
# ---------------------------------------------------------------------------


def calibrate(args, phy: PhyConfig, cfg: LearnConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    system = FLSystem(phy, cfg, args.dataset)
    model = make_model(args.dataset, system.num_classes).to(cfg.device)
    d = sum(p.numel() for p in model.parameters())
    channel = OFDMAirCompChannel(phy, d, noise_seed=cfg.seed + 777)
    topo = draw_topology(phy, cfg.seed)
    chan_rng = np.random.default_rng(cfg.seed + 424242)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    branches: List[Tuple[str, float]] = []
    for m in methods:
        if m == "full":
            branches.append(("full", 1.0))
        else:
            branches.extend((m, r) for r in ratios)

    ef_mem: Dict[Tuple[str, float], Dict[int, torch.Tensor]] = {}
    if args.ef_in_calibration:
        for br in branches:
            if br[0] in cfg.error_feedback_methods and br[0] != "full":
                ef_mem[br] = {cid: torch.zeros(d) for cid in range(phy.num_clients)}

    acc: Dict[Tuple[str, float], Dict[str, list]] = {
        br: {
            "omega": [], "beta2": [], "inv_b2": [], "noise_over_b2": [], "eclip_over_b2": [],
            "rho_clip": [], "d_clip": [], "papr_p99_db": [], "psr_round_db": [],
            "nmse_clip": [], "nmse_total": [], "silent_ratio": [],
            "b_star": [], "sigma_a": [], "sigma_dp": [], "loose": [], "papr_samples": [],
        }
        for br in branches
    }

    theta_trace: List[torch.Tensor] = []
    grad_trace: List[torch.Tensor] = []
    gx, gy = [], []          # ||gbar||^2  and  (1/N) sum ||g_i||^2
    zeta_samples: List[float] = []
    loss_trace: List[float] = []

    eta, tau = cfg.lr, cfg.local_steps
    for t in range(args.calib_rounds):
        deltas, losses = [], []
        for cid in range(phy.num_clients):
            delta, loss = system.client_delta(model, cid)
            deltas.append(delta)
            losses.append(loss)
        loss_trace.append(float(np.mean(losses)))
        grads = [(-delta / (eta * tau)) for delta in deltas]
        gbar = torch.stack(grads, dim=0).mean(dim=0)
        gx.append(float(gbar.pow(2).sum()))
        gy.append(float(np.mean([float(g.pow(2).sum()) for g in grads])))
        theta_trace.append(flatten_params(model))
        grad_trace.append(gbar)
        for cid in range(min(3, phy.num_clients)):
            ga = system.client_grad(model, cid)
            gb = system.client_grad(model, cid)
            zeta_samples.append(0.5 * float((ga - gb).pow(2).sum()))

        rc = draw_round_channel(phy, topo, chan_rng)
        g_min = float(rc.g_abs.min())
        for br in branches:
            method, ratio = br
            k = d if method == "full" else max(1, int(d * ratio))
            lim = scaling_limits(phy, channel.S, k, g_min, d)
            gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 10007 * t + int(ratio * 1_000_000))
            common_idx = None
            if method == "randk" and k < d:
                common_idx = torch.randperm(d, generator=gen)[:k]
            signals = []
            for cid in range(phy.num_clients):
                v = deltas[cid] + ef_mem[br][cid] if br in ef_mem else deltas[cid]
                tilde_v, mask, retained, _ = compress_update(v, method, ratio, gen, common_idx)
                s_tx = elementwise_clip(tilde_v, phy.c_tx)
                xi = tilde_v - s_tx
                acc[br]["omega"].append(retained)
                acc[br]["beta2"].append(float(xi.pow(2).sum()) / (eta * eta * tau * tau))
                if br in ef_mem:
                    ef_mem[br][cid] = v - tilde_v
                signals.append(s_tx)
            _, comm = channel.transmit_round(
                signals, lim["b_star"], t, collect_papr_samples=True, sigma_a=lim["sigma_a_client"]
            )
            a = acc[br]
            b2 = lim["b_star"] ** 2
            a["inv_b2"].append(1.0 / b2)
            a["noise_over_b2"].append(
                phy.sigma_sc2 / b2 + 2.0 * phy.num_clients * lim["sigma_a_client"] ** 2
            )
            a["eclip_over_b2"].append(comm["e_clip_over_b2"])
            a["rho_clip"].append(comm["rho_clip"])
            a["d_clip"].append(comm["d_clip"])
            a["papr_p99_db"].append(comm["papr_p99_db"])
            a["psr_round_db"].append(comm["psr_round_db"])
            a["nmse_clip"].append(comm["nmse_clip"])
            a["nmse_total"].append(comm["nmse_total"])
            a["silent_ratio"].append(comm["silent_symbol_ratio"])
            a["b_star"].append(lim["b_star"])
            a["sigma_a"].append(lim["sigma_a_client"])
            a["sigma_dp"].append(lim["sigma_dp"])
            a["loose"].append(1.0 if lim["regime"] == "loose" else 0.0)
            a["papr_samples"].append(comm["papr_db_samples"])

        if args.advance_model:
            ideal = torch.stack(deltas, dim=0).mean(dim=0)
            apply_update(model, ideal)
        print(f"[exp1-calib] {args.dataset} round {t + 1}/{args.calib_rounds} "
              f"(burst redraws={rc.redraws})", flush=True)

    # ---- global plug-in constants (quantile-based, NOT max-based) ----------
    if args.l_smooth > 0:
        l_hat = args.l_smooth
    elif len(theta_trace) >= 2 and args.advance_model:
        secants = []
        for i in range(len(theta_trace) - 1):
            dth = float((theta_trace[i + 1] - theta_trace[i]).norm())
            dgr = float((grad_trace[i + 1] - grad_trace[i]).norm())
            if dth > 1e-12:
                secants.append(dgr / dth)
        l_hat = float(np.median(secants)) if secants else 1.0
    else:
        l_hat = 1.0
    g2_hat, kappa2_hat = fit_g2_kappa2(np.asarray(gx), np.asarray(gy))
    zeta2_hat = float(np.median(zeta_samples)) if zeta_samples else 0.0
    delta_f_hat = max(1e-6, loss_trace[0])

    consts = {
        "L": l_hat, "G2": g2_hat, "kappa2": kappa2_hat, "zeta2": zeta2_hat,
        "delta_f": delta_f_hat, "eta": eta, "tau": tau, "T": float(args.rounds),
        "N": float(phy.num_clients), "d": float(d), "sigma_sc2": phy.sigma_sc2,
        "c_tx": phy.c_tx,
    }
    return {"branches": branches, "acc": acc, "consts": consts, "d": d, "S": channel.S}


# ---------------------------------------------------------------------------
# Objective assembly, selection, reporting
# ---------------------------------------------------------------------------


def build_rows(args, calib: Dict[str, Any]) -> List[Dict[str, Any]]:
    consts = calib["consts"]
    rows: List[Dict[str, Any]] = []
    for method, ratio in calib["branches"]:
        a = calib["acc"][(method, ratio)]
        d = calib["d"]
        k = d if method == "full" else max(1, int(d * ratio))
        omega = float(np.quantile(np.asarray(a["omega"]), args.omega_quantile))
        beta2 = float(np.median(np.asarray(a["beta2"])))
        mean_inv_b2 = float(np.mean(a["inv_b2"]))
        mean_noise_over_b2 = float(np.mean(a["noise_over_b2"]))
        mean_eclip_over_b2 = float(np.mean(a["eclip_over_b2"]))
        terms = bound_terms(
            omega=omega, beta2=beta2, mean_noise_over_b2=mean_noise_over_b2,
            mean_eclip_over_b2=mean_eclip_over_b2, consts=consts,
        )
        rows.append({
            "dataset": args.dataset,
            "method": method,
            "ratio": ratio,
            "k": k,
            "d": d,
            "bar_omega": omega,
            "beta2": beta2,
            "mean_inv_b2": mean_inv_b2,
            "mean_noise_over_b2": mean_noise_over_b2,
            "mean_eclip_over_b2": mean_eclip_over_b2,
            "b_star_mean": float(np.mean(a["b_star"])),
            "sigma_a_mean": float(np.mean(a["sigma_a"])),
            "sigma_dp_mean": float(np.mean(a["sigma_dp"])),
            "sigma_dp_over_ctx": float(np.mean(a["sigma_dp"])) / consts["c_tx"],
            "loose_frac": float(np.mean(a["loose"])),
            "rho_clip": float(np.mean(a["rho_clip"])),
            "d_clip": float(np.mean(a["d_clip"])),
            "papr_p99_db": float(np.nanmean(a["papr_p99_db"])),
            "psr_round_db": float(np.nanmean(a["psr_round_db"])),
            "nmse_clip": float(np.mean(a["nmse_clip"])),
            "nmse_total": float(np.mean(a["nmse_total"])),
            "silent_symbol_ratio": float(np.mean(a["silent_ratio"])),
            **terms,
        })
    return rows


def add_normalized_objective(rows: List[Dict[str, Any]], lam_ch: float, lam_dr: float) -> None:
    """Engineering view: per-method min-max normalized components + weights."""
    for method in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == method]
        for key in ("learning_term", "channel_term", "dr_term"):
            vals = np.log1p(np.asarray([r[key] for r in sub], dtype=float))
            lo, hi = float(vals.min()), float(vals.max())
            for r, v in zip(sub, vals):
                r[f"{key}_norm"] = 0.0 if hi - lo < 1e-12 else float((v - lo) / (hi - lo))
        for r in sub:
            r["J_norm"] = r["learning_term_norm"] + lam_ch * r["channel_term_norm"] + lam_dr * r["dr_term_norm"]
            r["J_norm_unaware"] = r["learning_term_norm"] + lam_ch * r["channel_term_norm"]


def select_best(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for method in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == method]
        if method == "full":
            best["full"] = sub[0]
            continue
        best[method] = min(sub, key=lambda r: r[key])
    return best


def interior_check(rows: List[Dict[str, Any]], best: Dict[str, Dict[str, Any]], key: str) -> List[str]:
    msgs = []
    for method in ("topk", "randk"):
        if method not in best:
            continue
        sub = sorted([r for r in rows if r["method"] == method], key=lambda r: r["ratio"])
        grid = [r["ratio"] for r in sub]
        k_star = best[method]["ratio"]
        pos = "INTERIOR" if grid[0] < k_star < grid[-1] else "BOUNDARY"
        msgs.append(f"{method} ({key}): k*/d = {k_star:.3f} -> {pos}"
                    + ("  [WARNING: boundary optimum, check epsilon/P_cap profile]" if pos == "BOUNDARY" else ""))
    return msgs


def trend_check(rows: List[Dict[str, Any]]) -> List[str]:
    msgs = []
    for method in ("topk", "randk"):
        sub = sorted([r for r in rows if r["method"] == method], key=lambda r: r["ratio"])
        if len(sub) < 2:
            continue
        learn = [r["learning_term"] for r in sub]
        chan = [r["channel_term"] for r in sub]
        dr = [r["dr_term"] for r in sub]
        msgs.append(f"{method}: learning non-increasing in k = {all(a >= b - 1e-15 for a, b in zip(learn, learn[1:]))}")
        msgs.append(f"{method}: channel  non-decreasing in k = {all(a <= b + 1e-15 for a, b in zip(chan, chan[1:]))}")
        msgs.append(f"{method}: dyn-range non-decreasing in k = {all(a <= b + 1e-15 for a, b in zip(dr, dr[1:]))}")
    return msgs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_bound(rows, best, out_dir: Path) -> None:
    methods = [m for m in ("topk", "randk") if any(r["method"] == m for r in rows)]
    fig, axes = plt.subplots(1, max(1, len(methods)), figsize=(6.2 * max(1, len(methods)), 4.6), squeeze=False)
    for ax, method in zip(axes[0], methods):
        sub = sorted([r for r in rows if r["method"] == method], key=lambda r: r["ratio"])
        x = [r["ratio"] for r in sub]
        ax.plot(x, [r["learning_term"] for r in sub], marker="o", label="Learning $\\Phi_A(k)$")
        ax.plot(x, [r["channel_term"] for r in sub], marker="s", label="Channel noise")
        ax.plot(x, [r["dr_term"] for r in sub], marker="^", label="Dynamic range")
        ax.plot(x, [r["J_bound"] for r in sub], marker="D", linewidth=2.4, color="black", label="Total bound")
        ax.axvline(best[method]["ratio"], color="red", linestyle="--", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Compression ratio k/d")
        ax.set_title(f"{method.upper()} bound-direct objective")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0][0].set_ylabel("Bound term value (log)")
    axes[0][-1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_objective_bound.png", dpi=240)
    plt.close(fig)


def plot_normalized(rows, best_norm, out_dir: Path) -> None:
    methods = [m for m in ("topk", "randk") if any(r["method"] == m for r in rows)]
    fig, axes = plt.subplots(1, max(1, len(methods)), figsize=(6.2 * max(1, len(methods)), 4.6), squeeze=False)
    for ax, method in zip(axes[0], methods):
        sub = sorted([r for r in rows if r["method"] == method], key=lambda r: r["ratio"])
        x = [r["ratio"] for r in sub]
        ax.plot(x, [r["learning_term_norm"] for r in sub], marker="o", label="Learning")
        ax.plot(x, [r["channel_term_norm"] for r in sub], marker="s", label="Channel noise")
        ax.plot(x, [r["dr_term_norm"] for r in sub], marker="^", label="Dynamic range")
        ax.plot(x, [r["J_norm"] for r in sub], marker="D", linewidth=2.4, color="black", label="Total objective")
        ax.axvline(best_norm[method]["ratio"], color="red", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Compression ratio k/d")
        ax.set_title(f"{method.upper()} normalized objective")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0][0].set_ylabel("Normalized value")
    axes[0][-1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_objective_normalized.png", dpi=240)
    plt.close(fig)


def plot_ablation(rows, out_dir: Path) -> None:
    sub = sorted([r for r in rows if r["method"] == "topk"], key=lambda r: r["ratio"])
    if not sub:
        return
    x = [r["ratio"] for r in sub]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, [r["J_bound"] for r in sub], marker="o", linewidth=2.2, label="DR-aware Top-k")
    ax.plot(x, [r["J_bound_unaware"] for r in sub], marker="s", linewidth=2.0, label="DR-unaware Top-k")
    aware = min(sub, key=lambda r: r["J_bound"])["ratio"]
    unaware = min(sub, key=lambda r: r["J_bound_unaware"])["ratio"]
    for ratio, label in ((aware, "aware k*"), (unaware, "unaware k*")):
        ax.axvline(ratio, linestyle="--", linewidth=1.2)
        ax.text(ratio, ax.get_ylim()[1] * 0.9, label, ha="center", va="top", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("Compression ratio k/d")
    ax.set_ylabel("Bound objective (log)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_topk_adc_ablation.png", dpi=240)
    plt.close(fig)


def plot_papr_ccdf(calib, best, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    picks = []
    for method in ("topk", "randk"):
        if method in best:
            picks.append((method, best[method]["ratio"]))
    if ("full", 1.0) in calib["branches"]:
        picks.append(("full", 1.0))
    for method, ratio in picks:
        samples = np.concatenate(calib["acc"][(method, ratio)]["papr_samples"])
        if samples.size == 0:
            continue
        xs = np.sort(samples)
        ccdf = 1.0 - np.arange(1, xs.size + 1) / xs.size
        ax.semilogy(xs, np.maximum(ccdf, 1.0 / xs.size), linewidth=2.0,
                    label=f"{method.upper()} k/d={ratio:.2f}")
    ax.set_xlabel("PAPR threshold $\\xi$ (dB)")
    ax.set_ylabel("CCDF  Pr{PAPR > $\\xi$}")
    ax.set_title("Structure-line (noiseless) PAPR CCDF at selected k*")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_papr_ccdf.png", dpi=240)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_outputs(args, out_dir: Path, calib, rows, best_bound, best_norm) -> None:
    csv_rows = [{k: v for k, v in r.items()} for r in rows]
    with (out_dir / "objective_terms.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    payload = {
        "args": vars(args),
        "constants": calib["consts"],
        "S_symbols": calib["S"],
        "d": calib["d"],
        "best_bound": {k: {kk: vv for kk, vv in v.items()} for k, v in best_bound.items()},
        "best_normalized": {k: {kk: vv for kk, vv in v.items()} for k, v in best_norm.items()},
        "interior_check_bound": interior_check(rows, best_bound, "J_bound"),
        "interior_check_norm": interior_check(rows, best_norm, "J_norm"),
        "trend_check": trend_check(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# 实验一（新场景 0810）：离线最优压缩率搜索\n\n")
        f.write("目标函数 = 收敛界三项分解（学习项 + 信道噪声项 + 动态范围项），"
                "常数由真实 client update 分位数校准；物理层为固定参与 + 逐轮 "
                "b_t*(k) + 理想 RMS-AGC + 径向限幅。\n\n")
        c = calib["consts"]
        f.write("## 校准常数\n\n")
        f.write(f"- L={c['L']:.4g}, G^2={c['G2']:.4g}, kappa^2={c['kappa2']:.4g}, "
                f"zeta^2={c['zeta2']:.4g}, Delta f={c['delta_f']:.4g}\n")
        f.write(f"- eta={c['eta']}, tau={c['tau']}, T={c['T']:.0f}, N={c['N']:.0f}, "
                f"d={c['d']:.0f}, sigma_sc^2={c['sigma_sc2']:.3e} W\n")
        f.write(f"- epsilon={args.epsilon}, delta={args.delta}, P_cap={args.p_cap_dbm} dBm, "
                f"B_clip={args.adc_backoff_db} dB, c_tx={args.c_tx}\n\n")
        f.write("## 趋势与内点检查\n\n")
        for msg in payload["trend_check"] + payload["interior_check_bound"] + payload["interior_check_norm"]:
            f.write(f"- {msg}\n")
        f.write("\n## 搜索结果（定理口径 J_bound）\n\n")
        f.write("| 方法 | k*/d | b*均值 | sigma_dp/c_tx | 宽松占比 | bar_omega | PAPR P99 | rho_clip | NMSE_total | J |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for name, row in best_bound.items():
            f.write(
                f"| {name} | {row['ratio']:.3f} | {row['b_star_mean']:.3e} | "
                f"{row['sigma_dp_over_ctx']:.2f} | {row['loose_frac']:.2f} | {row['bar_omega']:.3f} | "
                f"{row['papr_p99_db']:.2f} | {row['rho_clip']:.2e} | "
                f"{row['nmse_total']:.2e} | {row['J_bound']:.4e} |\n"
            )
        f.write("\n## 搜索结果（归一化工程口径 J_norm）\n\n")
        f.write("| 方法 | k*/d | J_norm |\n|---|---:|---:|\n")
        for name, row in best_norm.items():
            f.write(f"| {name} | {row['ratio']:.3f} | {row.get('J_norm', float('nan')):.4f} |\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="exp_0810/results/exp1_offline")
    parser.add_argument("--dataset", default="femnist", choices=["mnist", "femnist"])
    parser.add_argument("--methods", default="topk,randk,full")
    parser.add_argument("--ratios", default="0.01,0.02,0.05,0.10,0.15,0.20,0.30,0.50,0.80")
    parser.add_argument("--calib-rounds", type=int, default=8)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=200, help="T used inside the bound")
    # Physical scenario knobs.
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
    # Calibration statistics.
    parser.add_argument("--omega-quantile", type=float, default=0.10,
                        help="low quantile as certified retention envelope")
    parser.add_argument("--l-smooth", type=float, default=0.0,
                        help=">0 fixes L; 0 estimates by trajectory secant")
    parser.add_argument("--lambda-channel", type=float, default=0.15)
    parser.add_argument("--lambda-dr", type=float, default=0.50)
    parser.add_argument("--no-advance-model", dest="advance_model", action="store_false", default=True)
    parser.add_argument("--no-ef-in-calibration", dest="ef_in_calibration", action="store_false", default=True)
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
        device=args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu",
        lr=args.lr,
        local_steps=args.local_steps,
        rounds=args.rounds,
        mnist_root=args.mnist_root,
        femnist_path=args.femnist_path,
        femnist_test_path=args.femnist_test_path,
    )
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        cfg.device = "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = calibrate(args, phy, cfg)
    rows = build_rows(args, calib)
    add_normalized_objective(rows, args.lambda_channel, args.lambda_dr)
    best_bound = select_best(rows, "J_bound")
    best_norm = select_best(rows, "J_norm")

    plot_bound(rows, best_bound, out_dir)
    plot_normalized(rows, best_norm, out_dir)
    plot_ablation(rows, out_dir)
    plot_papr_ccdf(calib, best_bound, out_dir)
    write_outputs(args, out_dir, calib, rows, best_bound, best_norm)

    print(json.dumps({
        "output_dir": str(out_dir),
        "k_star_bound": {m: best_bound[m]["ratio"] for m in best_bound},
        "k_star_norm": {m: best_norm[m]["ratio"] for m in best_norm},
        "interior_check": interior_check(rows, best_bound, "J_bound"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
