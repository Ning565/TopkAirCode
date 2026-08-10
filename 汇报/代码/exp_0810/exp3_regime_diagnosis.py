#!/usr/bin/env python3
"""Exp3 pre-flight: DP design diagnosis under the artificial-noise top-up
mechanism (no training, pure stdlib, runs in seconds on any machine).

The per-round design is: b_t = B_P^t(k)/sqrt(F) (power-limited including the
noise power tax) and every client injects real Gaussian noise on the full
d-coordinate grid so that thermal + artificial noise meets the per-round
client-level (eps, delta)-DP target (Koda'20 / Wei JSAC'22 / Liu TWC'24
lineage).  This script answers, WITHOUT experiments:

  1. eps_loose(k): the epsilon at which the recovered DP noise drops to the
     public clip scale c_tx (learning becomes "loose").  Closed form,
     channel-independent:  eps_loose = (sqrt(2k)/N + sqrt(ln 1/dlt))^2 - ln 1/dlt.
  2. sigma_dp/c_tx over an (eps, k) grid -- training viability forecast.
  3. The noise power tax sqrt(F), F = 1 + 2d/(N margin^2), and the per-coord
     signal-to-artificial-noise ratio N margin/(2 sqrt(k)) (waveform structure
     visibility in the PAPR/clipping experiments).
  4. Honesty check on the intrinsic free region (sigma_a = 0): it requires
     margin*sqrt(F) >= rho = 2 min|g| sqrt(SM P_cap)/sigma_sc, and
     margin*sqrt(F) >= sqrt(2d/N) ~ 2e2 while rho ~ 7e4 on this link, so the
     free region stays astronomically far regardless of protocol parameters.

If torch + full_system_0810 are importable, all formulas are cross-checked
against scaling_limits() to prevent drift.

Usage:
  python3 exp_0810/exp3_regime_diagnosis.py \
      --output-dir exp_0810/results/exp3_regime_diagnosis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

# ----- defaults: MUST mirror PhyConfig in full_system_0810.py ---------------
N_CLIENTS = 20
CELL_RADIUS_M = 250.0
MIN_DISTANCE_M = 10.0
PL0_DB = 30.0
PL_EXP = 3.0
N0_DBM_HZ = -174.0
NF_DB = 5.0
SUBCARRIERS = 1024          # M
SPACING_HZ = 15e3           # delta f
H_CUT = 0.1
P_CAP_DBM = 20.0
DELTA_DP = 1e-3
C_TX = 0.02
D_MODEL = 403774            # StableCNN update dimension


def sigma_sc() -> float:
    n0 = 10.0 ** ((N0_DBM_HZ - 30.0) / 10.0)
    return math.sqrt(n0 * SPACING_HZ * 10.0 ** (NF_DB / 10.0))


def ln_delta() -> float:
    return math.log(1.0 / DELTA_DP)


def privacy_margin(eps: float) -> float:
    return math.sqrt(eps + ln_delta()) - math.sqrt(ln_delta())


def eps_loose(k: int) -> float:
    """sigma_dp = c_tx threshold; the sparsity-buys-privacy knob."""
    return (math.sqrt(2.0 * k) / N_CLIENTS + math.sqrt(ln_delta())) ** 2 - ln_delta()


def noise_tax_sqrt_f(eps: float) -> float:
    m = privacy_margin(eps)
    return math.sqrt(1.0 + 2.0 * D_MODEL / (N_CLIENTS * m * m))


def sigma_dp_over_ctx(eps: float, k: int) -> float:
    """Recovered DP noise over c_tx (thermal credit negligible on this link)."""
    return math.sqrt(2.0 * k) / (N_CLIENTS * privacy_margin(eps))


def sig_over_art(eps: float, k: int) -> float:
    """Per-coordinate aggregate signal-to-artificial-noise amplitude ratio."""
    return N_CLIENTS * privacy_margin(eps) / (2.0 * math.sqrt(k))


def draw_topology(rng: random.Random, radius_m: float) -> list[float]:
    betas = []
    for _ in range(N_CLIENTS):
        r = math.sqrt(MIN_DISTANCE_M ** 2 + rng.random() * (radius_m ** 2 - MIN_DISTANCE_M ** 2))
        betas.append(10.0 ** (-(PL0_DB + 10.0 * PL_EXP * math.log10(r)) / 10.0))
    return betas


def mc_g_min(seed: int, radius_m: float, topologies: int, rounds: int) -> list[float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(topologies):
        betas = draw_topology(rng, radius_m)
        for _ in range(rounds):
            while True:
                hs = [abs(complex(rng.gauss(0, 1), rng.gauss(0, 1))) / math.sqrt(2.0) for _ in range(N_CLIENTS)]
                if min(hs) >= H_CUT:
                    break
            samples.append(min(math.sqrt(b) * h for b, h in zip(betas, hs)))
    samples.sort()
    return samples


def quantile(sorted_vals: list[float], p: float) -> float:
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]


def cross_check() -> str:
    """If torch + full_system_0810 are available, verify formula parity."""
    try:
        from full_system_0810 import PhyConfig, scaling_limits  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - torch-less machines
        return f"skipped ({type(exc).__name__})"
    phy = PhyConfig()
    s = math.ceil(D_MODEL / phy.subcarriers)
    k = 4038
    g = 2.5e-6
    lim = scaling_limits(phy, s, k, g, D_MODEL)
    assert abs(lim["noise_tax_sqrt_f"] - noise_tax_sqrt_f(phy.epsilon)) < 1e-9
    assert abs(lim["eps_loose_k"] - eps_loose(k)) / eps_loose(k) < 1e-9
    mine = sigma_dp_over_ctx(phy.epsilon, k)
    assert abs(lim["sigma_dp_over_ctx"] - mine) / mine < 1e-6, (lim["sigma_dp_over_ctx"], mine)
    b_pow = g * math.sqrt(s * phy.subcarriers * phy.p_cap_w) / (phy.c_tx * math.sqrt(k))
    assert abs(lim["b_star"] - b_pow / noise_tax_sqrt_f(phy.epsilon)) / lim["b_star"] < 1e-9
    return "passed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("exp_0810/results/exp3_regime_diagnosis"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--topologies", type=int, default=5)
    parser.add_argument("--mc-rounds", type=int, default=400)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0])
    parser.add_argument("--kd-ratios", type=float, nargs="+", default=[2.5e-4, 1e-3, 2.5e-3, 1e-2, 0.05, 0.15])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    s_symbols = math.ceil(D_MODEL / SUBCARRIERS)
    check = cross_check()
    print(f"[diag] cross-check vs full_system_0810.scaling_limits: {check}")
    print(f"[diag] sigma_sc = {sigma_sc():.4e} sqrt(W); floor of margin*sqrt(F) = "
          f"{math.sqrt(2.0 * D_MODEL / N_CLIENTS):.1f}")

    g_sorted = mc_g_min(args.seed, CELL_RADIUS_M, args.topologies, args.mc_rounds)
    g_p50 = quantile(g_sorted, 0.50)
    rho_p50 = 2.0 * g_p50 * math.sqrt(s_symbols * SUBCARRIERS * 10.0 ** ((P_CAP_DBM - 30.0) / 10.0)) / sigma_sc()
    print(f"[diag] median min|g| = {g_p50:.3e}; rho = {rho_p50:.3e} "
          f"(intrinsic free region unreachable: floor << rho)")

    rows: list[dict] = []
    for eps in args.epsilons:
        for kd in args.kd_ratios:
            k = max(1, int(D_MODEL * kd))
            rows.append({
                "epsilon": eps,
                "kd": kd,
                "k": k,
                "sigma_dp_over_ctx": sigma_dp_over_ctx(eps, k),
                "eps_loose_k": eps_loose(k),
                "noise_tax_sqrt_f": noise_tax_sqrt_f(eps),
                "sig_over_art_amp": sig_over_art(eps, k),
            })
    with (args.output_dir / "dp_design_grid.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # k needed to place eps_loose at a target epsilon
    targets = {}
    for eps_t in (5.0, 10.0, 15.0, 20.0, 30.0):
        m = privacy_margin(eps_t)
        targets[f"eps_loose={eps_t:g}"] = int((N_CLIENTS * m) ** 2 / 2.0)
    (args.output_dir / "transition_summary.json").write_text(json.dumps({
        "cross_check": check,
        "rho_p50": rho_p50,
        "margin_sqrtF_floor": math.sqrt(2.0 * D_MODEL / N_CLIENTS),
        "k_for_target_eps_loose": targets,
    }, indent=2), encoding="utf-8")

    lines = [
        "# Exp3 DP design diagnosis (artificial-noise top-up)",
        "",
        f"- cross-check vs `scaling_limits`: {check}",
        f"- intrinsic free region: needs margin*sqrt(F) >= rho = {rho_p50:.3e}; "
        f"parameter-free floor sqrt(2d/N) = {math.sqrt(2.0 * D_MODEL / N_CLIENTS):.1f} -> unreachable, "
        "artificial noise is always active at eps <= 30 (reported honestly).",
        "- the operational transition is eps_loose(k) = (sqrt(2k)/N + sqrt(ln 1/dlt))^2 - ln 1/dlt:",
        "",
        "| k for eps_loose target | k | k/d |",
        "|---|---:|---:|",
    ]
    for name, kk in targets.items():
        lines.append(f"| {name} | {kk} | {kk / D_MODEL:.2e} |")
    lines += [
        "",
        "| eps | k/d | sigma_dp/c_tx | eps_loose(k) | sqrt(F) | sig/art amp |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['epsilon']:g} | {r['kd']:g} | {r['sigma_dp_over_ctx']:.2f} "
            f"| {r['eps_loose_k']:.1f} | {r['noise_tax_sqrt_f']:.1f} | {r['sig_over_art_amp']:.2f} |"
        )
    lines += [
        "",
        "Reading: `sigma_dp/c_tx` <= 1 marks the loose regime (accuracy saturating);",
        "`sig/art amp` >= 1 means the sparse waveform structure is visible above the",
        "injected DP noise in the PAPR/clipping statistics.  Suggested exp3 grid:",
        "eps in {1, 2.5, 5, 10, 15, 20, 30} at the exp1-selected k*.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[diag] wrote {args.output_dir}/dp_design_grid.csv, transition_summary.json, summary.md")


if __name__ == "__main__":
    main()
