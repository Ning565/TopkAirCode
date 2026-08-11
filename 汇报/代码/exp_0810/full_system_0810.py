#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""New-scenario OFDM-AirComp-FL common system for experiments 1 and 2.

This module implements the journal communication and receiver chain using the
algorithm/error interfaces in the latest `convergence_analysis.tex` and
`v8.tex`.  Experiment 0810 deliberately extends their intrinsic-noise privacy
interface with the artificial-noise top-up declared below.
It deliberately does NOT import anything from `exp/common/full_system.py`
(the old scenario code is kept untouched for later cross-checking).

Scenario summary implemented here
---------------------------------
Topology / channel
  * Single cell, N=20 single-antenna clients, area-uniform in a ring
    [r_min=10 m, R=250 m]; one topology per seed, fixed during training.
  * Log-distance path loss PL(r)=PL0+10*alpha*log10(r/r0), PL0=30 dB,
    r0=1 m, alpha=3  ->  beta_i = 1e-3 * r_i^{-alpha}.
  * Per-round Rayleigh block fading h_i^t ~ CN(0,1); effective channel
    g_i^t = sqrt(beta_i) * h_i^t.
  * Fixed participation: a candidate burst is a counted logical round only
    if |h_i^t| >= h_cut = 0.1 for ALL clients; otherwise the burst is
    redrawn (physical-layer wait/re-schedule).  Denominator is always N.

Noise / power
  * sigma_sc^2 = N0 * delta_f * F with N0=-174 dBm/Hz, delta_f=15 kHz,
    NF=5 dB  ->  sigma_sc^2 ~= 1.89e-16 W  (-127.2 dBm).
  * Whole-burst average transmit power budget P_cap (default 20 dBm).

Per-round public scaling and privacy (artificial-noise top-up)
  * c_tx is a declared, public, round-independent coordinate threshold.  It
    is not recomputed from the optimizer learning rate.
  * B_P^t(k) = min_i |g_i^t|*sqrt(S*M*P_cap)/(c_tx*sqrt(k)).
  * Every client adds real Gaussian noise on the complete public d-coordinate
    grid.  Thermal receiver noise is credited when calibrating the minimum
    required artificial-noise variance.
  * The conservative expected-power-feasible scale is
        b_t^*(k)=B_P^t(k)/sqrt(1+2d/(N*margin(epsilon)^2)).
    Privacy is per communication round; no sqrt(T) composition factor is used.

Receiver front end (PAPR / AGC / radial clipping, dual-line statistics)
  * One real model coordinate per complex data subcarrier, M=1024,
    S = ceil(d/M) OFDM symbols, sequential mapping j(q,m)=(q-1)M+m+1.
  * Frequency-domain aggregation Y = b_t * sum_i s_i + Z,
    Z ~ CN(0, sigma_sc^2), noise generated in frequency domain and passed
    through the SAME oversampled unitary IFFT as the signal.
  * Power-preserving oversampling: Q = L_os*M, zero padding scaled by
    sqrt(Q/M), unitary IFFT.
  * Structure line (noiseless r_sig): per-symbol PAPR (all-zero symbols
    excluded, silent ratio reported separately) and round-normalized
    peak-stress PSR_q = max_n|r_sig|^2 / ((1/SQ)*sum|r_sig|^2).
  * Hardware line (noisy r_rx): ideal per-round RMS-AGC
    P_avg = (1/SQ)*sum|r_rx|^2, A_rms = sqrt(P_avg), a_t = 1/A_rms,
    phase-preserving radial limiter C_gamma with gamma = 10^(B_clip/20),
    B_clip in {3,6,9,inf} dB.  A_max^t = gamma*A_rms^t.
  * Exact input-referred residual energy (theorem term):
    E_clip^t = (M/Q) * sum_{l,n} (|r_rx| - A_max^t)_+^2.
  * Recovery: clip -> explicit inverse AGC (multiply A_rms) -> unitary FFT
    -> extract data bins -> * sqrt(M/Q) -> Re -> / (b_t*N).
  * NMSE_clip (same noise realization, clip vs bypass) and NMSE_total
    (vs s_ideal = (1/N) sum_i s_i) are both reported.

The FL part (Top-k / Rand-k / Full with error feedback and element-wise
clipping at the independent public threshold c_tx) mirrors the algorithm flow
of the latest appendix.
"""

from __future__ import annotations

import gzip
import math
import pickle
import random
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PhyConfig:
    """Physical-layer parameters of the new scenario (整体场景.md §1-§5)."""

    num_clients: int = 20            # N
    cell_radius_m: float = 250.0     # R
    min_distance_m: float = 10.0     # r_min
    pl0_db: float = 30.0             # PL0 at r0 = 1 m
    pl_exponent: float = 3.0         # alpha
    ref_distance_m: float = 1.0      # r0

    n0_dbm_hz: float = -174.0        # thermal noise PSD
    nf_db: float = 5.0               # base-station noise figure
    subcarriers: int = 1024          # M
    subcarrier_spacing_hz: float = 15e3   # delta f
    oversampling: int = 4            # L_os
    h_cut: float = 0.1               # per-client feasibility threshold

    p_cap_dbm: float = 20.0          # whole-burst average power budget
    epsilon: float = 15.0            # per-round DP epsilon (journal-range baseline)
    delta: float = 1e-3              # per-round DP delta
    # Independent public coordinate threshold.  Keeping this separate from
    # eta*tau makes the privacy/power declaration invariant to optimizer
    # schedules and is the experiment-0810 protocol convention.
    # Empirical 0811 baseline.  A first-round FEMNIST scale audit found that
    # 0.02 makes the recovered DP noise unnecessarily large, while values
    # below 0.005 saturate almost every selected Top-k coordinate.  0.01 is
    # the declared compromise; it remains independent of eta and tau.
    c_tx: float = 0.01

    adc_backoff_db: float = 6.0      # B_clip; math.inf disables clipping
    max_burst_redraws: int = 10000   # safety bound on feasibility redraws

    def __post_init__(self) -> None:
        if self.num_clients <= 0 or self.subcarriers <= 0 or self.oversampling <= 0:
            raise ValueError("client/subcarrier/oversampling counts must be positive")
        if not (0.0 < self.delta < 1.0) or self.epsilon <= 0.0:
            raise ValueError("per-round DP parameters require epsilon>0 and 0<delta<1")
        if self.c_tx <= 0.0 or self.p_cap_w <= 0.0:
            raise ValueError("c_tx and P_cap must be positive")
        if not (0.0 <= self.h_cut < 1.0):
            raise ValueError("h_cut must lie in [0,1)")

    @property
    def sigma_sc2(self) -> float:
        n0_w_hz = 10.0 ** ((self.n0_dbm_hz - 30.0) / 10.0)
        noise_factor = 10.0 ** (self.nf_db / 10.0)
        return n0_w_hz * self.subcarrier_spacing_hz * noise_factor

    @property
    def sigma_sc(self) -> float:
        return math.sqrt(self.sigma_sc2)

    @property
    def p_cap_w(self) -> float:
        return 10.0 ** ((self.p_cap_dbm - 30.0) / 10.0)

    @property
    def gamma(self) -> float:
        if math.isinf(self.adc_backoff_db):
            return math.inf
        return 10.0 ** (self.adc_backoff_db / 20.0)

    def privacy_margin(self) -> float:
        log_delta = math.log(1.0 / self.delta)
        return math.sqrt(self.epsilon + log_delta) - math.sqrt(log_delta)


@dataclass
class LearnConfig:
    """FL training parameters (kept in line with the previous code base)."""

    seed: int = 2026
    device: str = "cuda:3"
    mnist_root: str = "data/MNIST/raw"
    femnist_path: str = "data/femnist/femnist_train.pkl"
    femnist_test_path: str = "data/femnist/femnist_test.pkl"

    rounds: int = 200
    local_steps: int = 5             # tau
    batch_size: int = 64
    eval_batch_size: int = 512
    eval_every: int = 1
    lr: float = 0.05                 # eta
    dirichlet_alpha: float = 0.3

    error_feedback_methods: Tuple[str, ...] = ("topk", "randk")
    randk_mask_mode: str = "common"  # one shared support per round
    # Optional post-processing denoising with the PUBLIC Rand-k mask only.
    # Disabled by default: the theorem counts channel noise on all d coords.
    randk_public_mask_denoise: bool = False


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Topology and per-round channel
# ---------------------------------------------------------------------------


@dataclass
class Topology:
    distances_m: np.ndarray          # (N,)
    beta: np.ndarray                 # (N,) large-scale power gains
    pathloss_db: np.ndarray          # (N,)


def draw_topology(phy: PhyConfig, seed: int) -> Topology:
    """Area-uniform client positions in the ring [r_min, R] (整体场景 §2)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=phy.num_clients)
    r = np.sqrt(phy.min_distance_m ** 2 + u * (phy.cell_radius_m ** 2 - phy.min_distance_m ** 2))
    pl_db = phy.pl0_db + 10.0 * phy.pl_exponent * np.log10(r / phy.ref_distance_m)
    beta = 10.0 ** (-pl_db / 10.0)
    return Topology(distances_m=r, beta=beta, pathloss_db=pl_db)


@dataclass
class RoundChannel:
    h: np.ndarray                    # (N,) complex small-scale fading
    g_abs: np.ndarray                # (N,) |g_i^t|
    redraws: int                     # candidate bursts discarded before success


def draw_round_channel(phy: PhyConfig, topo: Topology, rng: np.random.Generator) -> RoundChannel:
    """Draw h ~ CN(0,1); redraw until |h_i| >= h_cut for all i (fixed participation)."""
    redraws = 0
    while True:
        h = (rng.standard_normal(phy.num_clients) + 1j * rng.standard_normal(phy.num_clients)) / math.sqrt(2.0)
        if np.all(np.abs(h) >= phy.h_cut):
            break
        redraws += 1
        if redraws > phy.max_burst_redraws:
            raise RuntimeError("feasible burst not found; check h_cut / N")
    g_abs = np.sqrt(topo.beta) * np.abs(h)
    return RoundChannel(h=h, g_abs=g_abs, redraws=redraws)


# ---------------------------------------------------------------------------
# Per-round closed-form public scaling b_t^*(k)
# ---------------------------------------------------------------------------


def scaling_limits(
    phy: PhyConfig, s_symbols: int, k: int, g_abs_min: float, d_model: int
) -> Dict[str, float]:
    """Per-round public scaling under the artificial-noise top-up DP design.

    Mechanism (Koda'20 / Wei JSAC'22 / Liu TWC'24 lineage): the transmit scale
    is power-limited, and every client injects real Gaussian noise on the full
    d-coordinate public grid so that thermal + artificial noise meets the
    per-round client-level (eps, delta)-DP target.  All quantities are public
    (c_tx, k, N, eps, delta, sigma_sc, broadcast b_t): clients need no CSI.

    Closed forms (update domain, real aligned axis):
      * DP condition:      (b Delta(k))^2 <= margin^2 (sigma_sc^2 + 2 b^2 N sigma_a^2)
      * requirement:       N sigma_a0^2 = Delta(k)^2 / (2 margin^2)   (no thermal credit)
      * power tax:         F = 1 + 2 d / (N margin^2),  b_t = B_P^t(k) / sqrt(F)
      * thermal credit:    sigma_a^2 = max(0, Delta^2/margin^2 - sigma_sc^2/b^2) / (2N)
      * recovered noise:   sigma_dp = sqrt(sigma_sc^2/2 + b^2 N sigma_a^2) / (b N)
                           = Delta(k) / (sqrt(2) N margin)  when the credit is inactive
      * loose threshold:   eps_loose(k) solves sigma_dp = c_tx (k-driven, the
                           sparsity-buys-privacy knob):
                           eps_loose = (sqrt(2k)/N + sqrt(ln(1/delta)))^2 - ln(1/delta)
      * intrinsic free region (sigma_a = 0) needs margin*sqrt(F) >= rho with
        rho = 2 g_min sqrt(SM P_cap)/sigma_sc; since margin*sqrt(F) >=
        sqrt(2d/N), this is unreachable at small eps on healthy links --
        reported as eps_free_intrinsic for honesty, not as a claim.
    """
    sqrt_k = math.sqrt(max(1, k))
    delta_k = 2.0 * phy.c_tx * sqrt_k
    m = phy.privacy_margin()
    ln_d = math.log(1.0 / phy.delta)
    n = float(phy.num_clients)

    b_power = g_abs_min * math.sqrt(s_symbols * phy.subcarriers * phy.p_cap_w) / (phy.c_tx * sqrt_k)
    b_priv_intrinsic = phy.sigma_sc * m / delta_k

    tax_f = 1.0 + 2.0 * d_model / (n * m * m)
    b_star = b_power / math.sqrt(tax_f)

    sigma_a_sq = max(0.0, delta_k * delta_k / (m * m) - phy.sigma_sc2 / (b_star * b_star)) / (2.0 * n)
    v_real = phy.sigma_sc2 / 2.0 + b_star * b_star * n * sigma_a_sq
    sigma_dp = math.sqrt(v_real) / (b_star * n)
    expected_power_worst = (
        b_star * b_star * (max(1, k) * phy.c_tx * phy.c_tx + d_model * sigma_a_sq)
        / (s_symbols * phy.subcarriers * g_abs_min * g_abs_min)
    )

    eps_loose = (math.sqrt(2.0 * max(1, k)) / n + math.sqrt(ln_d)) ** 2 - ln_d
    rho = 2.0 * g_abs_min * math.sqrt(s_symbols * phy.subcarriers * phy.p_cap_w) / phy.sigma_sc
    m_free_sq = max(0.0, rho * rho - 2.0 * d_model / n)
    eps_free = (math.sqrt(m_free_sq) + math.sqrt(ln_d)) ** 2 - ln_d

    # Numerical invariant for the real-axis Gaussian mechanism:
    # (b Delta)^2 <= margin^2 (sigma_sc^2 + 2 b^2 N sigma_a^2).
    dp_lhs = (b_star * delta_k) ** 2
    dp_rhs = m * m * (phy.sigma_sc2 + 2.0 * b_star * b_star * n * sigma_a_sq)
    if dp_lhs > dp_rhs * (1.0 + 1e-10):
        raise RuntimeError("artificial-noise calibration violated the DP invariant")

    return {
        "b_power": b_power,
        "b_star": b_star,
        "noise_tax_sqrt_f": math.sqrt(tax_f),
        "sigma_a_client": math.sqrt(sigma_a_sq),
        "sigma_dp": sigma_dp,
        "sigma_dp_over_ctx": sigma_dp / phy.c_tx,
        "eps_loose_k": eps_loose,
        "free_intrinsic": 1.0 if sigma_a_sq == 0.0 else 0.0,
        "b_privacy_intrinsic": b_priv_intrinsic,
        "eps_free_intrinsic": eps_free,
        "regime": "loose" if sigma_dp <= phy.c_tx else "dp_noise",
        "dp_lhs_over_rhs": dp_lhs / max(dp_rhs, 1e-300),
        # Gaussian artificial noise is unbounded, so its power statement is in
        # expectation.  This diagnostic uses the worst aligned client channel.
        "expected_power_worst_w": expected_power_worst,
        "expected_power_utilization": expected_power_worst / phy.p_cap_w,
    }


# ---------------------------------------------------------------------------
# OFDM AirComp channel with oversampling / AGC / radial clipping
# ---------------------------------------------------------------------------


class OFDMAirCompChannel:
    """Full receive chain of PAPR场景.md §2-§3 with dual-line statistics.

    All waveform processing is done on CPU in complex128 so that physically
    small magnitudes (b_t ~ 1e-9 ... 1e-4, sigma_sc ~ 1.4e-8) remain accurate.
    """

    def __init__(self, phy: PhyConfig, d: int, noise_seed: int = 777):
        self.phy = phy
        self.d = d
        self.M = phy.subcarriers
        self.S = math.ceil(d / self.M)
        self.Q = phy.oversampling * self.M
        self.padded = self.S * self.M
        self.noise_seed = noise_seed
        # Bin split for standard band-limited interpolation zero padding:
        # low half at the start, high half at the end of the length-Q grid.
        self.half = self.M // 2

    # -- resource mapping ---------------------------------------------------
    def pack(self, x: torch.Tensor) -> torch.Tensor:
        """Flat real vector(s) (..., d) -> (..., S, M) sequential mapping."""
        lead = x.shape[:-1]
        out = torch.zeros(*lead, self.padded, dtype=torch.float64)
        out[..., : self.d] = x.to(torch.float64)
        return out.reshape(*lead, self.S, self.M)

    def unpack(self, grid: torch.Tensor) -> torch.Tensor:
        return grid.reshape(-1)[: self.d]

    # -- oversampling -------------------------------------------------------
    def _oversample_ifft(self, y_freq: torch.Tensor) -> torch.Tensor:
        """(S, M) complex -> (S, Q) unitary-IFFT time waveform, power preserving."""
        y_os = torch.zeros(self.S, self.Q, dtype=torch.complex128)
        y_os[:, : self.half] = y_freq[:, : self.half]
        y_os[:, self.Q - (self.M - self.half):] = y_freq[:, self.half:]
        y_os = y_os * math.sqrt(self.Q / self.M)
        return torch.fft.ifft(y_os, dim=-1, norm="ortho")

    def _fft_extract(self, r_time: torch.Tensor) -> torch.Tensor:
        """(S, Q) time waveform -> (S, M) data bins with inverse scaling."""
        spec = torch.fft.fft(r_time, dim=-1, norm="ortho")
        y = torch.zeros(self.S, self.M, dtype=torch.complex128)
        y[:, : self.half] = spec[:, : self.half]
        y[:, self.half:] = spec[:, self.Q - (self.M - self.half):]
        return y * math.sqrt(self.M / self.Q)

    # -- main entry ---------------------------------------------------------
    def transmit_round(
        self,
        signals: List[torch.Tensor],
        b_star: float,
        round_idx: int,
        collect_papr_samples: bool = False,
        sigma_a: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """One counted logical round: aggregate, clip, recover, measure.

        `sigma_a` is the per-client artificial DP noise std (update domain,
        real aligned axis, full d-coordinate grid).  The aggregate of the N
        independent client injections is drawn as one N(0, N sigma_a^2)
        vector, which is statistically identical.  The structure line r_sig
        keeps the pure sparse aggregate (no artificial, no thermal noise);
        the hardware line r_rx carries signal + artificial + thermal.
        """
        phy = self.phy
        n_clients = len(signals)
        stack = torch.stack([s.to(torch.float64) for s in signals], dim=0)
        s_ideal = stack.mean(dim=0)                       # (d,) true average
        agg = stack.sum(dim=0)                            # sum_i s_i

        freq_sig = b_star * self.pack(agg)                # structure line (S, M)
        if sigma_a > 0.0:
            gen_a = torch.Generator(device="cpu").manual_seed(self.noise_seed + 292663 * (round_idx + 1))
            art = torch.randn(self.d, generator=gen_a, dtype=torch.float64) * (
                math.sqrt(float(n_clients)) * sigma_a
            )
            freq_tx = b_star * self.pack(agg + art)       # transmitted composite
        else:
            freq_tx = freq_sig
        # Frequency-domain complex noise CN(0, sigma_sc^2), same IFFT as signal.
        gen = torch.Generator(device="cpu").manual_seed(self.noise_seed + 104729 * (round_idx + 1))
        noise_std = math.sqrt(phy.sigma_sc2 / 2.0)
        nr = torch.randn(self.S, self.M, generator=gen, dtype=torch.float64) * noise_std
        ni = torch.randn(self.S, self.M, generator=gen, dtype=torch.float64) * noise_std

        y_sig = torch.complex(freq_sig, torch.zeros_like(freq_sig))
        y_rx = torch.complex(freq_tx + nr, ni)

        r_sig = self._oversample_ifft(y_sig)              # structure line
        r_rx = self._oversample_ifft(y_rx)                # hardware line

        # ---- structure line: PAPR (noiseless) + PSR ------------------------
        p_sig = r_sig.abs().pow(2)                        # (S, Q)
        sym_pow = p_sig.mean(dim=-1)                      # (S,)
        sym_peak = p_sig.max(dim=-1).values               # (S,)
        silent = sym_pow <= 0.0
        active = ~silent
        if int(active.sum()) > 0:
            papr = sym_peak[active] / sym_pow[active]
            papr_db = 10.0 * torch.log10(papr)
            papr_mean_db = float(papr_db.mean())
            papr_p99_db = float(torch.quantile(papr_db, 0.99))
            papr_p999_db = float(torch.quantile(papr_db, 0.999))
            papr_max_db = float(papr_db.max())
        else:
            papr_db = torch.zeros(0, dtype=torch.float64)
            papr_mean_db = papr_p99_db = papr_p999_db = papr_max_db = float("nan")
        round_sig_pow = float(p_sig.mean())               # (1/SQ) sum |r_sig|^2
        if round_sig_pow > 0.0:
            psr = sym_peak / round_sig_pow                # PSR_q^t
            psr_round_db = float(10.0 * torch.log10(psr.max()))
        else:
            psr_round_db = float("nan")

        # ---- hardware line: ideal per-round RMS-AGC + radial clipping ------
        p_avg = float(r_rx.abs().pow(2).mean())           # P_avg^t
        a_rms = math.sqrt(p_avg)                          # A_rms^t
        gamma = phy.gamma
        r_bar = r_rx / a_rms                              # AGC-normalized
        mag_bar = r_bar.abs()
        if math.isinf(gamma):
            r_bar_clip = r_bar                            # true bypass
            clip_ratio = 0.0
            d_clip = 0.0
            e_clip = 0.0
        else:
            scale = torch.clamp(gamma / mag_bar.clamp_min(1e-300), max=1.0)
            r_bar_clip = r_bar * scale
            over = (mag_bar - gamma).clamp_min(0.0)
            clip_ratio = float((mag_bar > gamma).to(torch.float64).mean())
            denom = float(mag_bar.pow(2).sum())
            d_clip = float(over.pow(2).sum()) / max(denom, 1e-300)
            # Exact theorem residual: E_clip = (M/Q) sum (|r_rx| - A_max)_+^2
            e_clip = (self.M / self.Q) * float(over.pow(2).sum()) * (a_rms ** 2)

        # ---- recovery: inverse AGC -> FFT -> bins -> Re -> /(b N) ----------
        r_clip_phys = r_bar_clip * a_rms                  # explicit a_t^{-1}
        y_clip = self._fft_extract(r_clip_phys)
        rec_clip = self.unpack(y_clip.real) / (b_star * n_clients)

        # Bypass chain with the SAME noise realization (for NMSE_clip).
        y_lin = self._fft_extract(r_rx)
        rec_lin = self.unpack(y_lin.real) / (b_star * n_clients)

        diff_clip = rec_clip - rec_lin
        nmse_clip = float(diff_clip.pow(2).sum() / rec_lin.pow(2).sum().clamp_min(1e-300))
        diff_tot = rec_clip - s_ideal
        nmse_total = float(diff_tot.pow(2).sum() / s_ideal.pow(2).sum().clamp_min(1e-300))

        metrics = {
            "papr_mean_db": papr_mean_db,
            "papr_p99_db": papr_p99_db,
            "papr_p999_db": papr_p999_db,
            "papr_max_db": papr_max_db,
            "psr_round_db": psr_round_db,
            "silent_symbol_ratio": float(silent.to(torch.float64).mean()),
            "p_avg": p_avg,
            "a_rms": a_rms,
            "rho_clip": clip_ratio,
            "d_clip": d_clip,
            "e_clip": e_clip,
            "e_clip_over_b2": e_clip / (b_star * b_star),
            "nmse_clip": nmse_clip,
            "nmse_total": nmse_total,
            "sigma_a_client": sigma_a,
            "art_over_thermal_db": (
                10.0 * math.log10(2.0 * b_star * b_star * n_clients * sigma_a * sigma_a / phy.sigma_sc2)
                if sigma_a > 0.0 else -300.0
            ),
            "eff_noise_std": math.sqrt(phy.sigma_sc2 / 2.0 + b_star * b_star * n_clients * sigma_a * sigma_a)
            / (b_star * n_clients),
            "eff_noise_over_ctx": math.sqrt(phy.sigma_sc2 / 2.0 + b_star * b_star * n_clients * sigma_a * sigma_a)
            / (b_star * n_clients * phy.c_tx),
        }
        if collect_papr_samples:
            metrics["papr_db_samples"] = papr_db.numpy().copy()
        return rec_clip.to(torch.float32), metrics


# ---------------------------------------------------------------------------
# Sparsification / error feedback (appendix §A.1 algorithm flow)
# ---------------------------------------------------------------------------


def compress_update(
    v: torch.Tensor,
    method: str,
    ratio: float,
    gen: torch.Generator,
    common_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
    """Return (sparse vector, boolean mask, retained-energy ratio, k)."""
    d = v.numel()
    k = d if method == "full" else max(1, int(d * ratio))
    if method == "full" or k >= d:
        mask = torch.ones(d, dtype=torch.bool)
        return v.clone(), mask, 1.0, min(k, d)
    if method == "topk":
        _, idx = torch.topk(v.abs(), k)
    elif method == "randk":
        idx = common_idx if common_idx is not None else torch.randperm(d, generator=gen)[:k]
    else:
        raise ValueError(method)
    out = torch.zeros_like(v)
    out[idx] = v[idx]
    mask = torch.zeros(d, dtype=torch.bool)
    mask[idx] = True
    retained = float(out.pow(2).sum() / v.pow(2).sum().clamp_min(1e-30))
    return out, mask, retained, k


def elementwise_clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.clamp(x, min=-threshold, max=threshold)


# ---------------------------------------------------------------------------
# Datasets and models (kept identical to the previous code base for later
# alignment of the learning side; only the communication model is new)
# ---------------------------------------------------------------------------


def _read_idx_images(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image magic number {magic}")
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    return data.reshape(n, 1, rows, cols).astype(np.float32)


def _read_idx_labels(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label magic number {magic}")
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    return data.astype(np.int64)


def _idx_path(root: Path, stem: str) -> Path:
    for cand in (root / stem, root / f"{stem}.gz"):
        if cand.exists():
            return cand
    raise FileNotFoundError(stem)


def make_dirichlet(labels: np.ndarray, num_clients: int, alpha: float, seed: int) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    client_lists: Dict[int, List[int]] = {cid: [] for cid in range(num_clients)}
    for cls in range(num_classes):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        props = rng.dirichlet(np.repeat(alpha, num_clients))
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for cid, part in enumerate(np.split(idx, cuts)):
            client_lists[cid].extend(part.tolist())
    all_idx = np.arange(len(labels))
    for cid in range(num_clients):
        if len(client_lists[cid]) < 32:
            extra = rng.choice(all_idx, size=32 - len(client_lists[cid]), replace=True)
            client_lists[cid].extend(extra.tolist())
        rng.shuffle(client_lists[cid])
    return {cid: np.array(v, dtype=np.int64) for cid, v in client_lists.items()}


def load_mnist(cfg: LearnConfig, num_clients: int):
    root = Path(cfg.mnist_root)
    x_train = _read_idx_images(_idx_path(root, "train-images-idx3-ubyte"))
    y_train = _read_idx_labels(_idx_path(root, "train-labels-idx1-ubyte"))
    x_test = _read_idx_images(_idx_path(root, "t10k-images-idx3-ubyte"))
    y_test = _read_idx_labels(_idx_path(root, "t10k-labels-idx1-ubyte"))
    x_train = (x_train / 255.0 - 0.1307) / 0.3081
    x_test = (x_test / 255.0 - 0.1307) / 0.3081
    train = TensorDataset(torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train).long())
    test = TensorDataset(torch.from_numpy(x_test.astype(np.float32)), torch.from_numpy(y_test).long())
    clients = make_dirichlet(y_train, num_clients, cfg.dirichlet_alpha, cfg.seed)
    return train, test, clients, 10


def load_femnist(cfg: LearnConfig, num_clients: int):
    with open(cfg.femnist_path, "rb") as f:
        data = pickle.load(f)
    users = list(data["users"])
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(users)
    users = users[:num_clients]
    xs, ys = [], []
    clients: Dict[int, np.ndarray] = {}
    offset = 0
    for cid, user in enumerate(users):
        x = np.asarray(data["user_data"][user]["x"], dtype=np.float32).reshape(-1, 1, 28, 28)
        y = np.asarray(data["user_data"][user]["y"], dtype=np.int64)
        xs.append(x)
        ys.append(y)
        clients[cid] = np.arange(offset, offset + len(y), dtype=np.int64)
        offset += len(y)
    x_train = np.concatenate(xs, axis=0)
    y_train = np.concatenate(ys, axis=0)
    with open(cfg.femnist_test_path, "rb") as f:
        tdata = pickle.load(f)
    txs, tys = [], []
    for user in users:
        txs.append(np.asarray(tdata["user_data"][user]["x"], dtype=np.float32).reshape(-1, 1, 28, 28))
        tys.append(np.asarray(tdata["user_data"][user]["y"], dtype=np.int64))
    x_test = np.concatenate(txs, axis=0)
    y_test = np.concatenate(tys, axis=0)
    mean = x_train.mean()
    std = x_train.std() + 1e-5
    train = TensorDataset(torch.from_numpy(((x_train - mean) / std).astype(np.float32)), torch.from_numpy(y_train).long())
    test = TensorDataset(torch.from_numpy(((x_test - mean) / std).astype(np.float32)), torch.from_numpy(y_test).long())
    return train, test, clients, 62


class SimpleMLP(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class StableCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


def make_model(dataset: str, num_classes: int) -> nn.Module:
    return StableCNN(num_classes) if dataset == "femnist" else SimpleMLP(num_classes)


def model_dimension(dataset: str, num_classes: int) -> int:
    """Compute d from the instantiated current model; never use a stale literal."""
    return sum(p.numel() for p in make_model(dataset, num_classes).parameters())


def load_problem(cfg: LearnConfig, dataset: str, num_clients: int):
    if dataset == "mnist":
        return load_mnist(cfg, num_clients)
    if dataset == "femnist":
        return load_femnist(cfg, num_clients)
    raise ValueError(dataset)


def flatten_delta(model: nn.Module, base: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([(p.detach().cpu() - base[name]).reshape(-1) for name, p in model.named_parameters()]).float()


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()]).float()


def apply_update(model: nn.Module, update: torch.Tensor) -> None:
    with torch.no_grad():
        off = 0
        for p in model.parameters():
            n = p.numel()
            p.add_(update[off: off + n].view_as(p).to(p.device))
            off += n


# ---------------------------------------------------------------------------
# FL system: local training + physical channel per counted round
# ---------------------------------------------------------------------------


class FLSystem:
    """Federated trainer over the new-scenario AirComp channel."""

    def __init__(self, phy: PhyConfig, cfg: LearnConfig, dataset: str):
        self.phy = phy
        self.cfg = cfg
        self.dataset = dataset
        train, test, clients, num_classes = load_problem(cfg, dataset, phy.num_clients)
        self.train_set = train
        self.test_loader = DataLoader(test, batch_size=cfg.eval_batch_size, shuffle=False)
        self.clients = clients
        self.num_classes = num_classes
        self.loaders = {
            cid: DataLoader(Subset(train, idx), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
            for cid, idx in clients.items()
        }
        self.iters = {cid: iter(loader) for cid, loader in self.loaders.items()}

    def next_batch(self, cid: int):
        try:
            return next(self.iters[cid])
        except StopIteration:
            self.iters[cid] = iter(self.loaders[cid])
            return next(self.iters[cid])

    def client_delta(self, model: nn.Module, cid: int) -> Tuple[torch.Tensor, float]:
        cfg = self.cfg
        local = make_model(self.dataset, self.num_classes).to(cfg.device)
        local.load_state_dict(model.state_dict())
        local.train()
        base = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        opt = torch.optim.SGD(local.parameters(), lr=cfg.lr)
        loss_fn = nn.CrossEntropyLoss()
        losses = []
        for _ in range(cfg.local_steps):
            x, y = self.next_batch(cid)
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(local(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        delta = flatten_delta(local, base)
        del local, opt
        return delta, float(np.mean(losses))

    def client_grad(self, model: nn.Module, cid: int, batches: int = 1) -> torch.Tensor:
        """Mini-batch gradient proxy of nabla f_i at the current global model."""
        cfg = self.cfg
        model.eval()
        loss_fn = nn.CrossEntropyLoss()
        grads = None
        for _ in range(batches):
            x, y = self.next_batch(cid)
            x, y = x.to(cfg.device), y.to(cfg.device)
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            g = torch.cat([p.grad.detach().cpu().reshape(-1) for p in model.parameters()]).float()
            grads = g if grads is None else grads + g
        model.zero_grad(set_to_none=True)
        return grads / batches

    def evaluate(self, model: nn.Module) -> Tuple[float, float]:
        model.eval()
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)
                logits = model(x)
                loss_sum += float(loss_fn(logits, y).item())
                correct += int((logits.argmax(dim=1) == y).sum().item())
                total += y.numel()
        return 100.0 * correct / max(1, total), loss_sum / max(1, total)

    # -- one full training run over the physical channel --------------------
    def run_training(self, method: str, ratio: float, log_prefix: str = "") -> List[Dict]:
        phy, cfg = self.phy, self.cfg
        set_seed(cfg.seed)
        # FLSystem is reused across compression candidates in experiment 2.
        # Recreate every iterator after resetting the RNG so all candidates see
        # the same shuffled client-batch streams (common-random-number design).
        self.iters = {cid: iter(loader) for cid, loader in self.loaders.items()}
        model = make_model(self.dataset, self.num_classes).to(cfg.device)
        d = sum(p.numel() for p in model.parameters())
        channel = OFDMAirCompChannel(phy, d, noise_seed=cfg.seed + 777)
        topo = draw_topology(phy, cfg.seed)
        chan_rng = np.random.default_rng(cfg.seed + 424242)
        k = d if method == "full" else max(1, int(d * ratio))
        use_ef = method in cfg.error_feedback_methods and method != "full"
        memories = {cid: torch.zeros(d) for cid in range(phy.num_clients)} if use_ef else None
        rows: List[Dict] = []
        acc, test_loss = float("nan"), float("nan")
        for r in range(cfg.rounds):
            rc = draw_round_channel(phy, topo, chan_rng)
            lim = scaling_limits(phy, channel.S, k, float(rc.g_abs.min()), d)
            if r == 0:
                print(
                    f"{log_prefix}[{self.dataset}/{method} r={ratio:.3f}] regime forecast: "
                    f"eps={phy.epsilon:g} -> b*={lim['b_star']:.3e} "
                    f"(power tax sqrt(F)={lim['noise_tax_sqrt_f']:.1f}), "
                    f"sigma_dp/c_tx={lim['sigma_dp_over_ctx']:.2f} ({lim['regime']}), "
                    f"eps_loose(k)={lim['eps_loose_k']:.1f}, "
                    f"free_intrinsic={bool(lim['free_intrinsic'])}",
                    flush=True,
                )
            gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 10007 * r + int(ratio * 1_000_000))
            common_idx = None
            if method == "randk" and k < d and cfg.randk_mask_mode == "common":
                common_idx = torch.randperm(d, generator=gen)[:k]
            signals, losses, retained = [], [], []
            mask_union = torch.zeros(d, dtype=torch.bool)
            for cid in range(phy.num_clients):
                raw, loss = self.client_delta(model, cid)
                inp = raw + memories[cid] if use_ef else raw
                sparse, mask, ret, _ = compress_update(inp, method, ratio, gen, common_idx)
                sparse = elementwise_clip(sparse, phy.c_tx)
                if use_ef:
                    # Only the sparsification residual is fed back; the
                    # pre-transmission clipping bias is NOT (appendix flow).
                    sel = inp.clone()
                    sel[~mask] = 0.0
                    memories[cid] = inp - sel
                signals.append(sparse)
                losses.append(loss)
                retained.append(ret)
                mask_union |= mask
            update, comm = channel.transmit_round(signals, lim["b_star"], r, sigma_a=lim["sigma_a_client"])
            if method == "randk" and cfg.randk_public_mask_denoise and common_idx is not None:
                keep = torch.zeros(d, dtype=torch.bool)
                keep[common_idx] = True
                update = update * keep.float()
            apply_update(model, update)
            if (r + 1) % cfg.eval_every == 0 or r == 0 or r == cfg.rounds - 1:
                acc, test_loss = self.evaluate(model)
            comm_scalar = {kk: vv for kk, vv in comm.items() if not isinstance(vv, np.ndarray)}
            row = {
                "dataset": self.dataset,
                "method": method,
                "round": r + 1,
                "ratio": ratio,
                "k": k,
                "d": d,
                "train_loss": float(np.mean(losses)),
                "test_accuracy": float(acc),
                "test_loss": float(test_loss),
                "retained_energy": float(np.mean(retained)),
                "burst_redraws": rc.redraws,
                **lim,
                **comm_scalar,
            }
            rows.append(row)
            if (r + 1) % max(1, cfg.eval_every) == 0 or r == 0:
                print(
                    f"{log_prefix}[{self.dataset}/{method} r={ratio:.3f}] round {r + 1}/{cfg.rounds} "
                    f"acc={acc:.2f} b*={lim['b_star']:.3e} ({lim['regime']}) "
                    f"papr99={comm['papr_p99_db']:.2f}dB rho={comm['rho_clip']:.2e} "
                    f"nmse_tot={comm['nmse_total']:.2e}",
                    flush=True,
                )
        return rows
