#!/usr/bin/env python3
"""Experiment 1: direct plug-in evaluation of the convergence bound.

This script deliberately contains no fitted objective weights and performs no
per-method normalization. It evaluates the static form of Eq. (79) using one
shared calibration trajectory and reproducible empirical estimates of the
otherwise unavailable neural-network constants.

The resulting quantity is an empirical plug-in bound diagnostic. It is only a
certified theorem bound when the reported learning-rate condition is satisfied
and the empirical constants are valid global upper bounds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = ROOT / "exp" / "common"
sys.path.insert(0, str(COMMON_DIR))

from full_system import (  # noqa: E402
    Config,
    Experiment,
    OFDMAirCompADC,
    apply_update,
    compress_update,
    elementwise_clip,
    load_femnist,
    load_mnist,
    make_model,
    power_privacy_limits,
    set_seed,
)


def ratio_grid(step: float) -> list[float]:
    count = int(round(1.0 / step))
    values = [round(i * step, 10) for i in range(1, count + 1)]
    if not math.isclose(values[-1], 1.0):
        raise ValueError("ratio step must divide 1.0")
    return values


def flatten_model(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()]).float()


def load_problem(cfg: Config, dataset: str):
    if dataset == "femnist":
        return load_femnist(cfg)
    if dataset == "mnist":
        return load_mnist(cfg)
    raise ValueError(dataset)


def fixed_objective_gradient(
    model: torch.nn.Module,
    dataset,
    indices: list[int],
    device: str,
    batch_size: int,
) -> tuple[torch.Tensor, float]:
    """Evaluate one deterministic empirical objective and its gradient."""

    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    model.eval()
    model.zero_grad(set_to_none=True)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss_sum = loss_fn(model(x), y)
        total_loss += float(loss_sum.detach().item())
        total += int(y.numel())
        (loss_sum / len(indices)).backward()
    gradient = torch.cat([p.grad.detach().cpu().reshape(-1) for p in model.parameters()]).float()
    model.zero_grad(set_to_none=True)
    return gradient, total_loss / max(1, total)


def sampled_client_gradient(exp: Experiment, model: torch.nn.Module, cid: int) -> torch.Tensor:
    """Draw one mini-batch gradient at a fixed model point."""

    x, y = exp.next_batch(cid)
    x, y = x.to(exp.cfg.device), y.to(exp.cfg.device)
    model.train()
    model.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    gradient = torch.cat([p.grad.detach().cpu().reshape(-1) for p in model.parameters()]).float()
    model.zero_grad(set_to_none=True)
    return gradient


def calibrate_clean_full_learning_rate(
    cfg: Config,
    dataset: str,
    candidates: list[float],
    rounds: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Choose one shared learning rate using only ideal Full training."""

    rows: list[dict[str, Any]] = []
    for lr in candidates:
        set_seed(cfg.seed)
        train, test, clients, num_classes = load_problem(cfg, dataset)
        exp = Experiment(cfg, dataset, train, test, clients, num_classes)
        model = make_model(dataset, num_classes).to(cfg.device)
        initial_accuracy, initial_loss = exp.evaluate(model)
        local_steps = cfg.local_steps_femnist if dataset == "femnist" else cfg.local_steps_mnist
        stable = True
        train_loss = math.nan
        for round_idx in range(rounds):
            updates = []
            losses = []
            for cid in range(cfg.num_clients):
                update, loss = exp.client_delta(model, cid, lr, local_steps)
                updates.append(update)
                losses.append(loss)
            update_mean = torch.stack(updates).mean(dim=0)
            if not bool(torch.isfinite(update_mean).all()):
                stable = False
                break
            apply_update(model, update_mean)
            train_loss = float(np.mean(losses))
            print(f"[lr-calibration] lr={lr:g} round={round_idx + 1}/{rounds}", flush=True)
        final_accuracy, final_loss = exp.evaluate(model) if stable else (math.nan, math.inf)
        stable = stable and math.isfinite(final_loss) and final_loss <= max(10.0, 4.0 * initial_loss)
        rows.append({
            "learning_rate": lr,
            "rounds": rounds,
            "initial_test_accuracy": initial_accuracy,
            "initial_test_loss": initial_loss,
            "final_train_loss": train_loss,
            "final_test_accuracy": final_accuracy,
            "final_test_loss": final_loss,
            "stable": stable,
        })
    valid = [row for row in rows if row["stable"]]
    if not valid:
        raise RuntimeError("No stable learning-rate candidate in clean Full calibration")
    best = min(valid, key=lambda row: (row["final_test_loss"], -row["final_test_accuracy"]))
    return float(best["learning_rate"]), rows


def public_clip_quantiles(raw_updates: list[list[torch.Tensor]], samples_per_update: int = 4096) -> dict[str, float]:
    """Estimate shared coordinate thresholds from pre-sparsification updates."""

    samples = []
    for round_idx, updates in enumerate(raw_updates):
        for cid, update in enumerate(updates):
            count = min(samples_per_update, update.numel())
            gen = torch.Generator(device="cpu").manual_seed(9176 + 1009 * round_idx + cid)
            idx = torch.randperm(update.numel(), generator=gen)[:count]
            samples.append(update[idx].abs())
    values = torch.cat(samples)
    return {
        "q90": float(torch.quantile(values, 0.90).item()),
        "q95": float(torch.quantile(values, 0.95).item()),
        "q99": float(torch.quantile(values, 0.99).item()),
        "sample_count": int(values.numel()),
    }


def public_l2_clip_quantiles(raw_updates: list[list[torch.Tensor]]) -> dict[str, float]:
    norms = torch.tensor(
        [float(torch.linalg.vector_norm(update).item()) for updates in raw_updates for update in updates],
        dtype=torch.float64,
    )
    return {
        "q90": float(torch.quantile(norms, 0.90).item()),
        "q95": float(torch.quantile(norms, 0.95).item()),
        "q99": float(torch.quantile(norms, 0.99).item()),
        "sample_count": int(norms.numel()),
    }


def l2_clip(update: torch.Tensor, threshold: float) -> torch.Tensor:
    norm = float(torch.linalg.vector_norm(update).item())
    if norm <= threshold:
        return update
    return update * (threshold / max(norm, 1e-20))


def experiment1_power_privacy_limits(
    cfg: Config,
    k: int,
    total_rounds: int,
) -> tuple[float, float, float, str]:
    if cfg.privacy_scope == "total_replace_one_coordinate":
        return power_privacy_limits(cfg, k, total_rounds)

    if cfg.privacy_scope != "per_round_client_l2":
        raise ValueError(f"Unknown privacy scope: {cfg.privacy_scope}")

    # PFELS-style power envelope: every selected coordinate is bounded by the
    # common update-norm threshold, while the add/remove client sensitivity of
    # the sparsified vector is bounded by that norm independently of k.
    update_bound = cfg.update_l2_clip
    b_power = cfg.h_th * math.sqrt(cfg.p_max) / (update_bound * math.sqrt(max(1, k)))
    log_delta = math.log(1.0 / cfg.delta)
    privacy_margin = math.sqrt(cfg.epsilon + log_delta) - math.sqrt(log_delta)
    b_privacy = cfg.sigma0 * privacy_margin / update_bound
    b_star = min(b_power, b_privacy)
    regime = "power" if b_power <= b_privacy else "privacy"
    return b_power, b_privacy, b_star, regime


def estimate_problem_constants(
    cfg: Config,
    dataset: str,
    calib_rounds: int,
    constant_estimation_samples: int,
) -> tuple[list[list[torch.Tensor]], dict[str, float], int]:
    """Collect one ideal trajectory and estimate k-independent constants."""

    train, test, clients, num_classes = load_problem(cfg, dataset)
    exp = Experiment(cfg, dataset, train, test, clients, num_classes)
    model = make_model(dataset, num_classes).to(cfg.device)
    d = sum(p.numel() for p in model.parameters())
    lr = cfg.lr_femnist if dataset == "femnist" else cfg.lr_mnist
    tau = cfg.local_steps_femnist if dataset == "femnist" else cfg.local_steps_mnist

    all_updates: list[list[torch.Tensor]] = []
    gradient_rounds: list[torch.Tensor] = []
    global_gradients: list[torch.Tensor] = []
    theta_before: list[torch.Tensor] = []
    losses: list[float] = []
    fixed_losses: list[float] = []
    fixed_gradient_rounds: list[torch.Tensor] = []
    stochasticity: list[float] = []
    subset_size = min(constant_estimation_samples, len(train))
    subset_gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 271828)
    fixed_indices = torch.randperm(len(train), generator=subset_gen)[:subset_size].tolist()

    for round_idx in range(calib_rounds):
        theta_before.append(flatten_model(model))
        fixed_gradient, fixed_loss = fixed_objective_gradient(
            model, train, fixed_indices, cfg.device, cfg.eval_batch_size
        )
        fixed_gradient_rounds.append(fixed_gradient)
        fixed_losses.append(fixed_loss)
        updates: list[torch.Tensor] = []
        round_losses: list[float] = []
        for cid in range(cfg.num_clients):
            stochastic_g1 = sampled_client_gradient(exp, model, cid)
            stochastic_g2 = sampled_client_gradient(exp, model, cid)
            stochasticity.append(float(0.5 * (stochastic_g1 - stochastic_g2).pow(2).sum().item()))
            update, loss = exp.client_delta(model, cid, lr, tau)
            updates.append(update)
            round_losses.append(loss)

        gradients = torch.stack([-u / (lr * tau) for u in updates])
        global_gradient = gradients.mean(dim=0)
        all_updates.append(updates)
        gradient_rounds.append(gradients)
        global_gradients.append(global_gradient)
        losses.extend(round_losses)

        # Advance a common ideal, noiseless trajectory. Every k candidate is
        # calibrated from these exact same updates.
        apply_update(model, torch.stack(updates).mean(dim=0))
        print(f"[constants] round {round_idx + 1}/{calib_rounds}", flush=True)

    heterogeneity = []
    for gradients, global_gradient in zip(gradient_rounds, global_gradients):
        heterogeneity.append(float((gradients - global_gradient).pow(2).sum(dim=1).mean().item()))

    secants = []
    for idx in range(1, len(fixed_gradient_rounds)):
        numerator = torch.linalg.vector_norm(fixed_gradient_rounds[idx] - fixed_gradient_rounds[idx - 1]).item()
        denominator = torch.linalg.vector_norm(theta_before[idx] - theta_before[idx - 1]).item()
        if denominator > 1e-12:
            secants.append(numerator / denominator)

    heterogeneity_array = np.asarray(heterogeneity, dtype=np.float64)
    stochasticity_array = np.asarray(stochasticity, dtype=np.float64)
    constants = {
        "L_hat": float(max(secants)) if secants else 1.0,
        "G_hat": 1.0,
        "kappa_sq_hat": float(max(heterogeneity)) if heterogeneity else 0.0,
        "kappa_sq_median": float(np.median(heterogeneity_array)) if heterogeneity else 0.0,
        "kappa_sq_q90": float(np.quantile(heterogeneity_array, 0.90)) if heterogeneity else 0.0,
        "kappa_sq_max": float(np.max(heterogeneity_array)) if heterogeneity else 0.0,
        "zeta_sq_hat": float(np.max(stochasticity_array)) if stochasticity else 0.0,
        "zeta_sq_median": float(np.median(stochasticity_array)) if stochasticity else 0.0,
        "zeta_sq_q90": float(np.quantile(stochasticity_array, 0.90)) if stochasticity else 0.0,
        "zeta_sq_max": float(np.max(stochasticity_array)) if stochasticity else 0.0,
        # Cross entropy is nonnegative, so f*=0 is a valid lower bound.
        "delta_f_hat": float(fixed_losses[0]) if fixed_losses else 0.0,
        "mean_calibration_loss": float(np.mean(losses)) if losses else 0.0,
        "fixed_objective_samples": subset_size,
        "fixed_objective_initial_loss": float(fixed_losses[0]) if fixed_losses else 0.0,
        "secant_count": len(secants),
    }
    return all_updates, constants, d


def bound_terms(
    cfg: Config,
    constants: dict[str, float],
    d: int,
    k: int,
    omega_hat: float,
    beta_sq_hat: float,
    mean_clip_energy: float,
    b_star: float,
) -> dict[str, Any]:
    eta = cfg.lr_femnist
    tau = cfg.local_steps_femnist
    n = cfg.num_clients
    L = constants["L_hat"]
    G = constants["G_hat"]
    kappa_sq = constants["kappa_sq_hat"]
    zeta_sq = constants["zeta_sq_hat"]
    delta_f = constants["delta_f_hat"]

    omega = min(1.0, max(1e-8, omega_hat))
    mu = 0.0 if omega >= 1.0 - 1e-12 else 2.0 * (1.0 - omega) * (2.0 - omega) / (omega * omega)
    gamma = 2.0 + 16.0 * L * L * mu * eta * eta * tau * tau * G * G
    r_term = 8.0 * L * L * mu * eta * eta * tau * tau * (2.0 * kappa_sq + zeta_sq)
    e_term = (eta * tau + 2.0 * L * eta * eta * tau * tau) * beta_sq_hat
    c_adc = 4.0 / (eta * tau) + L
    descent_coefficient = (
        eta
        * tau
        * (0.5 + 4.0 * L * eta * tau)
        * (1.0 + 32.0 * eta * eta * tau * tau * G * G * L * L)
    )
    psi = 2.0 * (2.0 * kappa_sq + zeta_sq) + eta * L * (0.5 + 4.0 * L * eta * tau) * (
        32.0 * tau * kappa_sq + 8.0 * zeta_sq
    )

    phi_gap = 16.0 * gamma * delta_f / (cfg.rounds * eta * tau)
    phi_clip = 16.0 * gamma * e_term / (eta * tau)
    phi_local = 16.0 * gamma * eta * tau * L * psi
    phi_memory = 16.0 * gamma * descent_coefficient * r_term / (eta * tau) + r_term
    learning_total = phi_gap + phi_clip + phi_local + phi_memory
    channel_base = 8.0 * L * cfg.sigma0 * cfg.sigma0 * d / (
        eta * tau * n * n * b_star * b_star
    )
    adc_base = 16.0 * c_adc * mean_clip_energy / (
        eta * tau * n * n * b_star * b_star
    )
    channel = gamma * channel_base
    adc = gamma * adc_base
    total = learning_total + channel + adc
    dominant_share = max(learning_total, channel, adc) / max(total, 1e-30)

    eta_limit_local = math.inf if L <= 0 else 1.0 / (128.0 * L * tau * G * G)
    eta_limit_memory = math.inf if L <= 0 else 1.0 / (tau * G * L * math.sqrt(1.0 + 384.0 * mu))
    eta_limit = min(eta_limit_local, eta_limit_memory)
    return {
        "omega_hat": omega,
        "mu_hat": mu,
        "Gamma_hat": gamma,
        "R_hat": r_term,
        "E_hat": e_term,
        "C_adc_hat": c_adc,
        "descent_coefficient_hat": descent_coefficient,
        "Psi_hat": psi,
        "bound_gap": phi_gap,
        "bound_pre_tx_clip": phi_clip,
        "bound_local_stochasticity": phi_local,
        "bound_memory": phi_memory,
        "bound_learning": learning_total,
        "bound_channel_base": channel_base,
        "bound_channel": channel,
        "bound_adc_base": adc_base,
        "bound_adc": adc,
        "bound_total": total,
        "dominant_share": dominant_share,
        "dominant_term": max(
            (("learning", learning_total), ("channel", channel), ("adc", adc)), key=lambda item: item[1]
        )[0],
        "strongly_dominant": bool(dominant_share > 0.95),
        "eta_limit": eta_limit,
        "eta_condition_satisfied": bool(eta <= eta_limit),
    }


def evaluate_grid(
    cfg: Config,
    raw_updates: list[list[torch.Tensor]],
    constants: dict[str, float],
    d: int,
    ratios: list[float],
) -> list[dict[str, Any]]:
    channel = OFDMAirCompADC(cfg, d)
    rows: list[dict[str, Any]] = []
    lr = cfg.lr_femnist
    tau = cfg.local_steps_femnist

    for method in ("topk", "randk"):
        for ratio_idx, ratio in enumerate(ratios):
            # Keep exactly the same floor rule used by compress_update().
            k = max(1, int(d * ratio))
            p_actual = k / d
            b_power, b_privacy, b_star, regime = experiment1_power_privacy_limits(cfg, k, cfg.rounds)
            memories = [torch.zeros(d) for _ in range(cfg.num_clients)]
            retained_numerator = 0.0
            retained_denominator = 0.0
            retained_samples: list[float] = []
            ef_cross_correlations: list[float] = []
            ef_memory_ratios: list[float] = []
            beta_sq_samples: list[float] = []
            clip_energies: list[float] = []
            normalized_clip_energies: list[float] = []
            papr_p99: list[float] = []
            effective_noise: list[float] = []

            for round_idx, updates in enumerate(raw_updates):
                gen = torch.Generator(device="cpu").manual_seed(
                    cfg.seed + 10007 * round_idx + int(ratio * 1_000_000)
                )
                common_idx = torch.randperm(d, generator=gen)[:k] if method == "randk" and k < d else None
                signals: list[torch.Tensor] = []
                masks: list[torch.Tensor] = []
                for cid, raw in enumerate(updates):
                    previous_memory = memories[cid]
                    raw_norm = float(torch.linalg.vector_norm(raw).item())
                    memory_norm = float(torch.linalg.vector_norm(previous_memory).item())
                    if raw_norm > 1e-20 and memory_norm > 1e-20:
                        ef_cross_correlations.append(
                            float(torch.dot(raw, previous_memory).item() / (raw_norm * memory_norm))
                        )
                    value_before_l2_clip = raw + previous_memory
                    value = value_before_l2_clip
                    if cfg.privacy_scope == "per_round_client_l2":
                        value = l2_clip(value, cfg.update_l2_clip)
                    l2_clipping_bias = value_before_l2_clip - value
                    sparse, mask, _, _ = compress_update(value, method, ratio, gen, common_idx)
                    memories[cid] = value - sparse
                    clipped = elementwise_clip(sparse, cfg.element_clip)
                    clipping_bias = l2_clipping_bias + sparse - clipped
                    retained_numerator += float(sparse.pow(2).sum().item())
                    retained_denominator += float(value.pow(2).sum().item())
                    retained_samples.append(float(sparse.pow(2).sum().item() / max(value.pow(2).sum().item(), 1e-20)))
                    ef_memory_ratios.append(
                        float(memories[cid].pow(2).sum().item() / max(raw.pow(2).sum().item(), 1e-20))
                    )
                    beta_sq_samples.append(float(clipping_bias.pow(2).sum().item() / (lr * lr * tau * tau)))
                    signals.append(clipped)
                    masks.append(mask)

                _, metrics = channel.aggregate(signals, masks, b_star, round_idx)
                clip_energies.append(metrics["clip_energy"])
                normalized_clip_energies.append(metrics["normalized_clip_energy"])
                papr_p99.append(metrics["papr_p99_db"])
                effective_noise.append(metrics["effective_noise_std"])

            omega_hat = retained_numerator / max(retained_denominator, 1e-20)
            omega_q10 = float(np.quantile(retained_samples, 0.10))
            # Rand-k has the exact contraction p in expectation. For Top-k, the
            # q10 retained-energy statistic is an explicitly empirical design
            # surrogate; the theorem-only comparison below still uses p.
            omega_design = p_actual if method == "randk" else omega_q10
            beta_sq_hat = float(np.mean(beta_sq_samples))
            mean_clip_energy = float(np.mean(clip_energies))
            terms = bound_terms(cfg, constants, d, k, omega_design, beta_sq_hat, mean_clip_energy, b_star)
            strict_terms = bound_terms(cfg, constants, d, k, p_actual, beta_sq_hat, mean_clip_energy, b_star)
            rows.append({
                "method": method,
                "ratio": ratio,
                "ratio_actual": p_actual,
                "k": k,
                "d": d,
                "snr_max_db": cfg.snr_max_db,
                "epsilon_total": cfg.epsilon,
                "privacy_scope": cfg.privacy_scope,
                "update_l2_clip": cfg.update_l2_clip,
                "delta": cfg.delta,
                "p_max_derived": cfg.p_max,
                "sigma0_normalized": cfg.sigma0,
                "tx_coordinate_clip": cfg.element_clip,
                "adc_backoff_db": cfg.adc_backoff_db,
                "adc_gamma_derived": cfg.adc_backoff_gamma,
                "b_power": b_power,
                "b_privacy": b_privacy,
                "b_star": b_star,
                "active_constraint": regime,
                "omega_empirical_energy_weighted": omega_hat,
                "omega_empirical_q10": omega_q10,
                "omega_theorem_contraction": p_actual,
                "omega_design_source": "strict_randk_p" if method == "randk" else "empirical_topk_q10",
                "ef_cross_correlation_mean": float(np.mean(ef_cross_correlations)) if ef_cross_correlations else 0.0,
                "ef_cross_correlation_q90": float(np.quantile(ef_cross_correlations, 0.90)) if ef_cross_correlations else 0.0,
                "ef_memory_to_raw_energy_mean": float(np.mean(ef_memory_ratios)),
                "beta_sq_hat": beta_sq_hat,
                "mean_clip_energy": mean_clip_energy,
                "mean_normalized_clip_energy": float(np.mean(normalized_clip_energies)),
                "mean_papr_p99_db": float(np.mean(papr_p99)),
                "effective_noise_std": float(np.mean(effective_noise)),
                "strict_Gamma": strict_terms["Gamma_hat"],
                "strict_bound_learning": strict_terms["bound_learning"],
                "strict_bound_channel": strict_terms["bound_channel"],
                "strict_bound_adc": strict_terms["bound_adc"],
                "strict_bound_total": strict_terms["bound_total"],
                "strict_eta_condition_satisfied": strict_terms["eta_condition_satisfied"],
                **terms,
            })
            print(
                f"[grid] {method} {ratio_idx + 1:03d}/{len(ratios)} "
                f"p={ratio:.2f} bound={terms['bound_total']:.4e} regime={regime}",
                flush=True,
            )

    # Full is evaluated once and is not drawn as a fictitious curve over p.
    method = "full"
    ratio = 1.0
    k = d
    b_power, b_privacy, b_star, regime = experiment1_power_privacy_limits(cfg, k, cfg.rounds)
    memories = [torch.zeros(d) for _ in range(cfg.num_clients)]
    beta_sq_samples: list[float] = []
    clip_energies: list[float] = []
    normalized_clip_energies: list[float] = []
    papr_p99: list[float] = []
    effective_noise: list[float] = []
    for round_idx, updates in enumerate(raw_updates):
        signals, masks = [], []
        for cid, raw in enumerate(updates):
            raw_before_l2_clip = raw
            if cfg.privacy_scope == "per_round_client_l2":
                raw = l2_clip(raw, cfg.update_l2_clip)
            clipped = elementwise_clip(raw, cfg.element_clip)
            beta_sq_samples.append(
                float((raw_before_l2_clip - clipped).pow(2).sum().item() / (lr * lr * tau * tau))
            )
            signals.append(clipped)
            masks.append(torch.ones(d, dtype=torch.bool))
        _, metrics = channel.aggregate(signals, masks, b_star, round_idx)
        clip_energies.append(metrics["clip_energy"])
        normalized_clip_energies.append(metrics["normalized_clip_energy"])
        papr_p99.append(metrics["papr_p99_db"])
        effective_noise.append(metrics["effective_noise_std"])
    beta_sq_hat = float(np.mean(beta_sq_samples))
    mean_clip_energy = float(np.mean(clip_energies))
    terms = bound_terms(cfg, constants, d, k, 1.0, beta_sq_hat, mean_clip_energy, b_star)
    rows.append({
        "method": method,
        "ratio": ratio,
        "ratio_actual": 1.0,
        "k": k,
        "d": d,
        "snr_max_db": cfg.snr_max_db,
        "epsilon_total": cfg.epsilon,
        "privacy_scope": cfg.privacy_scope,
        "update_l2_clip": cfg.update_l2_clip,
        "delta": cfg.delta,
        "p_max_derived": cfg.p_max,
        "sigma0_normalized": cfg.sigma0,
        "tx_coordinate_clip": cfg.element_clip,
        "adc_backoff_db": cfg.adc_backoff_db,
        "adc_gamma_derived": cfg.adc_backoff_gamma,
        "b_power": b_power,
        "b_privacy": b_privacy,
        "b_star": b_star,
        "active_constraint": regime,
        "omega_empirical_energy_weighted": 1.0,
        "omega_empirical_q10": 1.0,
        "omega_theorem_contraction": 1.0,
        "omega_design_source": "full_exact",
        "ef_cross_correlation_mean": 0.0,
        "ef_cross_correlation_q90": 0.0,
        "ef_memory_to_raw_energy_mean": 0.0,
        "beta_sq_hat": beta_sq_hat,
        "mean_clip_energy": mean_clip_energy,
        "mean_normalized_clip_energy": float(np.mean(normalized_clip_energies)),
        "mean_papr_p99_db": float(np.mean(papr_p99)),
        "effective_noise_std": float(np.mean(effective_noise)),
        "strict_Gamma": terms["Gamma_hat"],
        "strict_bound_learning": terms["bound_learning"],
        "strict_bound_channel": terms["bound_channel"],
        "strict_bound_adc": terms["bound_adc"],
        "strict_bound_total": terms["bound_total"],
        "strict_eta_condition_satisfied": terms["eta_condition_satisfied"],
        **terms,
    })
    print(f"[grid] full p=1.00 bound={terms['bound_total']:.4e} regime={regime}", flush=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_constant_sensitivity(
    path: Path,
    cfg: Config,
    constants: dict[str, float],
    rows: list[dict[str, Any]],
) -> None:
    """Re-evaluate Eq. (79) under pre-registered constant uncertainty."""

    variants: list[tuple[str, str, float]] = [("baseline", "baseline", 1.0)]
    variants.extend((f"L_x{factor:g}", "L_hat", constants["L_hat"] * factor) for factor in (2.0, 4.0))
    variants.extend((f"G_{value:g}", "G_hat", value) for value in (1.5, 2.0))
    for statistic in ("median", "q90"):
        variants.append((f"kappa_{statistic}", "kappa_sq_hat", constants[f"kappa_sq_{statistic}"]))
        variants.append((f"zeta_{statistic}", "zeta_sq_hat", constants[f"zeta_sq_{statistic}"]))

    output: list[dict[str, Any]] = []
    for variant_name, key, value in variants:
        variant_constants = dict(constants)
        if key != "baseline":
            variant_constants[key] = value
        for row in rows:
            terms = bound_terms(
                cfg,
                variant_constants,
                int(row["d"]),
                int(row["k"]),
                float(row["omega_hat"]),
                float(row["beta_sq_hat"]),
                float(row["mean_clip_energy"]),
                float(row["b_star"]),
            )
            output.append({
                "variant": variant_name,
                "changed_constant": key,
                "changed_value": value,
                "method": row["method"],
                "ratio": row["ratio"],
                "L": variant_constants["L_hat"],
                "G": variant_constants["G_hat"],
                "kappa_sq": variant_constants["kappa_sq_hat"],
                "zeta_sq": variant_constants["zeta_sq_hat"],
                "Gamma": terms["Gamma_hat"],
                "bound_learning": terms["bound_learning"],
                "bound_channel_base": terms["bound_channel_base"],
                "bound_channel": terms["bound_channel"],
                "bound_adc": terms["bound_adc"],
                "bound_total": terms["bound_total"],
                "eta_condition_satisfied": terms["eta_condition_satisfied"],
            })
    write_csv(path, output)


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
    for ax, method in zip(axes, ("topk", "randk")):
        sub = [row for row in rows if row["method"] == method]
        x = [row["ratio"] for row in sub]
        ax.semilogy(x, [row["bound_learning"] for row in sub], label="Learning", linewidth=1.8)
        ax.semilogy(x, [row["bound_channel"] for row in sub], label="Channel", linewidth=1.8)
        ax.semilogy(x, [max(row["bound_adc"], 1e-30) for row in sub], label="ADC", linewidth=1.8)
        ax.semilogy(x, [row["bound_total"] for row in sub], label="Total", linewidth=2.5)
        best = min(sub, key=lambda row: row["bound_total"])
        ax.axvline(best["ratio"], color="black", linestyle="--", linewidth=1.0)
        ax.set_title(method.upper())
        if method == "topk":
            ax.semilogy(x, [row["strict_bound_total"] for row in sub], label="Strict contraction", linestyle=":")
        ax.set_xlabel("Sparsity retention k/d")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Direct plug-in bound term (log scale)")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def write_summary(path: Path, cfg: Config, constants: dict[str, float], rows: list[dict[str, Any]]) -> None:
    best_unrestricted = {
        method: min((row for row in rows if row["method"] == method), key=lambda row: row["bound_total"])
        for method in ("topk", "randk", "full")
    }
    best_valid = {}
    for method in ("topk", "randk", "full"):
        valid = [row for row in rows if row["method"] == method and row["strict_eta_condition_satisfied"]]
        best_valid[method] = min(valid, key=lambda row: row["strict_bound_total"]) if valid else None
    all_valid = all(bool(row["strict_eta_condition_satisfied"]) for row in rows)
    best_strict = {
        method: min((row for row in rows if row["method"] == method), key=lambda row: row["strict_bound_total"])
        for method in ("topk", "randk", "full")
    }
    payload = {
        "interpretation": "Eq. (79) without fitted weights; calibrated design and strict contraction diagnostics are both reported",
        "theorem_condition_satisfied_for_all_points": all_valid,
        "problem_constant_estimates": constants,
        "best_unrestricted": best_unrestricted,
        "best_strict_contraction": best_strict,
        "best_theorem_valid": best_valid,
        "config": config_payload(cfg),
    }
    path.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 实验一：式（79）无权重稀疏保留率扫描\n\n")
        handle.write("本次扫描不含经验权重，不做方法内归一化。Top-k/Rand-k 在固定完整 OFDM 栅格上稀疏发送，不宣称节省子载波。$P_{\\max}$、$b^*(k)$ 和 ADC 阈值均由公开参数派生。\n\n")
        handle.write("Top-k 候选使用固定校准轨迹保留能量的 10% 分位数；Rand-k 使用严格的 $\\bar\\omega=p$。另行保存两种方法都取通用严格收缩 $p$ 时的诊断结果，经验代理不表述为定理证书。\n\n")
        handle.write(f"- SNR 定义与基准：$P_{{\\max}}/(d\\sigma_0^2)={cfg.snr_max_db:.1f}$ dB\n")
        if cfg.privacy_scope == "per_round_client_l2":
            handle.write(
                f"- 单轮客户端级隐私：每轮 $(\\epsilon,\\delta)=({cfg.epsilon:g},{cfg.delta:g})$；"
                f"不声明 {cfg.rounds} 轮组合后的总隐私预算\n"
            )
            handle.write(f"- 公共更新范数阈值：$C_{{\\rm upd}}={cfg.update_l2_clip:.6g}$（add/remove-client 灵敏度）\n")
        else:
            handle.write(f"- 全程客户端级隐私：$(\\epsilon_{{\\rm total}},\\delta)=({cfg.epsilon:g},{cfg.delta:g})$，共 {cfg.rounds} 轮\n")
        handle.write(f"- 实际发送裁剪阈值：$c_{{\\rm tx}}={cfg.element_clip:g}$\n")
        handle.write(f"- ADC backoff：{cfg.adc_backoff_db:.1f} dB\n")
        handle.write(f"- 全部候选点满足定理步长条件：**{'是' if all_valid else '否'}**\n\n")
        handle.write("若步长条件为否，本结果只能作为可复现的 bound-structure 诊断，不能宣称为定理认证的最优点。\n\n")
        handle.write("## 估计常数\n\n")
        handle.write("| 常数 | 数值 | 固定估计规则 |\n|---|---:|---|\n")
        rules = {
            "L_hat": "固定校准子集上相邻模型点确定性梯度的最大割线斜率",
            "G_hat": "梯度方差恒等分解固定为 1",
            "kappa_sq_hat": "各轮客户端梯度相对全局均值的最大均方偏差",
            "zeta_sq_hat": "同一模型点两次独立 mini-batch 梯度差的一半平方范数最大值",
            "delta_f_hat": "首轮交叉熵均值，使用 $f^*=0$",
        }
        for key, rule in rules.items():
            handle.write(f"| `{key}` | {constants[key]:.6e} | {rule} |\n")
        handle.write("\n## 搜索结果\n\n")
        handle.write("| 方法 | 候选口径 | $k^*/d$ | 总项 | 学习项 | 基础信道项 | $\\Gamma$ 后信道项 | ADC 项 | 步长条件 |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for method in ("topk", "randk", "full"):
            row = best_unrestricted[method]
            handle.write(
                f"| {method.upper()} | 全网格诊断 | {row['ratio']:.2f} | {row['bound_total']:.4e} | "
                f"{row['bound_learning']:.4e} | {row['bound_channel_base']:.4e} | "
                f"{row['bound_channel']:.4e} | {row['bound_adc']:.4e} | "
                f"{'满足' if row['eta_condition_satisfied'] else '不满足'} |\n"
            )
            strict_row = best_strict[method]
            handle.write(
                f"| {method.upper()} | 通用严格收缩 | {strict_row['ratio']:.2f} | "
                f"{strict_row['strict_bound_total']:.4e} | {strict_row['strict_bound_learning']:.4e} | "
                f"{strict_row['bound_channel_base']:.4e} | {strict_row['strict_bound_channel']:.4e} | "
                f"{strict_row['strict_bound_adc']:.4e} | "
                f"{'满足' if strict_row['strict_eta_condition_satisfied'] else '不满足'} |\n"
            )
            valid_row = best_valid[method]
            if valid_row is not None:
                handle.write(
                    f"| {method.upper()} | 仅定理有效点 | {valid_row['ratio']:.2f} | "
                    f"{valid_row['strict_bound_total']:.4e} | {valid_row['strict_bound_learning']:.4e} | "
                    f"{valid_row['bound_channel_base']:.4e} | {valid_row['strict_bound_channel']:.4e} | "
                    f"{valid_row['strict_bound_adc']:.4e} | 满足 |\n"
                )
            else:
                handle.write(f"| {method.upper()} | 仅定理有效点 | -- | -- | -- | -- | -- | -- | 无有效点 |\n")


def config_payload(cfg: Config) -> dict[str, Any]:
    payload = asdict(cfg)
    payload.update({
        "snr_max_db": cfg.snr_max_db,
        "adc_backoff_db": cfg.adc_backoff_db,
        "snr_definition": "P_max / (d * sigma0^2)",
        "epsilon_scope": cfg.privacy_scope,
        "update_l2_clip": cfg.update_l2_clip,
    })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="exp/exp1-ksearch/bound_direct_snr15_eps30")
    parser.add_argument("--dataset", choices=("femnist", "mnist"), default="femnist")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--calib-rounds", type=int, default=4)
    parser.add_argument("--constant-estimation-samples", type=int, default=1024)
    parser.add_argument("--ratio-step", type=float, default=0.01)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--calibrate-learning-rate", action="store_true")
    parser.add_argument("--lr-calibration-grid", default="0.001,0.005,0.01,0.02,0.05")
    parser.add_argument("--lr-calibration-rounds", type=int, default=12)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--snr-max-db", type=float, default=15.0)
    parser.add_argument("--epsilon-total", type=float, default=30.0)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument(
        "--privacy-scope",
        choices=("total_replace_one_coordinate", "per_round_client_l2"),
        default="total_replace_one_coordinate",
    )
    parser.add_argument(
        "--tx-coordinate-clip",
        default="auto-q95",
        help="Positive number or one of auto-q90/auto-q95/auto-q99",
    )
    parser.add_argument("--calibration-cache", default="")
    parser.add_argument(
        "--update-l2-clip",
        default="auto-q95",
        help="Positive number or auto-q90/auto-q95/auto-q99; used by per_round_client_l2",
    )
    parser.add_argument("--h-th", type=float, default=0.1)
    parser.add_argument("--ofdm-subcarriers", type=int, default=1024)
    parser.add_argument("--oversampling", type=int, default=4)
    parser.add_argument("--adc-backoff-db", type=float, default=6.0)
    args = parser.parse_args()

    cfg = Config(
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        datasets=(args.dataset,),
        methods=("topk", "randk", "full"),
        rounds=args.rounds,
        num_clients=args.num_clients,
        local_steps_mnist=args.local_steps,
        local_steps_femnist=args.local_steps,
        batch_size=args.batch_size,
        lr_mnist=args.learning_rate,
        lr_femnist=args.learning_rate,
        dirichlet_alpha=args.dirichlet_alpha,
        epsilon=args.epsilon_total,
        delta=args.delta,
        sigma0=1.0,
        h_th=args.h_th,
        p_max=1.0,
        eta_tau_C=1.0,
        element_clip=1.0,
        ofdm_subcarriers=args.ofdm_subcarriers,
        oversampling=args.oversampling,
        adc_backoff_gamma=10.0 ** (args.adc_backoff_db / 20.0),
    )
    cfg.snr_max_db = args.snr_max_db
    cfg.adc_backoff_db = args.adc_backoff_db
    cfg.privacy_scope = args.privacy_scope
    cfg.update_l2_clip = math.inf
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        cfg.device = "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    if args.calibrate_learning_rate:
        candidates = [float(value) for value in args.lr_calibration_grid.split(",") if value.strip()]
        selected_lr, lr_rows = calibrate_clean_full_learning_rate(
            cfg, args.dataset, candidates, args.lr_calibration_rounds
        )
        cfg.lr_mnist = selected_lr
        cfg.lr_femnist = selected_lr
        write_csv(out_dir / "learning_rate_calibration.csv", lr_rows)
        print(f"[lr-calibration] selected={selected_lr:g}", flush=True)

    cache_path = Path(args.calibration_cache) if args.calibration_cache else None
    if cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        metadata = cached["metadata"]
        expected = {
            "dataset": args.dataset,
            "seed": cfg.seed,
            "num_clients": cfg.num_clients,
            "local_steps": cfg.local_steps_femnist if args.dataset == "femnist" else cfg.local_steps_mnist,
            "learning_rate": cfg.lr_femnist if args.dataset == "femnist" else cfg.lr_mnist,
        }
        if metadata != expected:
            raise ValueError(f"Calibration cache metadata mismatch: cached={metadata}, expected={expected}")
        raw_updates, constants, d = cached["raw_updates"], cached["constants"], int(cached["d"])
        print(f"[cache] loaded {cache_path}", flush=True)
    else:
        raw_updates, constants, d = estimate_problem_constants(
            cfg, args.dataset, args.calib_rounds, args.constant_estimation_samples
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "dataset": args.dataset,
                "seed": cfg.seed,
                "num_clients": cfg.num_clients,
                "local_steps": cfg.local_steps_femnist if args.dataset == "femnist" else cfg.local_steps_mnist,
                "learning_rate": cfg.lr_femnist if args.dataset == "femnist" else cfg.lr_mnist,
            }
            torch.save({"metadata": metadata, "raw_updates": raw_updates, "constants": constants, "d": d}, cache_path)
            print(f"[cache] saved {cache_path}", flush=True)

    clip_quantiles = public_clip_quantiles(raw_updates)
    l2_clip_quantiles = public_l2_clip_quantiles(raw_updates)
    clip_spec = str(args.tx_coordinate_clip).lower()
    if clip_spec.startswith("auto-"):
        clip_key = clip_spec.removeprefix("auto-")
        if clip_key not in clip_quantiles:
            raise ValueError(f"Unknown automatic clipping rule: {args.tx_coordinate_clip}")
        tx_clip = clip_quantiles[clip_key]
    else:
        tx_clip = float(clip_spec)
    if not math.isfinite(tx_clip) or tx_clip <= 0:
        raise ValueError("tx-coordinate-clip must be positive")
    cfg.element_clip = tx_clip
    cfg.eta_tau_C = tx_clip
    l2_clip_spec = str(args.update_l2_clip).lower()
    if l2_clip_spec.startswith("auto-"):
        l2_clip_key = l2_clip_spec.removeprefix("auto-")
        if l2_clip_key not in l2_clip_quantiles:
            raise ValueError(f"Unknown automatic L2 clipping rule: {args.update_l2_clip}")
        update_l2_clip = l2_clip_quantiles[l2_clip_key]
    else:
        update_l2_clip = float(l2_clip_spec)
    if not math.isfinite(update_l2_clip) or update_l2_clip <= 0:
        raise ValueError("update-l2-clip must be positive")
    cfg.update_l2_clip = update_l2_clip
    constants["tx_clip_q90"] = clip_quantiles["q90"]
    constants["tx_clip_q95"] = clip_quantiles["q95"]
    constants["tx_clip_q99"] = clip_quantiles["q99"]
    constants["tx_clip_sample_count"] = clip_quantiles["sample_count"]
    constants["tx_clip_selected"] = tx_clip
    constants["tx_clip_rule"] = clip_spec
    constants["update_l2_clip_q90"] = l2_clip_quantiles["q90"]
    constants["update_l2_clip_q95"] = l2_clip_quantiles["q95"]
    constants["update_l2_clip_q99"] = l2_clip_quantiles["q99"]
    constants["update_l2_clip_sample_count"] = l2_clip_quantiles["sample_count"]
    constants["update_l2_clip_selected"] = update_l2_clip
    constants["update_l2_clip_rule"] = l2_clip_spec
    print(f"[clip-calibration] rule={clip_spec} selected={tx_clip:.6e}", flush=True)
    print(f"[l2-clip-calibration] rule={l2_clip_spec} selected={update_l2_clip:.6e}", flush=True)

    cfg.p_max = d * cfg.sigma0 * cfg.sigma0 * 10.0 ** (cfg.snr_max_db / 10.0)
    ratios = ratio_grid(args.ratio_step)
    rows = evaluate_grid(cfg, raw_updates, constants, d, ratios)

    (out_dir / "config.json").write_text(json.dumps(config_payload(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "bound_terms.csv", rows)
    write_constant_sensitivity(out_dir / "constant_sensitivity.csv", cfg, constants, rows)
    write_plot(out_dir / "bound_decomposition.png", rows)
    write_summary(out_dir / "summary.md", cfg, constants, rows)
    print(json.dumps({"output_dir": str(out_dir), "constants": constants}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
