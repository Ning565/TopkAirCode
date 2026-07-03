#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end ADC-aware intrinsic-private OFDM-AirComp-FL experiment.

This script is intentionally self-contained. It models:
1. FL local training with Top-k/Rand-k/Full update transmission.
2. Element-wise clipping for sensitivity and power control.
3. Closed-form DP/power feasible AirComp scaling b*(k).
4. Frequency-domain AirComp aggregation.
5. Oversampled OFDM time-domain waveform, AGC/backoff ADC clipping,
   FFT recovery, and clipped aggregation update.

The first goal is not exhaustive parameter tuning. It is to validate that the
complete system model runs and that Top-k's accuracy-equivalent lower-k regime
is visible against Rand-k and Full.
"""

import argparse
import csv
import gzip
import json
import math
import pickle
import random
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset


@dataclass
class Config:
    seed: int = 2026
    device: str = "cuda:3"
    output_dir: str = "exp/full_system/results"
    mnist_root: str = "data/MNIST/raw"
    femnist_path: str = "data/femnist/femnist_train.pkl"
    femnist_test_path: str = "data/femnist/femnist_test.pkl"

    datasets: Tuple[str, ...] = ("mnist", "femnist")
    methods: Tuple[str, ...] = ("topk", "randk", "full")
    rounds: int = 200
    num_clients: int = 20
    local_steps_mnist: int = 5
    local_steps_femnist: int = 5
    batch_size: int = 64
    eval_batch_size: int = 512
    eval_every: int = 1
    lr_mnist: float = 0.05
    lr_femnist: float = 0.05
    dirichlet_alpha: float = 0.3

    # Accuracy-equivalent k choices from previous sweep.
    topk_ratio_mnist: float = 0.05
    topk_ratio_femnist: float = 0.10
    randk_ratio: float = 0.90
    full_ratio: float = 1.0
    topk_error_feedback: bool = True
    error_feedback_methods: Tuple[str, ...] = ("topk", "randk")
    element_clip: float = 0.05

    # OFDM-AirComp-ADC.
    ofdm_subcarriers: int = 2000
    oversampling: int = 4
    adc_backoff_gamma: float = 3.0

    # DP / power model. Epsilon is deliberately permissive for the first
    # convergence validation; later sweeps can tighten it.
    epsilon: float = 1e10
    delta: float = 1e-3
    sigma0: float = 0.01
    h_th: float = 0.1
    p_max: float = 1e6
    eta_tau_C: float = 0.05


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_idx_images(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image magic number {magic}")
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    return data.reshape(n, 1, rows, cols).astype(np.float32)


def read_idx_labels(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label magic number {magic}")
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    return data.astype(np.int64)


def idx_path(root: Path, stem: str) -> Path:
    raw = root / stem
    gz = root / f"{stem}.gz"
    if raw.exists():
        return raw
    if gz.exists():
        return gz
    raise FileNotFoundError(stem)


def make_dirichlet(labels: np.ndarray, num_clients: int, alpha: float, seed: int) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    client_lists = {cid: [] for cid in range(num_clients)}
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


def load_mnist(cfg: Config):
    root = Path(cfg.mnist_root)
    x_train = read_idx_images(idx_path(root, "train-images-idx3-ubyte"))
    y_train = read_idx_labels(idx_path(root, "train-labels-idx1-ubyte"))
    x_test = read_idx_images(idx_path(root, "t10k-images-idx3-ubyte"))
    y_test = read_idx_labels(idx_path(root, "t10k-labels-idx1-ubyte"))
    x_train = (x_train / 255.0 - 0.1307) / 0.3081
    x_test = (x_test / 255.0 - 0.1307) / 0.3081
    train = TensorDataset(torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train).long())
    test = TensorDataset(torch.from_numpy(x_test.astype(np.float32)), torch.from_numpy(y_test).long())
    clients = make_dirichlet(y_train, cfg.num_clients, cfg.dirichlet_alpha, cfg.seed)
    return train, test, clients, 10


def load_femnist_split(path: str, users: List[str]):
    with open(path, "rb") as f:
        data = pickle.load(f)
    xs, ys = [], []
    for user in users:
        x = np.asarray(data["user_data"][user]["x"], dtype=np.float32).reshape(-1, 1, 28, 28)
        y = np.asarray(data["user_data"][user]["y"], dtype=np.int64)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def load_femnist(cfg: Config):
    with open(cfg.femnist_path, "rb") as f:
        data = pickle.load(f)
    users = list(data["users"])
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(users)
    users = users[: cfg.num_clients]
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
    x_test, y_test = load_femnist_split(cfg.femnist_test_path, users)
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


def flatten_delta(model: nn.Module, base: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([(p.detach().cpu() - base[name]).reshape(-1) for name, p in model.named_parameters()]).float()


def apply_update(model: nn.Module, update: torch.Tensor) -> None:
    with torch.no_grad():
        off = 0
        for p in model.parameters():
            n = p.numel()
            p.add_(update[off : off + n].view_as(p).to(p.device))
            off += n


def elementwise_clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.clamp(x, min=-threshold, max=threshold)


def compress_update(v: torch.Tensor, method: str, ratio: float, gen: torch.Generator, common_idx=None):
    d = v.numel()
    k = d if method == "full" else max(1, int(d * ratio))
    if method == "full" or k >= d:
        mask = torch.ones(d, dtype=torch.bool)
        return v.clone(), mask, 1.0, k
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
    retained = float(out.pow(2).sum().item() / (v.pow(2).sum().item() + 1e-12))
    return out, mask, retained, k


def power_privacy_limits(cfg: Config, k: int, total_rounds: int) -> Tuple[float, float, float, str]:
    sqrt_k = math.sqrt(max(1, k))
    b_power = cfg.h_th * math.sqrt(cfg.p_max) / (cfg.eta_tau_C * sqrt_k)
    log_delta = math.log(1.0 / cfg.delta)
    privacy_margin = math.sqrt(cfg.epsilon + log_delta) - math.sqrt(log_delta)
    b_priv = cfg.sigma0 * privacy_margin / (2.0 * cfg.eta_tau_C * sqrt_k * math.sqrt(total_rounds))
    b_star = min(b_power, b_priv)
    regime = "power" if b_power <= b_priv else "privacy"
    return b_power, b_priv, b_star, regime


class OFDMAirCompADC:
    def __init__(self, cfg: Config, d: int):
        self.cfg = cfg
        self.d = d
        self.M = cfg.ofdm_subcarriers
        self.S = math.ceil(d / self.M)
        self.padded = self.S * self.M
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 314159)
        self.perm = torch.randperm(self.padded, generator=gen)

    def pack(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(0)
        out = torch.zeros(c, self.padded, dtype=x.dtype)
        out[:, self.perm[: self.d]] = x
        return out.view(c, self.S, self.M)

    def unpack(self, x_resource: torch.Tensor) -> torch.Tensor:
        flat = x_resource.reshape(-1)
        return flat[self.perm[: self.d]]

    def aggregate(self, signals: List[torch.Tensor], masks: List[torch.Tensor], b_star: float, round_idx: int):
        cfg = self.cfg
        c = len(signals)
        sig = torch.stack(signals, dim=0).float()
        mask = torch.stack(masks, dim=0).float()
        sig_sym = self.pack(sig)
        mask_sym = self.pack(mask)
        freq_sum = b_star * sig_sym.sum(dim=0)

        # Intrinsic channel noise in frequency domain.
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 104729 * round_idx)
        noise_std = math.sqrt(cfg.sigma0 ** 2 / 2.0)
        nr = torch.randn(self.S, self.M, generator=gen) * noise_std
        ni = torch.randn(self.S, self.M, generator=gen) * noise_std
        y_freq = torch.complex(freq_sum + nr, ni)

        n_fft = self.M * cfg.oversampling
        y_os = torch.zeros(self.S, n_fft, dtype=torch.complex64)
        y_os[:, : self.M] = y_freq.to(torch.complex64)
        y_time = torch.fft.ifft(y_os, dim=-1, norm="ortho")

        power = y_time.abs().pow(2)
        avg_power = power.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(avg_power + 1e-12)
        y_norm = y_time / rms
        mag = y_norm.abs()
        scale = torch.clamp(cfg.adc_backoff_gamma / (mag + 1e-12), max=1.0)
        y_clip = y_norm * scale * rms
        residual = y_clip - y_time

        y_back = torch.fft.fft(y_clip, dim=-1, norm="ortho")[:, : self.M]
        recovered_resource = y_back.real / (b_star * c + 1e-12)
        recovered = self.unpack(recovered_resource).float()

        active = self.unpack(mask_sym.sum(dim=0)).float()
        recovered = recovered * (active > 0).float()

        papr = power.max(dim=-1).values / (power.mean(dim=-1) + 1e-12)
        papr_db = 10 * torch.log10(papr + 1e-12)
        clip_energy = residual.abs().pow(2).sum().item()
        signal_energy = y_time.abs().pow(2).sum().item()
        clip_ratio = float((mag > cfg.adc_backoff_gamma).float().mean().item())
        u = mask_sym.sum(dim=0).numpy()
        active_u = u[u > 0]
        metrics = {
            "papr_mean_db": float(papr_db.mean().item()),
            "papr_p99_db": float(torch.quantile(papr_db, 0.99).item()),
            "papr_max_db": float(papr_db.max().item()),
            "clip_sample_ratio": clip_ratio,
            "normalized_clip_energy": float(clip_energy / (signal_energy + 1e-12)),
            "u_active_mean": float(active_u.mean()) if active_u.size else 0.0,
            "active_resource_ratio": float(np.mean(u > 0)),
        }
        return recovered, metrics


class Experiment:
    def __init__(self, cfg: Config, dataset_name: str, train, test, clients, num_classes):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.train = train
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

    def client_delta(self, model: nn.Module, cid: int, lr: float, local_steps: int):
        cfg = self.cfg
        local = make_model(self.dataset_name, self.num_classes).to(cfg.device)
        local.load_state_dict(model.state_dict())
        local.train()
        base = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        opt = torch.optim.SGD(local.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        losses = []
        for _ in range(local_steps):
            x, y = self.next_batch(cid)
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(local(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        delta = flatten_delta(local, base)
        del local, opt, loss_fn
        return delta, float(np.mean(losses))

    def evaluate(self, model: nn.Module):
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

    def ratio_for(self, method: str) -> float:
        override_key = f"{self.dataset_name}:{method}"
        if hasattr(self.cfg, "ratio_overrides") and override_key in self.cfg.ratio_overrides:
            return float(self.cfg.ratio_overrides[override_key])
        if method == "topk":
            return self.cfg.topk_ratio_femnist if self.dataset_name == "femnist" else self.cfg.topk_ratio_mnist
        if method == "randk":
            return self.cfg.randk_ratio
        return self.cfg.full_ratio

    def use_error_feedback(self, method: str) -> bool:
        if method == "topk" and self.cfg.topk_error_feedback:
            return True
        return method in getattr(self.cfg, "error_feedback_methods", ())

    def run_method(self, method: str):
        cfg = self.cfg
        set_seed(cfg.seed)
        model = make_model(self.dataset_name, self.num_classes).to(cfg.device)
        d = sum(p.numel() for p in model.parameters())
        channel = OFDMAirCompADC(cfg, d)
        ratio = self.ratio_for(method)
        k = d if method == "full" else max(1, int(d * ratio))
        b_p, b_eps, b_star, regime = power_privacy_limits(cfg, k, cfg.rounds)
        memories = {cid: torch.zeros(d) for cid in self.clients}
        lr = cfg.lr_femnist if self.dataset_name == "femnist" else cfg.lr_mnist
        local_steps = cfg.local_steps_femnist if self.dataset_name == "femnist" else cfg.local_steps_mnist
        rows = []
        for r in range(cfg.rounds):
            signals, masks, losses, retained = [], [], [], []
            gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 10007 * r + int(ratio * 1_000_000))
            common_idx = torch.randperm(d, generator=gen)[:k] if method == "randk" and k < d else None
            for cid in range(cfg.num_clients):
                raw, loss = self.client_delta(model, cid, lr, local_steps)
                use_ef = self.use_error_feedback(method)
                inp = raw + memories[cid] if use_ef else raw
                sparse, mask, ret, _ = compress_update(inp, method, ratio, gen, common_idx)
                sparse = elementwise_clip(sparse, cfg.element_clip)
                if use_ef:
                    memories[cid] = inp - sparse
                signals.append(sparse)
                masks.append(mask)
                losses.append(loss)
                retained.append(ret)
            update, comm = channel.aggregate(signals, masks, b_star, r)
            apply_update(model, update)
            if (r + 1) % cfg.eval_every == 0 or r == 0 or r == cfg.rounds - 1:
                acc, test_loss = self.evaluate(model)
            else:
                acc = rows[-1]["test_accuracy"] if rows else float("nan")
                test_loss = rows[-1]["test_loss"] if rows else float("nan")
            row = {
                "dataset": self.dataset_name,
                "method": method,
                "round": r + 1,
                "ratio": ratio,
                "k": k,
                "d": d,
                "train_loss": float(np.mean(losses)),
                "test_accuracy": float(acc),
                "test_loss": float(test_loss),
                "retained_energy": float(np.mean(retained)),
                "b_power": b_p,
                "b_privacy": b_eps,
                "b_star": b_star,
                "regime": regime,
                **comm,
            }
            rows.append(row)
            if (r + 1) % max(1, cfg.eval_every) == 0 or r == 0:
                print(
                    f"[{self.dataset_name}/{method}] round {r+1}/{cfg.rounds} "
                    f"acc={acc:.2f} papr99={comm['papr_p99_db']:.2f} "
                    f"nce={comm['normalized_clip_energy']:.2e} b={b_star:.3e}",
                    flush=True,
                )
        return rows


def write_rows(path: Path, rows: List[Dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(rows: List[Dict], out_dir: Path):
    for dataset in sorted({r["dataset"] for r in rows}):
        plt.figure(figsize=(7.2, 4.6))
        for method in ["topk", "randk", "full"]:
            sub = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            if not sub:
                continue
            plt.plot([r["round"] for r in sub], [r["test_accuracy"] for r in sub], linewidth=2.2, label=method.upper())
        plt.xlabel("Communication rounds")
        plt.ylabel("Test accuracy (%)")
        plt.title(f"{dataset.upper()} full system: rounds vs accuracy")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{dataset}_rounds_vs_accuracy.png", dpi=220)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--datasets", default="mnist,femnist")
    parser.add_argument("--methods", default="topk,randk,full")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--ratio-overrides", default="", help="Comma list like mnist:topk=0.1,mnist:randk=0.8")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=Config.epsilon)
    parser.add_argument("--delta", type=float, default=Config.delta)
    parser.add_argument("--sigma0", type=float, default=Config.sigma0)
    parser.add_argument("--h-th", type=float, default=Config.h_th)
    parser.add_argument("--p-max", type=float, default=Config.p_max)
    parser.add_argument("--eta-tau-c", type=float, default=Config.eta_tau_C)
    parser.add_argument("--adc-backoff-gamma", type=float, default=Config.adc_backoff_gamma)
    parser.add_argument("--ofdm-subcarriers", type=int, default=Config.ofdm_subcarriers)
    parser.add_argument("--error-feedback-methods", default="topk,randk")
    args = parser.parse_args()
    cfg = Config(
        seed=args.seed,
        device=args.device,
        rounds=args.rounds,
        datasets=tuple(x.strip() for x in args.datasets.split(",") if x.strip()),
        methods=tuple(x.strip() for x in args.methods.split(",") if x.strip()),
        output_dir=args.output_dir or Config.output_dir,
        eval_every=args.eval_every,
        epsilon=args.epsilon,
        delta=args.delta,
        sigma0=args.sigma0,
        h_th=args.h_th,
        p_max=args.p_max,
        eta_tau_C=args.eta_tau_c,
        adc_backoff_gamma=args.adc_backoff_gamma,
        ofdm_subcarriers=args.ofdm_subcarriers,
        error_feedback_methods=tuple(x.strip() for x in args.error_feedback_methods.split(",") if x.strip()),
    )
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        cfg.device = "cpu"
    cfg.ratio_overrides = {}
    if args.ratio_overrides:
        for item in args.ratio_overrides.split(","):
            if not item.strip():
                continue
            key, val = item.split("=")
            cfg.ratio_overrides[key.strip()] = float(val)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    all_rows: List[Dict] = []
    for dataset_name in cfg.datasets:
        if dataset_name == "mnist":
            train, test, clients, num_classes = load_mnist(cfg)
        elif dataset_name == "femnist":
            train, test, clients, num_classes = load_femnist(cfg)
        else:
            raise ValueError(dataset_name)
        for method in cfg.methods:
            exp = Experiment(cfg, dataset_name, train, test, clients, num_classes)
            rows = exp.run_method(method)
            all_rows.extend(rows)
            write_rows(out_dir / "metrics_rounds.csv", all_rows)
            with open(out_dir / "metrics_rounds.json", "w", encoding="utf-8") as f:
                json.dump(all_rows, f, indent=2, ensure_ascii=False)
            plot_accuracy(all_rows, out_dir)

    final = []
    for dataset in cfg.datasets:
        for method in cfg.methods:
            sub = [r for r in all_rows if r["dataset"] == dataset and r["method"] == method]
            if not sub:
                continue
            last = sub[-1]
            final.append({
                "dataset": dataset,
                "method": method,
                "final_accuracy": last["test_accuracy"],
                "ratio": last["ratio"],
                "k": last["k"],
                "papr_p99_last": last["papr_p99_db"],
                "clip_energy_last": last["normalized_clip_energy"],
                "b_star": last["b_star"],
                "regime": last["regime"],
            })
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# 完整系统 ADC/DP/功率约束实验\n\n")
        f.write("## 实验设置\n\n")
        f.write(f"- 通信轮数：{cfg.rounds}\n- epsilon：{cfg.epsilon}，delta：{cfg.delta}\n")
        ratio_lines = []
        for dataset in cfg.datasets:
            for method in cfg.methods:
                sub = [r for r in final if r["dataset"] == dataset and r["method"] == method]
                if sub:
                    ratio_lines.append(f"{dataset}:{method}={sub[0]['ratio']:.2f}")
        f.write(f"- 实际压缩率：{', '.join(ratio_lines)}\n")
        f.write(f"- OFDM 子载波数 M={cfg.ofdm_subcarriers}，过采样倍数={cfg.oversampling}，ADC backoff gamma={cfg.adc_backoff_gamma}\n\n")
        f.write("## 最终结果\n\n")
        f.write("| 数据集 | 方法 | 压缩率 | 最终准确率 | PAPR P99 | NCE | b* | 约束状态 |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        for r in final:
            f.write(
                f"| {r['dataset']} | {r['method']} | {r['ratio']:.2f} | {r['final_accuracy']:.2f} | "
                f"{r['papr_p99_last']:.2f} | {r['clip_energy_last']:.2e} | {r['b_star']:.3e} | {r['regime']} |\n"
            )


if __name__ == "__main__":
    main()
