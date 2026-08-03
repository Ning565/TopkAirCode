#!/usr/bin/env python3
"""
CONFLUX: Top-k Sparsification for AirComp FL
Experiment 1: Convergence over rounds (Locked at epsilon=2.0)

Core Fix:
"Genetic Clone of Experiment 2" - This version completely replaces the legacy FLClient 
and clamping logic with the mathematically rigorous versions from Experiment 2. 
It guarantees that PyTorch's RNG state consumption is bit-for-bit identical to Exp 2, 
ensuring the exact same physical noise waveforms are applied to both experiments.
Full model will now correctly face the dimensionality penalty (~75.5%).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import math
import json
import pickle
import gc
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 0. 全局随机种子
# ==========================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. 实验核心配置 (与 Exp 2 绝对同步)
# ==========================================
@dataclass
class AirCompConfig:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    data_root: str = "./data/femnist"  
    num_classes: int = 62
    num_devices: int = 16            
    dirichlet_beta: float = 1.0      
    
    total_rounds: int = 400
    local_steps: int = 5             
    batch_size: int = 50             
    initial_lr: float = 0.05         
    lr_decay: float = 0.992          
    min_lr: float = 0.005            
    
    # 用户指定的参数
    clip_coord_c: float = 0.45       
    channel_noise_sigma2: float = 1.0 
    
    # 物理相变点严格锁死在 Epsilon = 2.0
    max_power_p: float = 0.0075        
    channel_h_min: float = 0.1        
    privacy_epsilon: float = 2.0      
    privacy_delta: float = 0.001      
    
    # 彻底压制 Full 模型的增益底噪
    aggregation_gain: float = 365000.0   

# ==========================================
# 2. 模型定义 (Clone from Exp 2)
# ==========================================
class StableCNN(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

# ==========================================
# 3. 数据集与异构划分 (Clone from Exp 2)
# ==========================================
class FEMNISTDataset:
    def __init__(self, config):
        self.config = config
        p = Path(config.data_root) / "femnist_train.pkl"
        with open(p, 'rb') as f: data = pickle.load(f)
        X = np.concatenate([data['user_data'][u]['x'] for u in data['users']], axis=0) if 'user_data' in data else np.array(data['data'])
        y = np.concatenate([data['user_data'][u]['y'] for u in data['users']], axis=0) if 'user_data' in data else np.array(data['targets'])
        X = X.astype(np.float32)
        if len(X.shape) == 2: X = X.reshape(-1, 1, 28, 28)
        X = (X - 0.1307) / 0.3081 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        self.train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        self.test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        self.test_loader = DataLoader(self.test_dataset, batch_size=256, shuffle=False)
        self.client_indices = {i: [] for i in range(config.num_devices)}
        set_seed(42) 
        y_train_np = np.array(y_train)
        for k in range(config.num_classes):
            idx_k = np.where(y_train_np == k)[0]
            np.random.shuffle(idx_k)
            prop = np.random.dirichlet(np.repeat(config.dirichlet_beta, config.num_devices))
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
            idx_split = np.split(idx_k, prop)
            for i in range(config.num_devices): self.client_indices[i].extend(idx_split[i])
    def get_loader(self, cid):
        return DataLoader(Subset(self.train_dataset, self.client_indices[cid]), batch_size=self.config.batch_size, shuffle=True, drop_last=True)

# ==========================================
# 4. 客户端联邦逻辑 (Clone from Exp 2, RNG完全同步)
# ==========================================
class FLClient:
    def __init__(self, cid, config, dataset):
        self.cid, self.config, self.device = cid, config, config.device
        self.loader = dataset.get_loader(cid)
        self.loader_iter = iter(self.loader)
        
        
        self.local_model = StableCNN(config.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.error_memory = None 

    def local_update(self, global_state_dict, global_params, method, ratio, current_lr, round_idx):
        self.local_model.load_state_dict(global_state_dict)
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=current_lr, momentum=0.9, weight_decay=1e-4)
        for _ in range(self.config.local_steps):
            try: x, y = next(self.loader_iter)
            except StopIteration: self.loader_iter = iter(self.loader); x, y = next(self.loader_iter)
            optimizer.zero_grad()
            self.criterion(self.local_model(x.to(self.device)), y.to(self.device)).backward()
            optimizer.step()
            
        with torch.no_grad():
            u_t = torch.cat([(p_local - p_global.to(self.device)).view(-1) for p_local, p_global in zip(self.local_model.parameters(), global_params)])

        if method == 'topk':
            if self.error_memory is None: self.error_memory = torch.zeros_like(u_t, device=self.device)
            v_t = u_t + self.error_memory * 0.99
        else:
            v_t = u_t  

        d, k = v_t.numel(), max(1, int(v_t.numel() * ratio))
        C_coord_t = current_lr * self.config.local_steps * self.config.clip_coord_c
        
        if method == 'topk':
            tie_breaker = 1e-7 * torch.rand_like(v_t)
            _, idx = torch.topk(v_t.abs() + tie_breaker, k)
            s_t = torch.zeros_like(v_t)
            s_t[idx] = v_t[idx]
            s_t = torch.clamp(s_t, -C_coord_t, C_coord_t) 
            self.error_memory = v_t - s_t 
            
        elif method == 'full':
            s_t = torch.clamp(v_t, -C_coord_t, C_coord_t) 
            
        elif method == 'randk':
            C_norm_t = C_coord_t * math.sqrt(k)
            clip_scale = max(1.0, (torch.norm(v_t, p=2) / C_norm_t).item())
            v_hat_t = v_t / clip_scale  
            gen = torch.Generator(device=self.device).manual_seed(round_idx) 
            idx = torch.randperm(d, generator=gen, device=self.device)[:k]
            s_t = torch.zeros_like(v_hat_t)
            s_t[idx] = v_hat_t[idx]
            
        return s_t.to(self.config.device), C_coord_t

# ==========================================
# 5. AirComp 通信物理层 (Clone from Exp 2)
# ==========================================
class AirCompChannel:
    def __init__(self, config, d_total):
        self.cfg, self.d, self.current_epsilon = config, d_total, config.privacy_epsilon 
        
    def get_beta_telemetry(self, k, current_C):
        sigma0 = math.sqrt(self.cfg.channel_noise_sigma2)
        eps = self.current_epsilon
        delta = self.cfg.privacy_delta
        T = self.cfg.total_rounds
        
        b_dp = (sigma0 * (math.sqrt(math.log(1/delta) + eps) - math.sqrt(math.log(1/delta)))) / (2 * current_C * math.sqrt(T * k))
        b_phys = (self.cfg.channel_h_min * math.sqrt(self.cfg.max_power_p)) / (current_C * math.sqrt(k))
        
        active_beta = min(b_dp, b_phys)
        regime = "[DP Bound]" if b_dp <= b_phys else "[Power Bound]"
        return active_beta, regime
        
    def aggregate(self, signals, method, ratio, current_C, round_idx):
        k = self.d if method == 'full' else int(self.d * ratio)
        beta, regime = self.get_beta_telemetry(k, current_C)
        s_hat = torch.stack(signals).mean(dim=0)
        noise = torch.randn_like(s_hat) * math.sqrt(self.cfg.channel_noise_sigma2 / 2.0)
        
        if method == 'randk':
            gen = torch.Generator(device=s_hat.device).manual_seed(round_idx) 
            idx = torch.randperm(self.d, generator=gen, device=s_hat.device)[:k]
            mask = torch.zeros_like(s_hat); mask[idx] = 1.0
            s_noisy = s_hat + (noise / (beta * self.cfg.aggregation_gain)) * mask
        else:
            s_noisy = s_hat + (noise / (beta * self.cfg.aggregation_gain))
            
        return s_noisy, regime

# ==========================================
# 6. 主程序
# ==========================================
def run():
    config = AirCompConfig()
    dataset = FEMNISTDataset(config)
    Path("ef_result1_exp2para").mkdir(exist_ok=True)
    
    exps = [
        {"name": "CONFLUX (Ours)", "method": "topk", "ratio": 0.3, "color": "#d62728"},
        {"name": "PFELS (Rand-k)", "method": "randk", "ratio": 0.5, "color": "#1f77b4"},
        {"name": "WFL-PDP (Full)", "method": "full", "ratio": 1.0, "color": "#2ca02c"}
    ]
    
    d_total = sum(p.numel() for p in StableCNN().parameters())
    channel = AirCompChannel(config, d_total)
    
    print(f"[INFO] T={config.total_rounds} Evaluation: Strict Cross-Experiment Synchronization")
    
    for exp in exps:
        set_seed(42)  
        
        print(f"\n🔥 基准对比测试: {exp['name']} (ratio={exp['ratio']})")
        
        global_model = StableCNN(config.num_classes).to(config.device)
        clients = [FLClient(i, config, dataset) for i in range(config.num_devices)]
        logs = []
        
        for r in range(config.total_rounds):
            lr = max(config.initial_lr * (config.lr_decay ** r), config.min_lr)
            sigs = []; cur_C = 0
            global_state = global_model.state_dict()
            global_params = list(global_model.parameters())
            
            for c in clients:
                s_t, cur_C = c.local_update(global_state, global_params, exp['method'], exp['ratio'], lr, r)
                sigs.append(s_t)
            
            s_noisy, regime = channel.aggregate(sigs, exp['method'], exp['ratio'], cur_C, r)
            
            with torch.no_grad():
                idx = 0
                for p in global_model.parameters():
                    num = p.numel(); p.add_(s_noisy[idx:idx+num].view(p.shape)); idx += num
            
            del sigs, s_noisy
            if r % 50 == 0: gc.collect(); torch.cuda.empty_cache()
            
            # 评估逻辑：模型切入 eval 模式，关闭 dropout，因此评估不消耗随机数，完美保持 RNG 队列对齐。
            global_model.eval(); correct, total = 0, 0
            with torch.no_grad():
                for tx, ty in dataset.test_loader:
                    tx, ty = tx.to(config.device), ty.to(config.device)
                    correct += (global_model(tx).argmax(1) == ty).sum().item(); total += ty.size(0)
            acc = correct / total * 100
            logs.append({"round": r+1, "acc": acc})
            
            if r == 0 or (r+1) % 10 == 0:
                print(f"Round {r+1:3d} | Acc: {acc:5.2f}% | Regime: {regime}")

        # 计算最后 10 轮的平滑准确率
        smoothed_acc = sum([item['acc'] for item in logs[-10:]]) / 10.0
        print("-" * 75)
        print(f"✅ {exp['name']:<15} | Final Smoothed Acc (Last 10 Rounds): {smoothed_acc:5.2f}%")
        print("-" * 75)

        with open(f"ef_result1_exp2para/res_{exp['method']}.json", 'w') as f: json.dump(logs, f)
        del global_model, clients; gc.collect(); torch.cuda.empty_cache()

    # 绘图部分
    plt.figure(figsize=(9, 6))
    for exp in exps:
        with open(f"ef_result1_exp2para/res_{exp['method']}.json", 'r') as f:
            d = json.load(f)
            plt.plot([i['round'] for i in d], [i['acc'] for i in d], 
                     label=exp['name'], color=exp['color'], lw=1.5, alpha=0.9)
                     
    plt.xlabel("Communication Round", fontsize=14)
    plt.ylabel("Test Accuracy (%)", fontsize=14)
    plt.title(f"Convergence Comparison under DP ($\epsilon={config.privacy_epsilon}, T=400$)", fontsize=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, ls='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("ef_result1_exp2para/final_comparison.png", dpi=300)

if __name__ == "__main__":
    run()