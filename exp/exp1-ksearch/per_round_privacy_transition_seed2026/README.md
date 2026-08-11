# 单轮客户端级隐私约束过渡实验

本实验采用 add/remove-client 单轮隐私、公共更新范数裁剪和固定完整 OFDM 资源栅格。单轮预算不声明为多轮组合隐私保证。

| 扫描 | 数值 | 方法 | 最优 $p$ | 首个功率约束 $p$ | 最优点约束 |
|---|---:|---|---:|---:|---|
| baseline | baseline | TOPK | 0.74 | 0.47 | power |
| baseline | baseline | RANDK | 1.00 | 0.47 | power |
| baseline | baseline | FULL | 1.00 | 1.0 | power |
| epsilon_per_round | 0.5 | TOPK | 0.88 | 无 | privacy |
| epsilon_per_round | 0.5 | RANDK | 1.00 | 无 | privacy |
| epsilon_per_round | 0.5 | FULL | 1.00 | 无 | privacy |
| epsilon_per_round | 1 | TOPK | 0.88 | 无 | privacy |
| epsilon_per_round | 1 | RANDK | 1.00 | 无 | privacy |
| epsilon_per_round | 1 | FULL | 1.00 | 无 | privacy |
| epsilon_per_round | 10 | TOPK | 0.74 | 0.15 | power |
| epsilon_per_round | 10 | RANDK | 1.00 | 0.15 | power |
| epsilon_per_round | 10 | FULL | 1.00 | 1.0 | power |
| epsilon_per_round | 2 | TOPK | 0.86 | 无 | privacy |
| epsilon_per_round | 2 | RANDK | 1.00 | 无 | privacy |
| epsilon_per_round | 2 | FULL | 1.00 | 无 | privacy |
| epsilon_per_round | 3 | TOPK | 0.85 | 无 | privacy |
| epsilon_per_round | 3 | RANDK | 1.00 | 无 | privacy |
| epsilon_per_round | 3 | FULL | 1.00 | 无 | privacy |
| epsilon_per_round | 4 | TOPK | 0.74 | 0.7 | power |
| epsilon_per_round | 4 | RANDK | 1.00 | 0.7 | power |
| epsilon_per_round | 4 | FULL | 1.00 | 1.0 | power |
| epsilon_per_round | 8 | TOPK | 0.74 | 0.21 | power |
| epsilon_per_round | 8 | RANDK | 1.00 | 0.21 | power |
| epsilon_per_round | 8 | FULL | 1.00 | 1.0 | power |
| snr_db | 0 | TOPK | 0.66 | 0.02 | power |
| snr_db | 0 | RANDK | 1.00 | 0.02 | power |
| snr_db | 0 | FULL | 1.00 | 1.0 | power |
| snr_db | 10 | TOPK | 0.70 | 0.15 | power |
| snr_db | 10 | RANDK | 1.00 | 0.15 | power |
| snr_db | 10 | FULL | 1.00 | 1.0 | power |
| snr_db | 20 | TOPK | 0.88 | 无 | privacy |
| snr_db | 20 | RANDK | 1.00 | 无 | privacy |
| snr_db | 20 | FULL | 1.00 | 无 | privacy |
| snr_db | 25 | TOPK | 0.88 | 无 | privacy |
| snr_db | 25 | RANDK | 1.00 | 无 | privacy |
| snr_db | 25 | FULL | 1.00 | 无 | privacy |
| snr_db | 30 | TOPK | 0.88 | 无 | privacy |
| snr_db | 30 | RANDK | 1.00 | 无 | privacy |
| snr_db | 30 | FULL | 1.00 | 无 | privacy |
| snr_db | 5 | TOPK | 0.68 | 0.05 | power |
| snr_db | 5 | RANDK | 1.00 | 0.05 | power |
| snr_db | 5 | FULL | 1.00 | 1.0 | power |
