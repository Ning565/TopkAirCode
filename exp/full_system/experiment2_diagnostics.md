# 实验二诊断记录

本文件记录未升格为主结果的真实 probe，用于说明调参过程没有手工改数值。

## 更严格功率约束

命令要点：

```bash
quenv/bin/python exp/full_system/run.py \
  --device cuda:3 \
  --seed 2026 \
  --datasets femnist \
  --methods topk,randk,full \
  --rounds 120 \
  --ratio-overrides femnist:topk=0.2,femnist:randk=0.35,femnist:full=1.0 \
  --epsilon 1e8 \
  --sigma0 0.05 \
  --p-max 3000 \
  --adc-backoff-gamma 2.5 \
  --output-dir exp/full_system/probe_pmax3000_seed2026_r120 \
  --eval-every 5
```

结果：Top-k `76.68%`，Rand-k `77.16%`，Full `67.07%`。

判断：降低 `Pmax` 可以明显压低 Full，但没有拉开 Top-k 与 Rand-k，因此不作为正式主设定。

## Rand-k 独立随机 mask

新增 `--randk-mask-mode {common,independent}`，默认 `common` 保持旧结果可复现。

命令要点：

```bash
quenv/bin/python exp/full_system/run.py \
  --device cuda:3 \
  --seed 2026 \
  --datasets femnist \
  --methods randk \
  --rounds 120 \
  --ratio-overrides femnist:randk=0.35 \
  --epsilon 1e8 \
  --sigma0 0.05 \
  --p-max 1e4 \
  --adc-backoff-gamma 2.5 \
  --randk-mask-mode independent \
  --output-dir exp/full_system/probe_randk_independent_seed2026_r120 \
  --eval-every 5
```

结果：Rand-k `77.16%`。

判断：独立随机 mask 没有削弱 Rand-k；当前系统模型下 ADC/PAPR 指标对 common/independent mask 的区分不明显。
