# 完整系统 ADC/DP/功率约束实验

## 实验设置

- 通信轮数：200
- epsilon：100000000.0，delta：0.001
- 实际压缩率：femnist:topk=0.10, femnist:randk=0.50, femnist:full=1.00
- OFDM 子载波数 M=2000，过采样倍数=4，ADC backoff gamma=2.5

## 最终结果

| 数据集 | 方法 | 压缩率 | 最终准确率 | PAPR P99 | NCE | b* | 约束状态 |
|---|---|---:|---:|---:|---:|---:|---|
| femnist | topk | 0.10 | 82.93 | 11.30 | 1.34e-04 | 9.948e-01 | power |
| femnist | randk | 0.50 | 83.17 | 11.16 | 1.34e-04 | 4.449e-01 | power |
| femnist | full | 1.00 | 77.88 | 11.16 | 1.34e-04 | 3.146e-01 | power |
