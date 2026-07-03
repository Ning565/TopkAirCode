# 完整系统 ADC/DP/功率约束实验

## 实验设置

- 通信轮数：200
- epsilon：100000000.0，delta：0.001
- 实际压缩率：femnist:topk=0.20, femnist:randk=0.35
- OFDM 子载波数 M=2000，过采样倍数=4，ADC backoff gamma=2.5

## 最终结果

| 数据集 | 方法 | 压缩率 | 最终准确率 | PAPR P99 | NCE | b* | 约束状态 |
|---|---|---:|---:|---:|---:|---:|---|
| femnist | topk | 0.20 | 83.41 | 11.22 | 1.32e-04 | 7.034e-01 | power |
| femnist | randk | 0.35 | 81.49 | 11.09 | 1.34e-04 | 5.317e-01 | power |
