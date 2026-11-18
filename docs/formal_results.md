# 子问题四：Dense P1/P4 已完成结果汇总

> 更新日期：2026-11-12（UTC+8）  
> 数据来源：`/mnt/hd2/lewis/Subproblem4UnifiedReproduction/experiments/stage1/evaluations/dense_original_data_v1` 下 18 个正式评测条件的逐样本结果。  
> 本文只统计当前已完成的 Dense P1/P4；按当前安排，P2/P3 暂不评测，因此本文不是 P1—P4 全协议完成报告，也不混入 T2T、LCF、LCF-X 或 SwaT 结果。

## 1. 结果范围与完整性

### 1.1 模型组

| 模型组 | Sender | Receiver |
|---|---|---|
| M1 | Qwen3-0.6B | Qwen3-1.7B |
| M2 | Qwen3-0.6B | Qwen3-4B |
| M3 | Qwen3-4B | Qwen3-0.6B |

### 1.2 已完成范围

| 方法 | 每个模型组的已完成条件 | 每组样本数 | 三组总样本数 | 状态 |
|---|---:|---:|---:|---|
| Dense | 6 个：GSM8K、HotpotQA、HumanEval 的 P1/P4 | 14,802 | 44,406 | 当前范围全部完成 |

完整性核对结果如下：

- 18 个条件均存在 `summary.json` 和 `predictions.jsonl`，摘要中的完成条数与逐样本行数一致。
- 每个条件内的 `record_id` 均无重复；P1 与 P4 使用相同的对应测试样本集合。
- 6 个 HumanEval 条件均有 164 条独立执行评分记录，任务 ID 无重复。
- GSM8K、HotpotQA、HumanEval 每个模型组分别为 `2 × 1,319`、`2 × 5,918`、`2 × 164` 条，共 14,802 条。
- `queue_M1.status`、`queue_M2.status`、`queue_M3.status` 均为 `completed_all`。根目录的旧 `queue.status` 保留了一次早期启动失败记录，不代表后续三组正式队列的最终状态。

### 1.3 本文协议

| 协议 | Sender 初始输入 | Receiver 初始输入 | 本文测试范围 |
|---|---|---|---|
| P1：共享完整输入 | 完整问题与上下文 | 完整问题与上下文 | GSM8K、HotpotQA、HumanEval |
| P4：完整任务只在上游 | 完整任务 | 仅不含题目信息的 assistant/generation prefix | GSM8K、HotpotQA、HumanEval |

结果目录中的旧名称 `protocol_a` 对应 P1，`protocol_c` 对应 P4。P2（互补证据）和 P3（全部证据在上游、Receiver 仅有问题）目前没有纳入本文，也不会从 P1/P4 结果推算。

## 2. 训练配置与转换器规模

### 2.1 训练配置

- 方法：DenseLatentCommunication 原文两阶段训练路径。
- 训练数据共 12,836 条：GSM8K 7,473 条、MATH-algebra 4,244 条、ARC-C 1,119 条。
- Phase 1：Cache reconstruction，2,000 steps。
- Receiver self traces：greedy generation，thinking 开启，最多 768 个新 token。
- Phase 2：generation objective，2,000 steps；完整上下文已知与未知场景按 50%/50% 混合训练。
- Adapter expansion 为 16，权重与推理精度为 BF16。
- 本轮发送转换后的完整 Receiver-compatible KV Cache，没有进行 Cache group 选择或 Figure 5 式保留比例实验。

### 2.2 参数量与权重

参数量使用十进制量级，`1 M = 10^6` 个参数；权重文件大小使用十进制 MB，`1 MB = 10^6 bytes`。冻结的 Sender 和 Receiver 基座模型不计入转换器参数量。

| 模型组 | 可训练转换器参数量 | Phase 2 权重文件 | Phase 2 完成时间 |
|---|---:|---:|---|
| M1 | 235.86 M | 471.79 MB | 2026-10-16 07:07 |
| M2 | 303.24 M | 606.59 MB | 2026-10-19 21:05 |
| M3 | 235.86 M | 471.79 MB | 2026-10-27 05:42 |

## 3. 指标、单位与耗时口径

### 3.1 质量与置信区间

- GSM8K 使用最终答案 Exact Match；HotpotQA 使用 Answer EM 与 token F1；HumanEval 使用对 164 个题目的执行型 pass@1。
- 表中方括号为从逐样本结果重新计算的 95% 置信区间。二值 EM/pass@1 使用 Wilson 区间，HotpotQA F1 使用样本均值的正态近似区间并截断到 `[0, 1]`。
- 质量结果没有直接照抄摘要；本文重新读取了全部 44,406 条预测，以及 HumanEval 的 984 条执行评分记录。

### 3.2 通信量

- “平均通信量”采用逐样本 `logical_transmission_bytes`，在当前 Dense 评测器中等于转换后的完整 BF16 Receiver-compatible KV Cache 张量字节数。
- 该字段不包含真实序列化协议头、位置等元数据，也不包含网络栈开销，因此应理解为逻辑张量载荷，而不是抓包得到的线上总字节数。
- 通信量使用十进制单位：`1 MB = 10^6 bytes`，`1 GB = 10^9 bytes`。本批单样本均值均小于 1 GB，表中统一显示为 MB。

### 3.3 分项耗时

表中按用户指定的三段口径重新相加：

1. **上游** = Sender prefill，即生成源 KV Cache 的模型前向；
2. **转换** = Dense adapter 将源 KV Cache 转为 Receiver-compatible KV Cache；
3. **下游** = Receiver 本地 prefix prefill + Receiver 逐 token decode；
4. **合计计算耗时** = 上游 + 转换 + 下游。

这些分项均由 CUDA Event 测得。当前 Dense 评测没有单独记录提示组装、分词、反分词、打包/解包和真实传输耗时，所以它们不在三段合计内；原始 `total_compute_ms` 还混有 Python 调度与同步时间，本文为保持“上游 + 转换 + 下游”的固定结构，没有采用该 wall-time 字段。小于 1 秒的耗时用 ms，达到或超过 1 秒的耗时用 s。

### 3.4 测量环境

- 设备：NVIDIA GeForce RTX 3090；batch size 1；BF16；greedy decoding；Receiver thinking 关闭。
- Sender 输入按训练配置最多保留 512 tokens。
- HotpotQA Receiver 最大生成 64 tokens；GSM8K 与 HumanEval 最大生成 512 tokens。
- Sender、转换器和 Receiver 在同一张 GPU 上执行；未测量真实网络传输。

## 4. Dense P1/P4 正式结果

| 组 | 数据集/协议 | 样本 | 质量（均值，95% CI） | 平均通信量 | 平均计算耗时（上游 + 转换 + 下游 = 合计） |
|---|---|---:|---|---:|---:|
| M1 | GSM8K/P1 | 1,319 | EM 72.71% [70.24, 75.04] | 10.555 MB | 33.22 ms + 27.52 ms + 4.64 s = 4.70 s |
| M1 | GSM8K/P4 | 1,319 | EM 24.64% [22.39, 27.04] | 10.555 MB | 28.41 ms + 30.84 ms + 5.29 s = 5.35 s |
| M1 | HotpotQA/P1 | 5,918 | EM 2.20% [1.85, 2.60]; F1 19.11% [18.57, 19.64] | 58.660 MB | 41.19 ms + 40.05 ms + 957.58 ms = 1.04 s |
| M1 | HotpotQA/P4 | 5,918 | EM 0.00% [0.00, 0.06]; F1 1.03% [0.97, 1.10] | 58.660 MB | 40.08 ms + 37.08 ms + 1.66 s = 1.74 s |
| M1 | HumanEval/P1 | 164 | pass@1 50.00% [42.44, 57.56] | 17.688 MB | 36.58 ms + 39.74 ms + 4.26 s = 4.34 s |
| M1 | HumanEval/P4 | 164 | pass@1 0.00% [0.00, 2.29] | 17.688 MB | 30.82 ms + 28.47 ms + 9.40 s = 9.46 s |
| M2 | GSM8K/P1 | 1,319 | EM 88.93% [87.12, 90.51] | 13.571 MB | 31.92 ms + 40.03 ms + 6.92 s = 7.00 s |
| M2 | GSM8K/P4 | 1,319 | EM 13.57% [11.83, 15.53] | 13.571 MB | 27.01 ms + 33.49 ms + 5.62 s = 5.68 s |
| M2 | HotpotQA/P1 | 5,918 | EM 0.00% [0.00, 0.06]; F1 11.28% [11.06, 11.51] | 75.420 MB | 30.07 ms + 32.02 ms + 2.19 s = 2.25 s |
| M2 | HotpotQA/P4 | 5,918 | EM 0.00% [0.00, 0.06]; F1 0.82% [0.76, 0.88] | 75.420 MB | 32.00 ms + 34.57 ms + 2.20 s = 2.26 s |
| M2 | HumanEval/P1 | 164 | pass@1 61.59% [53.96, 68.68] | 22.741 MB | 29.94 ms + 35.73 ms + 2.16 s = 2.22 s |
| M2 | HumanEval/P4 | 164 | pass@1 0.00% [0.00, 2.29] | 22.741 MB | 26.16 ms + 27.09 ms + 13.67 s = 13.72 s |
| M3 | GSM8K/P1 | 1,319 | EM 51.78% [49.08, 54.47] | 10.555 MB | 34.35 ms + 21.08 ms + 3.40 s = 3.46 s |
| M3 | GSM8K/P4 | 1,319 | EM 3.26% [2.43, 4.36] | 10.555 MB | 33.75 ms + 21.15 ms + 3.63 s = 3.69 s |
| M3 | HotpotQA/P1 | 5,918 | EM 0.37% [0.25, 0.56]; F1 14.78% [14.36, 15.20] | 58.660 MB | 78.45 ms + 22.05 ms + 679.35 ms = 779.85 ms |
| M3 | HotpotQA/P4 | 5,918 | EM 0.00% [0.00, 0.06]; F1 0.41% [0.36, 0.46] | 58.660 MB | 78.31 ms + 21.66 ms + 1.28 s = 1.38 s |
| M3 | HumanEval/P1 | 164 | pass@1 26.22% [20.09, 33.44] | 17.688 MB | 41.81 ms + 21.36 ms + 3.77 s = 3.84 s |
| M3 | HumanEval/P4 | 164 | pass@1 0.00% [0.00, 2.29] | 17.688 MB | 41.84 ms + 21.33 ms + 6.69 s = 6.75 s |

## 5. 主要观察

1. **P1 在 GSM8K 与 HumanEval 上有效，但 HotpotQA 明显较弱。** M2 的 GSM8K EM 为 88.93%，HumanEval pass@1 为 61.59%，均为三组最高；HotpotQA P1 的三组 EM 只有 0.00%—2.20%，F1 为 11.28%—19.11%。这与训练集包含 GSM8K、MATH-algebra、ARC-C，但不包含 HotpotQA 的任务域差异一致，不过这里只能说明相关性，不能单凭本轮结果断言唯一原因。
2. **P4 质量显著下降。** 三组 HumanEval P4 的 pass@1 均为 0；GSM8K P4 只有 M1 保留到 24.64%，M2/M3 分别为 13.57% 和 3.26%；HotpotQA P4 的 EM 全为 0。尽管 Phase 2 采用已知/未知上下文各 50% 的训练方式，当前权重在“Receiver 完全看不到题目文本”时仍没有稳定泛化。
3. **P1/P4 通信量相同是预期现象。** 两个协议的 Sender 都处理同一完整任务，并发送 100% 转换后 Cache；改变的只是 Receiver 是否本地持有完整输入。由于没有做 group 筛选，通信量不会随协议变化。
4. **通信量随输入长度与 Receiver Cache 结构变化。** HotpotQA 输入更长，因此平均载荷显著高于 GSM8K 和 HumanEval。M2 的 Receiver 有 36 层，而 M1/M3 的 Receiver 都是 28 层且 KV head/head-dim 配置相同，所以 M2 的载荷约为对应 M1/M3 的 `36/28`；M1 与 M3 的平均通信量相同并不表示源模型计算相同。
5. **当前计算耗时主要由下游生成决定。** Dense 转换均值约为 21.08—40.05 ms，而下游阶段约为 679.35 ms—13.67 s。HumanEval P4 虽然质量为 0，但仍可能生成很长的无效输出，因此“质量低”不必然对应“下游耗时短”。
6. **当前只是已完成协议的正式统计。** P1/P4 的 44,406 条结果完整有效，但统一实验计划中的 HotpotQA P2/P3 尚未执行；本文不将它们标成 Dense 全协议完成，也未加入相对文本基线差值。

## 6. 结果、权重与配置位置

- 正式评测根目录：`/mnt/hd2/lewis/Subproblem4UnifiedReproduction/experiments/stage1/evaluations/dense_original_data_v1`
- Dense Phase 1/2 权重：`/mnt/hd2/lewis/Subproblem4UnifiedReproduction/experiments/stage1/converters/dense_gsm8k_math_algebra_arc_c`
- 训练与模型配置：`/home/lewis/Subproblem4UnifiedReproduction/configs/stage1/converters/dense_gsm8k_math_algebra_arc_c`
- 训练数据：`/mnt/hd2/lewis/Subproblem4UnifiedReproduction/data/DenseLatentCommunication/train.jsonl`
- 统一测试数据：`/mnt/hd2/lewis/Subproblem4UnifiedReproduction/data/canonical/eval`

每个条件的生成文本、质量字段、输入/输出 token 数、Cache 字节数及分项耗时保存在 `predictions.jsonl`；HumanEval 的独立执行结果保存在 `humaneval_scores.jsonl`；汇总配置保存在 `summary.json`。本文没有依赖 NFS 路径，也没有在统计过程中启动、停止或修改任何 GPU 评测进程。
