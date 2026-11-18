# 独立仓库与迁移说明

本仓库是 Dense Latent Communication 当前复现状态的独立快照，不包含原仓库 Git
历史或 remote。根目录是 Dense 实现；`reproduction/current/` 是本次子问题四 M1/M2/M3
训练和统一协议测评的精确配置/脚本快照。

## 已纳入 Git

- Dense adapter、Phase 1/2、trace 生成、cache 注入与测试代码；
- M1 `0.6B→1.7B`、M2 `0.6B→4B`、M3 `4B→0.6B` 正式配置；
- 正式训练、P1/P4 测评和 HotpotQA 无 token 上限补测入口；
- 固定提示词 ZIP、正式结果、实验计划及跨方法对比文档。

模型、规范化数据、Phase 1/2 权重和逐样本结果不进入 Git。完整清单见
`migration/external_assets.json`。

## 迁移流程

```bash
# 原机器确认资产
python tools/verify_external_assets.py --mode current --profile resume
python tools/verify_external_assets.py --mode current --profile results

# 制作可离线 clone 的代码 bundle
bash tools/create_code_bundle.sh /path/to/export
```

按 manifest 将每个 `current_path` 复制到新仓库的 `recommended_path` 后执行：

```bash
python tools/verify_external_assets.py --mode restored --profile evaluate
```

本次运行脚本保存了旧机器的绝对路径，迁移后用内置工具统一改写；默认是 dry-run：

```bash
python tools/relocate_snapshot.py --root reproduction/current \
  --map /home/lewis/Subproblem4UnifiedReproduction="$PWD/reproduction/current" \
  --map /mnt/hd2/lewis/Subproblem4UnifiedReproduction="$PWD/assets/unified" \
  --map /mnt/hd2/lewis/SwaTReproduction/models/qwen="$PWD/assets/models/qwen" \
  --map /home/lewis/DenseLatentCommunication="$PWD"
```

确认输出后追加 `--write`。Python 环境路径也可用同一工具替换，或直接调用根目录的
`scripts/train_phase1.py`、`scripts/generate_traces.py`、`scripts/train_phase2.py`，并传入
改写后的 JSON 配置。

固定提示词位于 `reproduction/prompts/提示词模板.zip`。统一 Dense evaluator 已改为从
当前仓库导入 `dense_kv`，不再要求原统一实验仓库存在。

