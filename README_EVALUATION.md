# Agentless 完整测评流程

## 🚀 快速开始

### 前置条件

1. **vLLM 服务已启动**（端口 8003）
2. **conda 环境已配置**（agentless 环境）
3. **数据集和仓库已准备**

### 一键运行

```bash
cd /workspace/locbench/Agentless
./run_full_evaluation.sh
```

## 📋 脚本功能

脚本会自动完成以下步骤：

1. **环境检查**
   - 检查 conda 环境
   - 检查 vLLM 服务
   - 检查数据集和仓库

2. **清理工作**
   - 清理残留的 git worktree
   - 清理临时目录

3. **三级定位**
   - 步骤 1: 文件级定位
   - 步骤 2: 相关元素定位
   - 步骤 3: 编辑位置定位

4. **结果评估**
   - 使用评估脚本计算准确率
   - 生成评估报告

## 📊 输出文件

所有结果保存在 `results/locbench/` 目录下：

- `file_level/loc_outputs.jsonl` - 文件级定位结果
- `related_elements/loc_outputs.jsonl` - 相关元素定位结果
- `edit_location_samples/loc_outputs.jsonl` - 编辑位置定位结果（最终结果）
- `evaluation_results.json` - 评估结果

## ⚙️ 配置参数

可以在脚本开头修改以下参数：

```bash
VLLM_URL="http://localhost:8003/v1"      # vLLM 服务地址
MODEL_NAME="qwen2.5-32b"                  # 模型名称
DATASET_PATH="/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl"
REPO_ROOT="/workspace/locbench/repos/locbench_repos"
OUTPUT_BASE="results/locbench"            # 输出目录
NUM_THREADS=1                              # 线程数（单线程避免冲突）
MAX_CONTEXT_LENGTH=32768                   # 上下文长度
```

## 🔍 监控进度

```bash
# 查看实时日志
tail -f results/locbench/file_level/localization_logs/*.log

# 查看处理进度
wc -l results/locbench/*/loc_outputs.jsonl

# 查看 vLLM 日志
tail -f /workspace/logs/vllm_qwen32b.log
```

## ⚠️ 注意事项

1. **单线程运行**：避免 git worktree 并发冲突
2. **跳过已处理**：使用 `--skip_existing`，可安全重复运行
3. **上下文长度**：确保 vLLM 的 `--max-model-len` ≥ Agentless 的 `--max_context_length`
4. **Python 2 警告**：可以忽略，不影响结果

## 🐛 故障排除

### vLLM 服务未运行
```bash
# 启动 vLLM 服务
CUDA_VISIBLE_DEVICES=3,4,5,6 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/model/Qwen__Qwen2.5-32B-Instruct/Qwen/Qwen2.5-32B-Instruct \
  --host 0.0.0.0 --port 8003 \
  --served-model-name qwen2.5-32b \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  > /workspace/logs/vllm_qwen32b.log 2>&1 &
```

### Git worktree 冲突
```bash
# 清理所有 worktree
cd /workspace/locbench/repos/locbench_repos
for repo in */; do cd "$repo" && git worktree prune && cd ..; done
rm -rf /tmp/agentless_worktree_*
```

### 评估失败
如果评估脚本失败，可以手动运行：
```bash
cd /workspace/locbench/Agentless
python results/locbench/run_evaluation.py \
  results/locbench/edit_location_samples/loc_outputs.jsonl \
  /workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  results/locbench/evaluation_results.json
```
