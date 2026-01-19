#!/bin/bash
# Agentless 单线程运行脚本

cd /workspace/locbench/Agentless
conda activate agentless
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 步骤 1: 文件级定位
echo "📁 步骤 1: 文件级定位（单线程）..."
python agentless/fl/localize.py \
    --file_level \
    --output_format locbench \
    --dataset_path /workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
    --local_repo_root /workspace/locbench/repos/locbench_repos \
    --output_folder results/locbench/file_level \
    --model qwen2.5-32b \
    --backend openai \
    --max_context_length 32768 \
    --num_threads 1 \
    --skip_existing

echo ""
echo "✅ 步骤 1 完成！"
echo "📊 检查结果:"
wc -l results/locbench/file_level/loc_outputs.jsonl 2>/dev/null || echo "结果文件尚未生成"
