#!/bin/bash
# Agentless 完整测评流程脚本
# 包含：三级定位 + 结果评估
#
# 使用方法：
#   1. 修改脚本顶部的"模型配置"部分，设置模型路径和相关参数
#   2. 确保 vLLM 服务已启动（或使用脚本提供的启动命令）
#   3. 运行脚本: ./run_full_evaluation.sh
#
# 切换模型示例：
#   MODEL_PATH="/workspace/model/Qwen__Qwen3-32B/Qwen/Qwen3-32B"
#   MODEL_NAME="qwen3-32b"
#   MAX_MODEL_LEN=65536  # 根据模型调整
#

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================
# 模型配置 - 修改这些变量即可切换模型
# ============================================================
# API 模型名称（Agentless 调用时使用，需要与 vLLM 的 --served-model-name 一致）
MODEL_NAME="qwen3-32b"
# vLLM 服务端口
VLLM_PORT=8003
# vLLM 服务地址（自动生成）
VLLM_URL="http://localhost:${VLLM_PORT}/v1"
# Agentless 使用的上下文长度（通常与 vLLM 的 --max-model-len 一致）
# 注意：Qwen3-32B 的 max_position_embeddings=40960，建议使用 32768 或更小
MAX_CONTEXT_LENGTH=32768          # 根据模型能力调整，模型最大支持 40960

# ============================================================
# 常用模型配置示例（取消注释并修改即可使用）
# ============================================================
# Qwen2.5-32B-Instruct (32K 上下文)
# MODEL_NAME="qwen2.5-32b"
# VLLM_PORT=8003
# MAX_CONTEXT_LENGTH=32768

# Qwen3-32B (64K 上下文)
# MODEL_NAME="qwen3-32b"
# VLLM_PORT=8003
# MAX_CONTEXT_LENGTH=65536

# 其他模型...
# MODEL_NAME="your-model-name"
# VLLM_PORT=8003
# MAX_CONTEXT_LENGTH=根据模型调整

# ============================================================
# 其他配置
# ============================================================
DATASET_PATH="/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl"
REPO_ROOT="/workspace/locbench/repos/locbench_repos"
OUTPUT_BASE="results/locbench"
NUM_THREADS=1

echo "=========================================="
echo "🚀 Agentless 完整测评流程"
echo "=========================================="
echo ""

# 1. 检查环境
echo "📋 步骤 0: 检查环境..."
cd /workspace/locbench/Agentless

# 检查 conda 环境
if ! conda env list | grep -q "agentless"; then
    echo -e "${RED}❌ 错误: 未找到 agentless conda 环境${NC}"
    exit 1
fi

# 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate agentless
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 检查 vLLM 服务
echo "🔍 检查 vLLM 服务..."
echo "   模型名称: $MODEL_NAME"
echo "   服务地址: $VLLM_URL"
if ! curl -s "${VLLM_URL}/models" > /dev/null 2>&1; then
    echo -e "${RED}❌ 错误: vLLM 服务未运行！${NC}"
    echo "   请先启动 vLLM 服务（确保服务地址为 $VLLM_URL）"
    echo "   确保 --served-model-name 参数设置为: $MODEL_NAME"
    exit 1
fi
echo -e "${GREEN}✅ vLLM 服务正常运行${NC}"

# 检查数据集和仓库
if [ ! -f "$DATASET_PATH" ]; then
    echo -e "${RED}❌ 错误: 数据集文件不存在: $DATASET_PATH${NC}"
    exit 1
fi

if [ ! -d "$REPO_ROOT" ]; then
    echo -e "${RED}❌ 错误: 仓库目录不存在: $REPO_ROOT${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 环境检查通过${NC}"
echo ""

# 2. 清理残留的 worktree（避免冲突）
echo "🧹 清理残留的 worktree..."
cd "$REPO_ROOT"
for repo in */; do
    cd "$repo" 2>/dev/null && git worktree prune 2>/dev/null || true
    cd .. 2>/dev/null
done
rm -rf /tmp/agentless_worktree_* 2>/dev/null || true
echo -e "${GREEN}✅ 清理完成${NC}"
echo ""

# 3. 步骤 1: 文件级定位
echo "=========================================="
echo "📁 步骤 1: 文件级定位"
echo "=========================================="
cd /workspace/locbench/Agentless

python agentless/fl/localize.py \
    --file_level \
    --output_format locbench \
    --dataset_path "$DATASET_PATH" \
    --local_repo_root "$REPO_ROOT" \
    --output_folder "${OUTPUT_BASE}/file_level" \
    --model "$MODEL_NAME" \
    --backend openai \
    --max_context_length "$MAX_CONTEXT_LENGTH" \
    --num_threads "$NUM_THREADS" \
    --skip_existing

if [ ! -f "${OUTPUT_BASE}/file_level/loc_outputs.jsonl" ]; then
    echo -e "${RED}❌ 错误: 步骤 1 输出文件未生成${NC}"
    exit 1
fi

FILE_COUNT=$(wc -l < "${OUTPUT_BASE}/file_level/loc_outputs.jsonl")
echo -e "${GREEN}✅ 步骤 1 完成！处理了 $FILE_COUNT 个实例${NC}"
echo ""

# 4. 步骤 2: 相关元素定位
echo "=========================================="
echo "🔗 步骤 2: 相关元素定位"
echo "=========================================="

python agentless/fl/localize.py \
    --related_level \
    --output_format locbench \
    --dataset_path "$DATASET_PATH" \
    --local_repo_root "$REPO_ROOT" \
    --output_folder "${OUTPUT_BASE}/related_elements" \
    --model "$MODEL_NAME" \
    --backend openai \
    --max_context_length "$MAX_CONTEXT_LENGTH" \
    --top_n 3 \
    --compress_assign \
    --compress \
    --start_file "${OUTPUT_BASE}/file_level/loc_outputs.jsonl" \
    --num_threads "$NUM_THREADS" \
    --skip_existing

if [ ! -f "${OUTPUT_BASE}/related_elements/loc_outputs.jsonl" ]; then
    echo -e "${RED}❌ 错误: 步骤 2 输出文件未生成${NC}"
    exit 1
fi

RELATED_COUNT=$(wc -l < "${OUTPUT_BASE}/related_elements/loc_outputs.jsonl")
echo -e "${GREEN}✅ 步骤 2 完成！处理了 $RELATED_COUNT 个实例${NC}"
echo ""

# 5. 步骤 3: 编辑位置定位
echo "=========================================="
echo "📍 步骤 3: 编辑位置定位"
echo "=========================================="

python agentless/fl/localize.py \
    --fine_grain_line_level \
    --output_format locbench \
    --dataset_path "$DATASET_PATH" \
    --local_repo_root "$REPO_ROOT" \
    --output_folder "${OUTPUT_BASE}/edit_location_samples" \
    --model "$MODEL_NAME" \
    --backend openai \
    --max_context_length "$MAX_CONTEXT_LENGTH" \
    --top_n 3 \
    --compress \
    --temperature 0.8 \
    --num_samples 4 \
    --start_file "${OUTPUT_BASE}/related_elements/loc_outputs.jsonl" \
    --num_threads "$NUM_THREADS" \
    --skip_existing

if [ ! -f "${OUTPUT_BASE}/edit_location_samples/loc_outputs.jsonl" ]; then
    echo -e "${RED}❌ 错误: 步骤 3 输出文件未生成${NC}"
    exit 1
fi

EDIT_COUNT=$(wc -l < "${OUTPUT_BASE}/edit_location_samples/loc_outputs.jsonl")
echo -e "${GREEN}✅ 步骤 3 完成！处理了 $EDIT_COUNT 个实例${NC}"
echo ""

# 6. 评估结果
echo "=========================================="
echo "📊 步骤 4: 评估结果"
echo "=========================================="

cd /workspace/locbench/Agentless

# 创建评估脚本（使用 Agentless 自己的评估脚本）
cat > "${OUTPUT_BASE}/run_evaluation.py" << 'PYEOF'
import sys
import os

# 添加路径
sys.path.insert(0, '/workspace/locbench/Agentless')
sys.path.insert(0, '/workspace/locbench/LocAgent')

# 尝试使用 Agentless 的评估脚本
try:
    from evaluation.eval_metric import evaluate_results
    USE_AGENTLESS_EVAL = True
except ImportError:
    # 如果 Agentless 的评估脚本不存在，尝试使用 LocAgent 的
    try:
        import sys
        sys.path.insert(0, '/workspace/locbench/LocAgent')
        from evaluation.eval_metric import evaluate_results
        USE_AGENTLESS_EVAL = False
    except ImportError:
        print("❌ 错误: 找不到评估脚本")
        sys.exit(1)

import json
import pandas as pd

loc_file = sys.argv[1]
dataset_path = sys.argv[2]
output_file = sys.argv[3]

level2key_dict = {
    'file': 'found_files',
    'module': 'found_modules',
    'function': 'found_entities'
}

print(f"📊 评估文件: {loc_file}")
print(f"📊 数据集: {dataset_path}")

try:
    # 使用 Loc-Bench 数据集
    results = evaluate_results(
        loc_file=loc_file,
        level2key_dict=level2key_dict,
        dataset='czlll/Loc-Bench_V1',
        split='test',
        dataset_path=dataset_path
    )
    
    # 保存结果
    if isinstance(results, pd.DataFrame):
        results_dict = results.to_dict('records')[0]
    else:
        results_dict = results

    # 确保所有 key 都是字符串（处理多级列索引产生的 tuple key）
    # 例如：('file', 'acc@1') -> 'file.acc@1'
    def convert_keys_to_string(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                # 将 tuple key 转换为点号分隔的字符串
                if isinstance(k, tuple):
                    new_key = '.'.join(str(x) for x in k)
                elif isinstance(k, (int, float, bool)) or k is None:
                    new_key = str(k)
                else:
                    new_key = k
                new_dict[new_key] = convert_keys_to_string(v)
            return new_dict
        elif isinstance(obj, list):
            return [convert_keys_to_string(item) for item in obj]
        else:
            return obj

    results_dict = convert_keys_to_string(results_dict)

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print("\n📊 评估结果:")
    if isinstance(results, pd.DataFrame):
        print(results.to_string())
    else:
        print(json.dumps(results_dict, indent=2, ensure_ascii=False))
    print(f"\n✅ 结果已保存到: {output_file}")
    
except Exception as e:
    print(f"❌ 评估出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

# 运行评估
if [ -f "/workspace/locbench/LocAgent/evaluation/eval_metric.py" ]; then
    # 激活 LocAgent 环境（如果需要）
    if conda env list | grep -q "locagent"; then
        conda activate locagent
    fi
    
    python "${OUTPUT_BASE}/run_evaluation.py" \
        "${OUTPUT_BASE}/edit_location_samples/loc_outputs.jsonl" \
        "$DATASET_PATH" \
        "${OUTPUT_BASE}/evaluation_results.json"
    
    if [ -f "${OUTPUT_BASE}/evaluation_results.json" ]; then
        echo ""
        echo "📈 评估结果摘要:"
        python -c "
import json
try:
    with open('${OUTPUT_BASE}/evaluation_results.json', 'r') as f:
        data = json.load(f)
        for key, value in data.items():
            if isinstance(value, dict):
                print(f\"{key}:\")
                for k, v in value.items():
                    print(f\"  {k}: {v}\")
            else:
                print(f\"{key}: {value}\")
except Exception as e:
    print(f'读取结果时出错: {e}')
" 2>/dev/null || cat "${OUTPUT_BASE}/evaluation_results.json"
    fi
else
    echo -e "${YELLOW}⚠️  警告: LocAgent 评估脚本不存在，跳过评估${NC}"
    echo "   你可以手动运行评估："
    echo "   cd /workspace/locbench/LocAgent"
    echo "   python evaluation/eval_metric.py ..."
fi

echo ""
echo "=========================================="
echo -e "${GREEN}🎉 完整测评流程完成！${NC}"
echo "=========================================="
echo ""
echo "📁 结果文件位置:"
echo "   - 文件级定位: ${OUTPUT_BASE}/file_level/loc_outputs.jsonl"
echo "   - 相关元素定位: ${OUTPUT_BASE}/related_elements/loc_outputs.jsonl"
echo "   - 编辑位置定位: ${OUTPUT_BASE}/edit_location_samples/loc_outputs.jsonl"
echo "   - 评估结果: ${OUTPUT_BASE}/evaluation_results.json"
echo ""
echo "📊 统计信息:"
echo "   - 文件级: $FILE_COUNT 个实例"
echo "   - 相关元素: $RELATED_COUNT 个实例"
echo "   - 编辑位置: $EDIT_COUNT 个实例"
echo ""
