#!/bin/bash
# Agentless 中间结果查看工具
# 用法: ./check_progress.sh

OUTPUT_BASE="results/locbench"
TOTAL_INSTANCES=560

echo "=========================================="
echo "📊 Agentless 中间结果查看工具"
echo "=========================================="
echo ""

# 步骤1结果
if [ -f "${OUTPUT_BASE}/file_level/loc_outputs.jsonl" ]; then
    FILE_COUNT=$(wc -l < "${OUTPUT_BASE}/file_level/loc_outputs.jsonl" | tr -d ' ')
    echo "✅ 步骤1（文件级定位）: 完成 $FILE_COUNT 个实例"
    if [ "$FILE_COUNT" -eq "$TOTAL_INSTANCES" ]; then
        echo "   📁 输出文件: ${OUTPUT_BASE}/file_level/loc_outputs.jsonl"
    fi
else
    echo "⏳ 步骤1（文件级定位）: 尚未开始"
fi

echo ""

# 步骤2结果
if [ -f "${OUTPUT_BASE}/related_elements/loc_outputs.jsonl" ]; then
    RELATED_COUNT=$(wc -l < "${OUTPUT_BASE}/related_elements/loc_outputs.jsonl" | tr -d ' ')
    PERCENTAGE=$(python3 -c "print(f'{${RELATED_COUNT} * 100 / ${TOTAL_INSTANCES}:.1f}')" 2>/dev/null || echo "计算中")
    echo "🔄 步骤2（相关元素定位）: 已完成 $RELATED_COUNT / $TOTAL_INSTANCES 个实例 ($PERCENTAGE%)"
    echo "   📁 输出文件: ${OUTPUT_BASE}/related_elements/loc_outputs.jsonl"
    
    # 显示最新处理的实例ID
    if [ "$RELATED_COUNT" -gt 0 ]; then
        echo ""
        echo "   最新处理的5个实例ID:"
        tail -5 "${OUTPUT_BASE}/related_elements/loc_outputs.jsonl" 2>/dev/null | \
            python3 -c "
import sys, json
for line in sys.stdin:
    try:
        data = json.loads(line.strip())
        instance_id = data.get('instance_id', 'N/A')
        found_files = len(data.get('found_files', []))
        found_modules = len(data.get('found_modules', []))
        print(f'      - {instance_id} (文件: {found_files}, 模块: {found_modules})')
    except:
        pass
" 2>/dev/null || echo "      (解析中...)"
    fi
else
    echo "⏳ 步骤2（相关元素定位）: 尚未开始"
fi

echo ""

# 步骤3结果
if [ -f "${OUTPUT_BASE}/edit_location_samples/loc_outputs.jsonl" ]; then
    EDIT_COUNT=$(wc -l < "${OUTPUT_BASE}/edit_location_samples/loc_outputs.jsonl" | tr -d ' ')
    PERCENTAGE=$(python3 -c "print(f'{${EDIT_COUNT} * 100 / ${TOTAL_INSTANCES}:.1f}')" 2>/dev/null || echo "计算中")
    echo "🔄 步骤3（编辑位置定位）: 已完成 $EDIT_COUNT / $TOTAL_INSTANCES 个实例 ($PERCENTAGE%)"
    echo "   📁 输出文件: ${OUTPUT_BASE}/edit_location_samples/loc_outputs.jsonl"
else
    echo "⏳ 步骤3（编辑位置定位）: 尚未开始"
fi

echo ""
echo "=========================================="
echo "📁 所有结果文件位置:"
echo "=========================================="
echo "  - 步骤1: ${OUTPUT_BASE}/file_level/loc_outputs.jsonl"
echo "  - 步骤2: ${OUTPUT_BASE}/related_elements/loc_outputs.jsonl"
echo "  - 步骤3: ${OUTPUT_BASE}/edit_location_samples/loc_outputs.jsonl"
echo "  - 评估结果: ${OUTPUT_BASE}/evaluation_results.json"
echo ""
echo "💡 提示: 可以使用以下命令查看具体结果:"
echo "  - 查看步骤1结果: head -1 ${OUTPUT_BASE}/file_level/loc_outputs.jsonl | python3 -m json.tool"
echo "  - 查看步骤2结果: head -1 ${OUTPUT_BASE}/related_elements/loc_outputs.jsonl | python3 -m json.tool"
echo "  - 查看步骤3结果: head -1 ${OUTPUT_BASE}/edit_location_samples/loc_outputs.jsonl | python3 -m json.tool"
