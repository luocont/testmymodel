#!/bin/bash
# 快速开始脚本 - 完整的评估流程示例

echo "======================================"
echo "  精简版评估框架 - 快速开始"
echo "======================================"
echo ""

# 1. 检查配置
echo "[1/5] 检查配置文件..."
if [ ! -f "config.py" ]; then
    echo "❌ 错误: 请先复制 config.py.template 为 config.py 并填入您的凭据"
    echo "   cp config.py.template config.py"
    echo "   然后编辑 config.py 填入您的 Azure OpenAI 凭据"
    exit 1
fi
echo "✓ 配置文件存在"
echo ""

# 2. 启动 vLLM 服务
echo "[2/5] 确保 vLLM 服务已启动..."
echo "   如果没有启动，请在另一个终端运行:"
echo "   python3 -m vllm.entrypoints.openai.api_server \\"
echo "     --model /path/to/your/model \\"
echo "     --dtype float16 \\"
echo "     --host 0.0.0.0 \\"
echo "     --port 8000"
echo ""
read -p "按 Enter 继续（确认 vLLM 服务已启动）..."
echo ""

# 3. 生成对话数据
echo "[3/5] 生成对话数据..."
python generate/generate_dialogue.py \
  --input dataset/test_data.json \
  --output output/sessions \
  --max_turns 10

if [ $? -ne 0 ]; then
  echo "❌ 生成对话失败"
  exit 1
fi
echo "✓ 对话数据已生成到 output/sessions/"
echo ""

# 4. 运行评估
echo "[4/5] 运行评估..."

echo "  → 运行多样性评估..."
python evaluate/diversity.py \
  --input output/sessions \
  --output results/diversity.json

echo "  → 运行 CTRS 评估..."
python evaluate/ctrs.py \
  --input output/sessions \
  --output results/ctrs \
  --max_iter 1

echo "  → 运行 WAI 评估..."
python evaluate/wai.py \
  --input output/sessions \
  --output results/wai \
  --max_iter 1

echo "  → 运行 PANAS 评估..."
python evaluate/panas.py \
  --input output/sessions \
  --dataset dataset/test_data.json \
  --output results/panas \
  --max_iter 1

echo ""

# 5. 显示结果
echo "[5/5] 评估完成！"
echo ""
echo "======================================"
echo "  结果文件位置:"
echo "======================================"
echo "  多样性: results/diversity.json"
echo "  CTRS:   results/ctrs/ctrs_results.json"
echo "  WAI:    results/wai/wai_results.json"
echo "  PANAS:  results/panas/panas_results.json"
echo ""
echo "======================================"
echo "  查看结果:"
echo "======================================"
echo "  cat results/diversity.json | grep -A 5 average"
echo "  cat results/ctrs/ctrs_results.json | grep -A 10 average"
echo "  cat results/wai/wai_results.json | grep -A 5 average"
echo "  cat results/panas/panas_results.json | grep -A 5 average"
echo ""
