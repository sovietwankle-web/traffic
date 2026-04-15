#!/bin/bash
# ================================================================
# 智能网联汽车运行行为数据监测大模型系统 - 4090全自动运行脚本
# 适配: NVIDIA RTX 4090 (24GB VRAM)
# 预计总耗时: 6-8小时
#
# 使用方法:
#   nohup bash run_4090.sh > run.log 2>&1 &
#   # 或
#   bash run_4090.sh
# ================================================================

set -e
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

LOG_FILE="run_4090.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "  智能网联汽车运行行为数据监测大模型系统"
echo "  NVIDIA RTX 4090 全自动运行"
echo "  开始时间: $(date)"
echo "============================================================"

# Step 0: 检查GPU
echo ""
echo "--- Step 0: 环境检查 ---"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory/1024**3:.1f}GB') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('No GPU!')"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: No GPU detected!"
    echo "Install CUDA PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

# Step 1: 安装依赖
echo ""
echo "--- Step 1: 安装依赖 ---"
pip install -r requirements.txt -q 2>&1 | tail -3

# Step 2: 生成完整数据集 (30 runs, 900s each)
echo ""
echo "--- Step 2: 生成仿真数据集 ---"
echo "开始时间: $(date)"
python data/generate_dataset.py --runs 30 --duration 900 --output data/dataset
echo "完成时间: $(date)"

# Step 3: 转换为LLM训练格式
echo ""
echo "--- Step 3: 数据格式转换 ---"
python data/prepare_data.py

# Step 4: 评估基线
echo ""
echo "--- Step 4: 评估基线 ---"
python main.py baselines

# Step 5: LoRA微调 1.5B
echo ""
echo "--- Step 5: LoRA微调 Qwen2.5-1.5B (3 epochs) ---"
echo "开始时间: $(date)"
python model/train.py --epochs 3 --batch-size 8 --lora-r 16
echo "完成时间: $(date)"

# Step 6: 评估1.5B模型
echo ""
echo "--- Step 6: 评估1.5B模型 ---"
python model/inference.py

# Step 7: 检查是否达标，如果<90%则尝试7B
echo ""
echo "--- Step 7: 检查目标达成情况 ---"
ACC=$(python -c "
import json
try:
    with open('data/dataset/evaluation_results.json') as f:
        r = json.load(f)
    best = 0
    for k, v in r.items():
        if isinstance(v, dict) and 'accuracy' in v:
            if 'llm' in k or 'adapter' in k:
                best = max(best, v['accuracy'])
    print(f'{best:.4f}')
    # find best LLM result
    best = 0
    for k, v in r.items():
        if isinstance(v, dict) and 'accuracy' in v:
            if 'llm' in k or 'adapter' in k:
                best = max(best, v['accuracy'])
    print(f'{best:.4f}')
except:
    print('0')
" 2>/dev/null || echo "0")

echo "Best LLM accuracy: $ACC"

if python -c "exit(0 if float('$ACC' or '0') < 0.90 else 1)" 2>/dev/null; then
    echo "1.5B未达90%，开始训练Qwen2.5-7B + QLoRA..."
    echo "开始时间: $(date)"
    python model/train_7b.py --epochs 2
    echo "完成时间: $(date)"

    echo ""
    echo "--- Step 8: 评估7B模型 ---"
    python model/inference.py
else
    echo "1.5B已达到90%以上目标！跳过7B训练。"
fi

# Step 9: 可视化
echo ""
echo "--- Step 9: 生成可视化 ---"
python visualization/visualize.py

# Final
echo ""
echo "============================================================"
echo "  全部完成！"
echo "  结束时间: $(date)"
echo "============================================================"

# 打印最终结果
echo ""
echo "=== 最终结果 ==="
python -c "
import json
with open('data/dataset/evaluation_results.json') as f:
    r = json.load(f)
print(f\"{'Method':<30} {'Accuracy':>10} {'F1':>10}\")
print('-' * 52)
for k, v in r.items():
    if isinstance(v, dict) and 'accuracy' in v:
        label = k.replace('_', ' ').title()
        print(f'{label:<30} {v[\"accuracy\"]:>9.4f} {v.get(\"f1\", 0):>9.4f}')

best = max(v.get('accuracy', 0) for v in r.values() if isinstance(v, dict) and 'accuracy' in v)
print(f'\nBest accuracy: {best:.4f}')
print(f'Target (>=90%): {\"ACHIEVED\" if best >= 0.90 else \"NOT MET\"}')
" 2>/dev/null

echo ""
echo "输出文件:"
echo "  数据集:       data/dataset/"
echo "  1.5B模型:    model/final_adapter/"
echo "  7B模型:      model/final_adapter_7b/"
echo "  可视化:       visualization/"
echo "  评估结果:     data/dataset/evaluation_results.json"
echo "  完整日志:     $LOG_FILE"
