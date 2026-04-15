"""
一键全自动运行脚本：适配NVIDIA RTX 4090 (24GB VRAM)
完成从仿真数据生成到模型训练评估的全部流程。
预计总耗时：6-8小时

使用方法：
  python run_all_4090.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_step(cmd, desc, timeout=None):
    """运行一个步骤"""
    log(f"=== START: {desc} ===")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        cmd, shell=True, env=env, timeout=timeout,
        capture_output=False, text=True,
    )
    dt = time.time() - t0
    if result.returncode != 0:
        log(f"!!! FAILED: {desc} (returncode={result.returncode}, time={dt:.0f}s)")
        return False
    log(f"=== DONE: {desc} (time={dt:.0f}s) ===")
    return True


def check_gpu():
    """检查GPU环境"""
    import torch
    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            log(f"GPU {i}: {props.name}, {props.total_memory/1024**3:.1f}GB VRAM")
        return True
    else:
        log("WARNING: No GPU detected! Training will be very slow on CPU.")
        return False


def main():
    start_time = time.time()
    log("=" * 70)
    log("  智能网联汽车运行行为数据监测大模型系统 - 全自动运行")
    log("  适配: NVIDIA RTX 4090 (24GB VRAM)")
    log("=" * 70)

    # Step 0: 检查环境
    log("\n--- Step 0: 环境检查 ---")
    has_gpu = check_gpu()
    if not has_gpu:
        log("ERROR: No GPU found. This script requires a GPU.")
        log("Please install CUDA-enabled PyTorch:")
        log("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    # Step 1: 安装依赖（如需要）
    log("\n--- Step 1: 检查依赖 ---")
    try:
        import transformers, peft, trl, datasets, sklearn, matplotlib
        log("All dependencies installed.")
    except ImportError as e:
        log(f"Missing dependency: {e}, installing...")
        run_step(
            "pip install -r requirements.txt -q",
            "Install dependencies",
            timeout=600,
        )

    # Step 2: 生成完整数据集
    log("\n--- Step 2: 生成仿真数据集 (30 runs * 900s) ---")
    step2_ok = run_step(
        "python data/generate_dataset.py --runs 30 --duration 900 --output data/dataset",
        "Generate full dataset",
        timeout=14400,  # 4 hours max
    )
    if not step2_ok:
        log("Dataset generation failed, trying with fewer runs...")
        run_step(
            "python data/generate_dataset.py --runs 20 --duration 600 --output data/dataset",
            "Generate reduced dataset",
            timeout=7200,
        )

    # Step 3: 转换为LLM训练格式
    log("\n--- Step 3: 转换数据为LLM训练格式 ---")
    run_step(
        "python data/prepare_data.py",
        "Prepare LLM training data",
        timeout=300,
    )

    # Step 4: 评估基线（训练前先跑，不依赖GPU模型）
    log("\n--- Step 4: 评估基线方法 ---")
    run_step(
        "python main.py baselines",
        "Evaluate rule-based and RF baselines",
        timeout=300,
    )

    # Step 5: LoRA微调 - Qwen2.5-1.5B
    log("\n--- Step 5: LoRA微调 (Qwen2.5-1.5B-Instruct, 3 epochs) ---")
    step5_ok = run_step(
        "python model/train.py --epochs 3 --batch-size 8 --lora-r 16",
        "LoRA fine-tuning 1.5B model",
        timeout=14400,  # 4 hours max
    )

    # Step 6: 评估1.5B模型
    log("\n--- Step 6: 评估1.5B微调模型 ---")
    step6_ok = False
    if step5_ok and os.path.exists("model/final_adapter/adapter_config.json"):
        step6_ok = run_step(
            "python model/inference.py",
            "Evaluate 1.5B fine-tuned model",
            timeout=7200,  # 2 hours max for inference
        )

    # Step 7: 尝试更大的模型（如果1.5B不够好且有时间）
    if step6_ok:
        # 读取1.5B评估结果
        results_file = "data/dataset/evaluation_results.json"
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results = json.load(f)
            llm_acc = results.get("llm", {}).get("accuracy", 0)
            log(f"1.5B model accuracy: {llm_acc:.4f}")

            if llm_acc < 0.90:
                log("\n--- Step 7: 1.5B未达90%，尝试Qwen2.5-7B + QLoRA ---")
                log("Downloading and training 7B model with 4-bit quantization...")
                step7_ok = run_step(
                    "python model/train_7b.py --epochs 2",
                    "QLoRA fine-tuning 7B model",
                    timeout=14400,
                )
                if step7_ok and os.path.exists("model/final_adapter_7b/adapter_config.json"):
                    run_step(
                        "python model/inference.py --model-dir model/final_adapter_7b",
                        "Evaluate 7B model",
                        timeout=7200,
                    )
            else:
                log("1.5B model already achieved >=90% accuracy! Skipping 7B training.")
        else:
            log("No evaluation results found, skipping 7B training attempt.")

    # Step 8: 生成可视化
    log("\n--- Step 8: 生成可视化图表 ---")
    run_step(
        "python visualization/visualize.py",
        "Generate visualizations",
        timeout=120,
    )

    # Final summary
    total_time = time.time() - start_time
    log("\n" + "=" * 70)
    log("  ALL DONE!")
    log(f"  Total time: {total_time/3600:.1f} hours")
    log("=" * 70)

    # Print results summary
    if os.path.exists("data/dataset/evaluation_results.json"):
        with open("data/dataset/evaluation_results.json", "r") as f:
            results = json.load(f)
        log("\nFinal Results:")
        for method, r in results.items():
            if isinstance(r, dict) and "accuracy" in r:
                log(f"  {method}: accuracy={r['accuracy']:.4f}, F1={r.get('f1', 0):.4f}")

    log("\nOutput files:")
    log("  Dataset:       data/dataset/")
    log("  Model:         model/final_adapter/")
    log("  Visualizations: visualization/")
    log("  Results:       data/dataset/evaluation_results.json")


if __name__ == "__main__":
    main()
