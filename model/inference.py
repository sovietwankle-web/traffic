"""
推理与评估脚本：加载微调后的模型，在测试集上评估异常检测准确率。
同时实现规则基线和随机森林基线进行对比。
适配NVIDIA RTX 4090。
"""

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ============================================================
# 通用评估工具
# ============================================================

def compute_metrics(predictions: List[str], labels: List[str]) -> dict:
    tp = sum(1 for p, l in zip(predictions, labels) if p == "abnormal" and l == "abnormal")
    fp = sum(1 for p, l in zip(predictions, labels) if p == "abnormal" and l == "normal")
    fn = sum(1 for p, l in zip(predictions, labels) if p == "normal" and l == "abnormal")
    tn = sum(1 for p, l in zip(predictions, labels) if p == "normal" and l == "normal")

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": total,
    }


def print_evaluation_report(results: dict, name: str):
    print(f"\n{'='*60}")
    print(f"  Evaluation Report: {name}")
    print(f"{'='*60}")
    print(f"  Total samples: {results['total']}")
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1:        {results['f1']:.4f}")
    print(f"  TP={results['tp']} FP={results['fp']} FN={results['fn']} TN={results['tn']}")
    print(f"{'='*60}")


# ============================================================
# 基线1：规则阈值法（评分制）
# ============================================================

def rule_based_classify(journey: dict) -> str:
    obs_list = journey["observations"]
    if not obs_list:
        return "normal"

    speeds = [o["speed_kmh"] for o in obs_list]
    limits = [o["speed_limit_kmh"] for o in obs_list]
    deviations = [o.get("speed_deviation_kmh", 0) for o in obs_list]
    speed_ratios = [s / l if l > 0 else 1.0 for s, l in zip(speeds, limits)]
    n = len(obs_list)

    score = 0.0

    over_speed_ratio = sum(1 for r in speed_ratios if r > 1.1) / n
    if over_speed_ratio > 0.5:
        score += 2.0
    elif over_speed_ratio > 0.2:
        score += 1.0

    if any(r > 1.2 for r in speed_ratios):
        score += 1.5

    avg_deviation = sum(deviations) / n
    if avg_deviation < -10:
        score += 1.5
    elif avg_deviation < -5:
        score += 0.5

    if avg_deviation > 15:
        score += 1.5
    elif avg_deviation > 10:
        score += 0.5

    speed_std = np.std(speeds) if n > 1 else 0
    if speed_std > 15:
        score += 1.5
    elif speed_std > 10:
        score += 0.5

    under_speed_ratio = sum(1 for r in speed_ratios if r < 0.85) / n
    if under_speed_ratio > 0.5:
        score += 1.5

    return "abnormal" if score >= 2.0 else "normal"


def evaluate_rule_baseline(test_data: List[dict]) -> dict:
    predictions = [rule_based_classify(j) for j in test_data]
    labels = [j["label"] for j in test_data]
    results = compute_metrics(predictions, labels)
    print_evaluation_report(results, "Rule-Based Baseline")
    return results


# ============================================================
# 基线2：随机森林
# ============================================================

def extract_features(journey: dict) -> dict:
    obs_list = journey["observations"]
    if not obs_list:
        return {}

    speeds = [o["speed_kmh"] for o in obs_list]
    limits = [o["speed_limit_kmh"] for o in obs_list]
    deviations = [o.get("speed_deviation_kmh", 0) for o in obs_list]
    accels = [o.get("acceleration", 0) for o in obs_list]
    densities = [o.get("traffic_density", 0) for o in obs_list]
    speed_ratios = [s / l if l > 0 else 1.0 for s, l in zip(speeds, limits)]
    n = len(obs_list)
    lanes = [o.get("lane", 0) for o in obs_list]
    lane_changes = sum(1 for i in range(1, len(lanes)) if lanes[i] != lanes[i-1])

    return {
        "n_observations": n,
        "mean_speed": np.mean(speeds),
        "max_speed": max(speeds),
        "min_speed": min(speeds),
        "speed_range": max(speeds) - min(speeds),
        "speed_std": np.std(speeds) if n > 1 else 0,
        "mean_speed_ratio": np.mean(speed_ratios),
        "max_speed_ratio": max(speed_ratios),
        "min_speed_ratio": min(speed_ratios),
        "mean_deviation": np.mean(deviations),
        "max_deviation": max(abs(d) for d in deviations),
        "mean_accel": np.mean(accels),
        "max_abs_accel": max(abs(a) for a in accels),
        "accel_std": np.std(accels) if n > 1 else 0,
        "lane_changes": lane_changes,
        "lane_change_rate": lane_changes / max(n - 1, 1),
        "mean_density": np.mean(densities),
        "over_speed_ratio": sum(1 for r in speed_ratios if r > 1.1) / n,
        "under_speed_ratio": sum(1 for r in speed_ratios if r < 0.85) / n,
    }


def evaluate_rf_baseline(test_data: List[dict], train_data: List[dict]) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    train_features = [extract_features(j) for j in train_data]
    test_features = [extract_features(j) for j in test_data]

    valid_train = [(f, j) for f, j in zip(train_features, train_data) if f]
    valid_test = [(f, j) for f, j in zip(test_features, test_data) if f]

    if not valid_train or not valid_test:
        print("Not enough data for RF baseline")
        return {}

    train_X = np.array([[f[k] for k in sorted(valid_train[0][0].keys())] for f, _ in valid_train])
    train_y = np.array([1 if j["label"] == "abnormal" else 0 for _, j in valid_train])
    test_X = np.array([[f[k] for k in sorted(valid_train[0][0].keys())] for f, _ in valid_test])
    test_y = np.array([1 if j["label"] == "abnormal" else 0 for _, j in valid_test])

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15, class_weight='balanced')
    rf.fit(train_X, train_y)

    pred_y = rf.predict(test_X)
    predictions = ["abnormal" if p == 1 else "normal" for p in pred_y]
    labels = ["abnormal" if l == 1 else "normal" for l in test_y]

    results = compute_metrics(predictions, labels)

    feature_names = sorted(valid_train[0][0].keys())
    importances = rf.feature_importances_
    print_evaluation_report(results, "Random Forest Baseline")
    print("\n  Feature Importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]:
        print(f"    {name}: {imp:.4f}")

    return results


# ============================================================
# LLM推理与评估
# ============================================================

def parse_llm_output(text: str) -> Tuple[str, float, str]:
    json_match = re.search(r'\{[^{}]*"classification"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            classification = result.get("classification", "normal")
            confidence = float(result.get("confidence", 0.5))
            reason = result.get("reason", "")
            return classification, confidence, reason
        except (json.JSONDecodeError, ValueError):
            pass

    text_lower = text.lower()
    if "abnormal" in text_lower:
        return "abnormal", 0.5, text[:100]
    return "normal", 0.5, text[:100]


def detect_model_base(model_dir: str) -> str:
    """自动检测adapter对应的基座模型"""
    config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        # PEFT adapter_config里有base_model_name_or_path
        base = config.get("base_model_name_or_path", "")
        if base:
            return base
    # 回退：根据目录名猜测
    if "7b" in model_dir.lower():
        return "Qwen/Qwen2.5-7B-Instruct"
    return "Qwen/Qwen2.5-1.5B-Instruct"


def find_local_model(model_id):
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_name = model_id.replace("/", "--")
    model_cache = cache_dir / f"models--{model_name}"
    if model_cache.exists():
        snapshots = model_cache / "snapshots"
        if snapshots.exists():
            dirs = list(snapshots.iterdir())
            if dirs:
                return str(dirs[0])
    return model_id


def evaluate_llm(test_data: List[dict], model_dir: str = "model/final_adapter") -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model.prompts import SYSTEM_PROMPT, format_journey_prompt

    base_model_id = detect_model_base(model_dir)
    model_source = find_local_model(base_model_id)
    local_only = model_source != base_model_id
    is_7b = "7b" in base_model_id.lower()

    print(f"\nLoading model: base={base_model_id}, adapter={model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, padding_side="right",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 7B模型需要4-bit量化
    load_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_only,
    )
    if is_7b:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        load_kwargs["torch_dtype"] = torch.bfloat16 if use_bf16 else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    print("Model loaded.")

    predictions = []
    labels = []
    driver_types = []
    confidences = []
    parse_failures = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(test_data, desc="Evaluating LLM")
    except ImportError:
        iterator = test_data

    for i, journey in enumerate(iterator):
        label = journey["label"]
        labels.append(label)
        driver_types.append(journey["driver_type"])

        user_content = format_journey_prompt(journey)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=False,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        classification, confidence, reason = parse_llm_output(response)
        if classification not in ("normal", "abnormal"):
            parse_failures += 1
            classification = "normal"

        predictions.append(classification)
        confidences.append(confidence)

        if (i + 1) % 100 == 0:
            acc_so_far = sum(1 for p, l in zip(predictions, labels) if p == l) / len(predictions)
            print(f"  Progress: {i+1}/{len(test_data)}, accuracy so far: {acc_so_far:.4f}")

    results = compute_metrics(predictions, labels)

    print_evaluation_report(results, f"Fine-tuned LLM ({base_model_id} + LoRA)")
    if parse_failures > 0:
        print(f"  Parse failures: {parse_failures}/{len(test_data)}")

    # 按驾驶员类型分解
    print("\n  Per-driver-type accuracy:")
    for dtype in sorted(set(driver_types)):
        idx = [i for i, d in enumerate(driver_types) if d == dtype]
        dtype_labels = ["abnormal" if dtype in ("rule_breaker", "fatigued") else "normal" for _ in idx]
        dtype_preds = [predictions[i] for i in idx]
        correct = sum(1 for p, l in zip(dtype_preds, dtype_labels) if p == l)
        total = len(idx)
        print(f"    {dtype}: {correct}/{total} = {correct/total:.2%}")

    # 按场景分解
    print("\n  Per-scenario accuracy:")
    scenarios = [j["scenario"] for j in test_data]
    for scenario in sorted(set(scenarios)):
        idx = [i for i, s in enumerate(scenarios) if s == scenario]
        s_preds = [predictions[i] for i in idx]
        s_labels = [labels[i] for i in idx]
        correct = sum(1 for p, l in zip(s_preds, s_labels) if p == l)
        total = len(idx)
        print(f"    {scenario}: {correct}/{total} = {correct/total:.2%}")

    results["parse_failures"] = parse_failures
    results["predictions"] = predictions
    results["confidences"] = confidences
    results["base_model"] = base_model_id
    return results


# ============================================================
# 主评估流程
# ============================================================

def run_full_evaluation(
    data_dir: str = "data/dataset",
    model_dir: str = "model/final_adapter",
):
    test_file = os.path.join(data_dir, "test.json")
    train_file = os.path.join(data_dir, "train.json")

    if not os.path.exists(test_file):
        print(f"Test data not found: {test_file}")
        sys.exit(1)

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print(f"Test data: {len(test_data)} samples")
    label_counts = Counter(j["label"] for j in test_data)
    print(f"  Normal: {label_counts.get('normal', 0)}, Abnormal: {label_counts.get('abnormal', 0)}")

    # 1. 规则基线
    print("\n" + "#" * 60)
    print("# 1. Rule-Based Baseline")
    print("#" * 60)
    rule_results = evaluate_rule_baseline(test_data)

    # 2. 随机森林
    print("\n" + "#" * 60)
    print("# 2. Random Forest Baseline")
    print("#" * 60)
    rf_results = evaluate_rf_baseline(test_data, train_data)

    # 3. LLM
    all_results = {
        "rule_baseline": {k: v for k, v in rule_results.items() if k not in ("predictions",)},
        "rf_baseline": {k: v for k, v in rf_results.items() if k not in ("predictions",)},
    }

    # 检查有哪些adapter可用
    adapter_dirs = [d for d in ["model/final_adapter", "model/final_adapter_7b"]
                    if os.path.exists(os.path.join(d, "adapter_config.json"))]

    for adapter_dir in adapter_dirs:
        model_name = os.path.basename(adapter_dir)
        print(f"\n{'#' * 60}")
        print(f"# 3. Fine-tuned LLM ({model_name})")
        print(f"{'#' * 60}")
        llm_results = evaluate_llm(test_data, adapter_dir)
        llm_key = f"llm_{model_name}" if len(adapter_dirs) > 1 else "llm"
        all_results[llm_key] = {k: v for k, v in llm_results.items()
                                if k not in ("predictions", "confidences")}

    if not adapter_dirs:
        print("\nNo fine-tuned model found. Skipping LLM evaluation.")

    # 对比汇总
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Method':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*70}")
    for name, r in all_results.items():
        if isinstance(r, dict) and "accuracy" in r:
            label = name.replace("_", " ").title()
            print(f"  {label:<30} {r['accuracy']:>9.4f} {r['precision']:>9.4f} {r['recall']:>9.4f} {r['f1']:>9.4f}")
    print("=" * 70)

    # 保存结果
    results_file = os.path.join(data_dir, "evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_file}")

    # 判断是否达到目标
    best_acc = max(r.get("accuracy", 0) for r in all_results.values() if isinstance(r, dict))
    if best_acc >= 0.90:
        print(f"\n  TARGET ACHIEVED: {best_acc:.4f} >= 0.90")
    else:
        print(f"\n  TARGET NOT YET MET: {best_acc:.4f} < 0.90")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate vehicle behavior detection models")
    parser.add_argument("--data-dir", type=str, default="data/dataset")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Specific adapter dir (default: auto-detect all)")
    args = parser.parse_args()

    if args.model_dir:
        # 只评估指定模型
        test_file = os.path.join(args.data_dir, "test.json")
        train_file = os.path.join(args.data_dir, "train.json")
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        with open(train_file, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        evaluate_rule_baseline(test_data)
        evaluate_rf_baseline(test_data, train_data)
        evaluate_llm(test_data, args.model_dir)
    else:
        run_full_evaluation(args.data_dir)
