"""
可视化工具：绘制车辆行为分析图表。
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_speed_by_driver_type(journeys: List[dict], output_path: str = "visualization/speed_by_type.png"):
    """按驾驶员类型绘制速度分布箱线图"""
    type_speeds = defaultdict(list)
    for j in journeys:
        speeds = [o["speed_kmh"] for o in j["observations"]]
        if speeds:
            type_speeds[j["driver_type"]].append(np.mean(speeds))

    fig, ax = plt.subplots(figsize=(10, 6))
    types = sorted(type_speeds.keys())
    data = [type_speeds[t] for t in types]
    labels_map = {
        "rule_follower": "遵规AI",
        "aggressive_ai": "激进AI",
        "normal_human": "普通人类",
        "rule_breaker": "违规追速",
        "fatigued": "疲劳驾驶",
    }
    labels = [labels_map.get(t, t) for t in types]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ['#4CAF50', '#2196F3', '#9E9E9E', '#F44336', '#FF9800']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('平均速度 (km/h)')
    ax.set_title('不同驾驶员类型的平均速度分布')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_speed_deviation_by_type(journeys: List[dict], output_path: str = "visualization/deviation_by_type.png"):
    """按驾驶员类型绘制速度偏差分布"""
    type_devs = defaultdict(list)
    for j in journeys:
        devs = [o.get("speed_deviation_kmh", 0) for o in j["observations"]]
        if devs:
            type_devs[j["driver_type"]].append(np.mean(devs))

    fig, ax = plt.subplots(figsize=(10, 6))
    types = sorted(type_devs.keys())
    data = [type_devs[t] for t in types]
    labels_map = {
        "rule_follower": "遵规AI",
        "aggressive_ai": "激进AI",
        "normal_human": "普通人类",
        "rule_breaker": "违规追速",
        "fatigued": "疲劳驾驶",
    }
    labels = [labels_map.get(t, t) for t in types]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ['#4CAF50', '#2196F3', '#9E9E9E', '#F44336', '#FF9800']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_ylabel('平均速度偏差 (km/h)')
    ax.set_title('不同驾驶员类型与车流平均的速度偏差')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_confusion_matrix(predictions: List[str], labels: List[str],
                          output_path: str = "visualization/confusion_matrix.png"):
    """绘制混淆矩阵"""
    # 2x2矩阵
    tp = sum(1 for p, l in zip(predictions, labels) if p == "abnormal" and l == "abnormal")
    fp = sum(1 for p, l in zip(predictions, labels) if p == "abnormal" and l == "normal")
    fn = sum(1 for p, l in zip(predictions, labels) if p == "normal" and l == "abnormal")
    tn = sum(1 for p, l in zip(predictions, labels) if p == "normal" and l == "normal")

    matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='Blues')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                   fontsize=20, color='white' if matrix[i, j] > matrix.max()/2 else 'black')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Abnormal'])
    ax.set_yticklabels(['Normal', 'Abnormal'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_comparison(results: dict, output_path: str = "visualization/model_comparison.png"):
    """绘制模型对比柱状图"""
    methods = []
    accuracies = []
    f1s = []

    if "rule_baseline" in results:
        methods.append("Rule-Based")
        accuracies.append(results["rule_baseline"]["accuracy"])
        f1s.append(results["rule_baseline"]["f1"])
    if "rf_baseline" in results:
        methods.append("Random Forest")
        accuracies.append(results["rf_baseline"]["accuracy"])
        f1s.append(results["rf_baseline"]["f1"])
    if "llm" in results:
        methods.append("LLM (LoRA)")
        accuracies.append(results["llm"]["accuracy"])
        f1s.append(results["llm"]["f1"])

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#2196F3')
    bars2 = ax.bar(x + width/2, f1s, width, label='F1 Score', color='#FF9800')

    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% Target')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.annotate(f'{bar.get_height():.2%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.2%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_visualizations(data_dir: str = "data/dataset"):
    """生成所有可视化图表"""
    os.makedirs("visualization", exist_ok=True)

    # 加载训练数据用于分布图
    train_file = os.path.join(data_dir, "train.json")
    if os.path.exists(train_file):
        with open(train_file, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        plot_speed_by_driver_type(train_data)
        plot_speed_deviation_by_type(train_data)

    # 加载评估结果
    results_file = os.path.join(data_dir, "evaluation_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        plot_model_comparison(results)

    print("\nVisualization complete!")


if __name__ == "__main__":
    generate_all_visualizations()
