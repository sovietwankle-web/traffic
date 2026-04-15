"""
将仿真生成的旅程数据转换为LLM训练格式（JSONL chat格式）。
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.prompts import journey_to_training_sample


def prepare_llm_dataset(
    input_dir: str = "data/dataset",
    output_dir: str = "data/dataset",
):
    """转换train/val/test JSON为JSONL格式"""
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        input_file = os.path.join(input_dir, f"{split}.json")
        output_file = os.path.join(output_dir, f"{split}.jsonl")

        if not os.path.exists(input_file):
            print(f"Skipping {split}: {input_file} not found")
            continue

        with open(input_file, "r", encoding="utf-8") as f:
            journeys = json.load(f)

        print(f"\nProcessing {split}: {len(journeys)} journeys")

        samples = []
        stats = defaultdict(int)
        token_counts = []

        for journey in journeys:
            sample = journey_to_training_sample(journey)
            samples.append(sample)
            stats[journey["label"]] += 1

            # 粗略估计token数（中文约1.5字/token）
            total_chars = sum(len(m["content"]) for m in sample["messages"])
            estimated_tokens = int(total_chars * 0.7)
            token_counts.append(estimated_tokens)

        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0
        print(f"  Saved to {output_file}")
        print(f"  Labels: {dict(stats)}")
        print(f"  Avg estimated tokens: {avg_tokens:.0f}, Max: {max_tokens}")

    # 打印一个示例
    print("\n" + "=" * 60)
    print("Sample training example:")
    print("=" * 60)

    sample_file = os.path.join(output_dir, "train.jsonl")
    if os.path.exists(sample_file):
        with open(sample_file, "r", encoding="utf-8") as f:
            sample = json.loads(f.readline())
        for msg in sample["messages"]:
            role = msg["role"]
            content = msg["content"]
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"\n[{role}]")
            print(content)


if __name__ == "__main__":
    prepare_llm_dataset()
