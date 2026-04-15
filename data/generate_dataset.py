"""
批量运行仿真，生成车辆行为数据集。
每个场景运行多次（不同随机种子），收集监测点观测数据，
按车辆分组，划分训练/验证/测试集。
"""

import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios import ALL_SCENARIOS


def run_scenario(scenario_name: str, builder, seed: int, duration: float = 900.0) -> List[dict]:
    """运行单个场景，返回车辆旅程列表"""
    engine = builder(seed=seed)
    engine.run(duration=duration, progress_interval=300.0)

    enriched = engine.monitor_system.get_enriched_observations_by_vehicle()
    journeys = []

    for vid, obs_list in enriched.items():
        if len(obs_list) < 2:
            continue  # 过少观测点无法判断

        driver_type = obs_list[0]["driver_type"]
        label = "abnormal" if driver_type in ("rule_breaker", "fatigued") else "normal"

        journey = {
            "vehicle_id": vid,
            "driver_type": driver_type,
            "label": label,
            "scenario": scenario_name,
            "seed": seed,
            "observations": [
                {
                    "monitor": o["monitor_name"],
                    "timestamp": round(o["timestamp"], 2),
                    "speed_kmh": round(o["speed_kmh"], 1),
                    "speed_limit_kmh": round(o["speed_limit_kmh"], 1),
                    "lane": o["lane"],
                    "avg_speed_kmh": round(o["avg_speed_kmh"], 1),
                    "traffic_density": round(o["traffic_density"], 1),
                    "speed_deviation_kmh": round(o["speed_deviation_kmh"], 1),
                    "acceleration": round(o["acceleration"], 2),
                }
                for o in obs_list
            ],
        }
        journeys.append(journey)

    return journeys


def generate_full_dataset(
    output_dir: str = "data/dataset",
    runs_per_scenario: int = 30,
    duration: float = 900.0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """生成完整数据集"""
    os.makedirs(output_dir, exist_ok=True)

    all_journeys = []
    stats = defaultdict(lambda: defaultdict(int))

    for scenario_name, builder in ALL_SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*60}")

        scenario_journeys = []
        for run_idx in range(runs_per_scenario):
            seed = run_idx * 1000 + hash(scenario_name) % 1000
            print(f"  Run {run_idx+1}/{runs_per_scenario} (seed={seed})...")
            journeys = run_scenario(scenario_name, builder, seed, duration)
            scenario_journeys.extend(journeys)

            for j in journeys:
                stats[scenario_name][j["driver_type"]] += 1

        all_journeys.extend(scenario_journeys)

        # 保存场景数据
        scenario_file = os.path.join(output_dir, f"{scenario_name}_journeys.json")
        with open(scenario_file, "w", encoding="utf-8") as f:
            json.dump(scenario_journeys, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(scenario_journeys)} journeys to {scenario_file}")

    # 打印统计
    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total journeys: {len(all_journeys)}")

    total_by_type = defaultdict(int)
    for scenario, type_counts in stats.items():
        print(f"\n  {scenario}:")
        for dtype, count in sorted(type_counts.items()):
            print(f"    {dtype}: {count}")
            total_by_type[dtype] += count

    print(f"\n  Overall:")
    for dtype, count in sorted(total_by_type.items()):
        label = "abnormal" if dtype in ("rule_breaker", "fatigued") else "normal"
        print(f"    {dtype} ({label}): {count}")

    normal = sum(c for d, c in total_by_type.items() if d not in ("rule_breaker", "fatigued"))
    abnormal = sum(c for d, c in total_by_type.items() if d in ("rule_breaker", "fatigued"))
    print(f"\n  Normal: {normal}, Abnormal: {abnormal}, Ratio: {abnormal/(normal+abnormal)*100:.1f}%")

    # 划分数据集
    random.seed(42)
    random.shuffle(all_journeys)

    n = len(all_journeys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_data = all_journeys[:n_train]
    val_data = all_journeys[n_train:n_train + n_val]
    test_data = all_journeys[n_train + n_val:]

    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        split_file = os.path.join(output_dir, f"{split_name}.json")
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)

        # 统计
        label_counts = defaultdict(int)
        for j in split_data:
            label_counts[j["label"]] += 1
        print(f"\n  {split_name}: {len(split_data)} journeys (normal={label_counts['normal']}, abnormal={label_counts['abnormal']})")

    print(f"\nDataset generation complete! Files saved to {output_dir}/")
    return all_journeys


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate traffic simulation dataset")
    parser.add_argument("--runs", type=int, default=30, help="Runs per scenario")
    parser.add_argument("--duration", type=float, default=900.0, help="Simulation duration (seconds)")
    parser.add_argument("--output", type=str, default="data/dataset", help="Output directory")
    args = parser.parse_args()

    generate_full_dataset(
        output_dir=args.output,
        runs_per_scenario=args.runs,
        duration=args.duration,
    )
