"""
主入口：一键运行完整流程（仿真→生成→训练→评估）
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_simulation(args):
    """Phase 1-3: 运行仿真生成数据集"""
    print("\n" + "=" * 60)
    print("  Phase 1-3: Traffic Simulation & Dataset Generation")
    print("=" * 60)

    from data.generate_dataset import generate_full_dataset
    generate_full_dataset(
        output_dir=args.data_dir,
        runs_per_scenario=args.runs,
        duration=args.duration,
    )


def run_prepare(args):
    """Phase 4: 数据格式转换"""
    print("\n" + "=" * 60)
    print("  Phase 4: Data Preparation for LLM Training")
    print("=" * 60)

    from data.prepare_data import prepare_llm_dataset
    prepare_llm_dataset(
        input_dir=args.data_dir,
        output_dir=args.data_dir,
    )


def run_train(args):
    """Phase 5: LoRA微调"""
    print("\n" + "=" * 60)
    print("  Phase 5: LoRA Fine-tuning")
    print("=" * 60)

    from model.train import train
    train()


def run_evaluate(args):
    """Phase 6: 评估"""
    print("\n" + "=" * 60)
    print("  Phase 6: Evaluation")
    print("=" * 60)

    from model.inference import run_full_evaluation
    run_full_evaluation(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
    )


def run_visualize(args):
    """生成可视化"""
    from visualization.visualize import generate_all_visualizations
    generate_all_visualizations(data_dir=args.data_dir)


def main():
    parser = argparse.ArgumentParser(
        description="智能网联汽车运行行为数据监测大模型系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py simulate              # 运行仿真生成数据集
  python main.py prepare               # 转换数据为LLM格式
  python main.py train                  # LoRA微调
  python main.py evaluate               # 评估模型
  python main.py visualize              # 生成可视化
  python main.py all                    # 完整流程
  python main.py baselines              # 仅评估基线(不需要GPU模型)
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 仿真
    sim_parser = subparsers.add_parser("simulate", help="Run traffic simulation")
    sim_parser.add_argument("--runs", type=int, default=30, help="Runs per scenario")
    sim_parser.add_argument("--duration", type=float, default=900.0, help="Duration per run (s)")
    sim_parser.add_argument("--data-dir", type=str, default="data/dataset")

    # 数据准备
    prep_parser = subparsers.add_parser("prepare", help="Prepare LLM training data")
    prep_parser.add_argument("--data-dir", type=str, default="data/dataset")

    # 训练
    train_parser = subparsers.add_parser("train", help="LoRA fine-tuning")
    train_parser.add_argument("--4bit", action="store_true")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=4)

    # 评估
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    eval_parser.add_argument("--data-dir", type=str, default="data/dataset")
    eval_parser.add_argument("--model-dir", type=str, default="model/final_adapter")

    # 可视化
    vis_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    vis_parser.add_argument("--data-dir", type=str, default="data/dataset")

    # 完整流程
    all_parser = subparsers.add_parser("all", help="Run complete pipeline")
    all_parser.add_argument("--runs", type=int, default=30)
    all_parser.add_argument("--duration", type=float, default=900.0)
    all_parser.add_argument("--data-dir", type=str, default="data/dataset")
    all_parser.add_argument("--model-dir", type=str, default="model/final_adapter")

    # 仅基线
    bl_parser = subparsers.add_parser("baselines", help="Evaluate baselines only")
    bl_parser.add_argument("--data-dir", type=str, default="data/dataset")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "simulate":
        run_simulation(args)
        run_prepare(args)
    elif args.command == "prepare":
        run_prepare(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "visualize":
        run_visualize(args)
    elif args.command == "baselines":
        # 仅评估基线
        from model.inference import evaluate_rule_baseline, evaluate_rf_baseline
        test_file = os.path.join(args.data_dir, "test.json")
        train_file = os.path.join(args.data_dir, "train.json")
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        with open(train_file, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        evaluate_rule_baseline(test_data)
        evaluate_rf_baseline(test_data, train_data)
    elif args.command == "all":
        run_simulation(args)
        run_prepare(args)
        run_train(args)
        run_evaluate(args)
        run_visualize(args)


if __name__ == "__main__":
    main()
