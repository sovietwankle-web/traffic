# 智能网联汽车运行行为数据监测大模型系统

## 项目概述

本项目构建了一个完整的智能网联汽车安全监测系统，包含：

1. **交通仿真引擎** - 基于IDM跟驰模型+MOBIL换道模型的tick-based仿真器
2. **5种驾驶员行为模型** - 遵规AI、激进AI、普通人类、违规追速、疲劳驾驶
3. **7个交通场景** - 连续路口、高速出入口、井字胡同、立交桥、环岛、隧道、事故瓶颈
4. **监测点数据采集** - 模拟有限监测点的稀疏观测
5. **LLM微调** - 基于Qwen2.5-1.5B-Instruct的LoRA微调，用于异常行为检测
6. **基线对比** - 规则阈值法 + 随机森林

## 快速开始

### 环境要求
- Python 3.10+
- PyTorch 2.1+
- GPU (>=8GB显存) 或 CPU (训练较慢)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行仿真生成数据
```bash
# 完整数据集生成（30次运行，每次900s）
python main.py simulate --runs 30 --duration 900

# 快速测试（5次运行，每次600s）
python main.py simulate --runs 5 --duration 600
```

### 转换为LLM训练格式
```bash
python main.py prepare
```

### LoRA微调训练
```bash
# GPU训练
python main.py train --epochs 3

# CPU训练（自动减少数据量）
python main.py train --epochs 1

# 4-bit量化训练（低显存）
python main.py train --4bit --epochs 3
```

### 评估
```bash
# 仅评估基线（不需要GPU）
python main.py baselines

# 完整评估（包含LLM）
python main.py evaluate

# 可视化
python main.py visualize
```

### 一键运行全部流程
```bash
python main.py all --runs 30 --duration 900
```

## 项目结构

```
taffic/
├── simulation/
│   ├── road_network.py      # 道路网络（有向图）
│   ├── vehicle.py           # 车辆状态 + IDM跟驰 + MOBIL换道
│   ├── engine.py            # 仿真引擎（tick-based, dt=0.1s）
│   ├── monitor.py           # 监测点数据采集
│   └── drivers/             # 5种驾驶员参数
├── scenarios/               # 7个交通场景
├── data/
│   ├── generate_dataset.py  # 批量仿真+数据集划分
│   ├── prepare_data.py      # 转换为JSONL训练格式
│   └── dataset/             # 生成的数据文件
├── model/
│   ├── prompts.py           # 提示词模板
│   ├── train.py             # LoRA微调脚本
│   ├── inference.py          # 推理+评估+基线对比
│   └── checkpoints/          # 模型检查点
├── visualization/
│   └── visualize.py          # matplotlib可视化
├── main.py                  # 主入口
└── requirements.txt
```

## 驾驶员类型

| 类型 | 期望速度系数 | 反应延迟 | 特殊行为 | 标签 |
|---|---|---|---|---|
| 遵规AI | 0.95-1.0 | 0 | 完美遵守规则 | 正常 |
| 激进AI | 1.0-1.05 | 0 | 频繁换道但仍合法 | 正常 |
| 普通人类 | 0.9-1.05 | 2-5 ticks | 正常有波动 | 正常 |
| 违规追速 | 1.15-1.35 | 1 | 超速+闯红灯+反侦察 | **异常** |
| 疲劳驾驶 | 0.75-0.9 | 8-15 ticks | 低速+漂移+微睡眠 | **异常** |

## 基线结果（当前）

| 方法 | 准确率 | 精确率 | 召回率 | F1 |
|---|---|---|---|---|
| 规则阈值 | 57.8% | 0.41 | 0.69 | 0.51 |
| 随机森林 | 87.2% | 0.91 | 0.67 | 0.77 |

目标：微调LLM达到 >=90% 准确率