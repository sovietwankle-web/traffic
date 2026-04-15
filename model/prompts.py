"""
提示词模板设计：将监测点观测数据转换为LLM可理解的文本格式。
"""

import json
from typing import List

# 场景描述映射
SCENARIO_DESCRIPTIONS = {
    "consecutive_intersections": "两个连续信号灯十字路口，主路限速50km/h，支路限速40km/h",
    "highway_ramps": "高速公路三出入口路段，主路限速120km/h，匝道限速60km/h",
    "hutong_grid": "井字形双向单车道胡同路段，限速30km/h",
    "interchange": "十字立交桥，主路限速100km/h，环形匝道限速40km/h",
    "roundabout": "都市六出口环岛，环内限速30km/h，进出道限速50km/h",
    "tunnel": "长隧道路段(1500m)，隧道限速80km/h，进出口限速60km/h，隧道内禁止换道",
    "bottleneck": "事故瓶颈路段，正常段4车道限速100km/h，施工段2车道限速60km/h",
}

SYSTEM_PROMPT = """你是智能网联汽车安全监测专用大模型。你的任务是分析车辆通过多个道路监测点时的行为数据，判断该车辆的驾驶行为是否异常。

异常行为包括但不限于：
- 持续大幅超速（远高于限速和平均车流速度）
- 频繁急加速/急减速
- 异常低速行驶（远低于车流平均速度）
- 速度波动异常大（疲劳驾驶特征）
- 违反交通规则的行为模式

请基于监测点数据序列进行综合分析，输出JSON格式的判断结果：
- "classification": "normal"（正常）或 "abnormal"（异常）
- "confidence": 0.0到1.0之间的置信度
- "reason": 简要分析理由"""


def format_observation(obs: dict, obs_index: int) -> str:
    """格式化单个监测点观测"""
    time_min = int(obs["timestamp"] // 60)
    time_sec = obs["timestamp"] % 60

    parts = [
        f"[监测点: {obs['monitor']}]",
        f"时间: {time_min:02d}:{time_sec:05.2f}",
        f"车速: {obs['speed_kmh']:.1f}km/h",
        f"限速: {obs['speed_limit_kmh']:.0f}km/h",
    ]

    if "lane" in obs:
        parts.append(f"车道: {obs['lane']+1}")

    parts.append(f"路段平均车速: {obs['avg_speed_kmh']:.1f}km/h")
    parts.append(f"车流密度: {obs['traffic_density']:.0f}辆/分钟")

    if abs(obs.get("acceleration", 0)) > 0.01:
        parts.append(f"加速度: {obs['acceleration']:+.2f}m/s²")

    deviation = obs.get("speed_deviation_kmh", 0)
    if abs(deviation) > 1.0:
        parts.append(f"速度偏差: {deviation:+.1f}km/h")

    return " | ".join(parts)


def format_journey_prompt(journey: dict) -> str:
    """将一条车辆旅程转换为用户提示词"""
    scenario = journey["scenario"]
    desc = SCENARIO_DESCRIPTIONS.get(scenario, scenario)

    lines = [f"场景: {desc}", "", "该车辆的监测点观测记录（按时间顺序）:", ""]

    for i, obs in enumerate(journey["observations"]):
        lines.append(format_observation(obs, i))

    lines.append("")
    lines.append("请分析该车辆的驾驶行为是否异常。")

    return "\n".join(lines)


def generate_reason(journey: dict) -> str:
    """根据驾驶员类型和观测数据生成分析理由"""
    driver_type = journey["driver_type"]
    obs_list = journey["observations"]

    if not obs_list:
        return "观测数据不足，无法判断。"

    speeds = [o["speed_kmh"] for o in obs_list]
    limits = [o["speed_limit_kmh"] for o in obs_list]
    deviations = [o.get("speed_deviation_kmh", 0) for o in obs_list]
    accels = [o.get("acceleration", 0) for o in obs_list]

    avg_speed = sum(speeds) / len(speeds)
    max_speed = max(speeds)
    min_speed = min(speeds)
    speed_range = max_speed - min_speed
    avg_limit = sum(limits) / len(limits)
    avg_deviation = sum(deviations) / len(deviations)
    max_accel = max(abs(a) for a in accels)

    if driver_type == "rule_breaker":
        reasons = []
        over_limit_count = sum(1 for s, l in zip(speeds, limits) if s > l * 1.1)
        if over_limit_count > 0:
            reasons.append(f"在{over_limit_count}个监测点超速超过10%")
        if avg_deviation > 10:
            reasons.append(f"平均车速高于路段平均{avg_deviation:.0f}km/h")
        if max_accel > 3.0:
            reasons.append("存在急加速或急减速行为")
        if not reasons:
            reasons.append("行驶速度持续高于车流平均水平，存在激进驾驶倾向")
        return "该车辆驾驶行为异常：" + "，".join(reasons) + "。"

    elif driver_type == "fatigued":
        reasons = []
        if speed_range > 15:
            reasons.append(f"速度波动范围达{speed_range:.0f}km/h")
        if avg_speed < avg_limit * 0.85:
            reasons.append(f"平均车速仅{avg_speed:.0f}km/h，显著低于限速")
        if avg_deviation < -5:
            reasons.append("持续低于路段平均车速")
        if not reasons:
            reasons.append("行驶速度异常偏低且不稳定，呈现疲劳驾驶特征")
        return "该车辆驾驶行为异常：" + "，".join(reasons) + "。"

    elif driver_type == "rule_follower":
        return "该车辆行驶平稳，严格遵守限速，各监测点速度一致性高，属于正常驾驶行为。"

    elif driver_type == "aggressive_ai":
        return "该车辆行驶速度接近限速，加速较快但未超速，属于正常驾驶行为。"

    else:  # normal_human
        return "该车辆行驶速度在正常范围内，有轻微波动属于正常人类驾驶特征，属于正常驾驶行为。"


def journey_to_training_sample(journey: dict) -> dict:
    """将车辆旅程转换为LLM训练样本（chat格式）"""
    user_content = format_journey_prompt(journey)
    reason = generate_reason(journey)
    label = journey["label"]

    # 置信度：根据特征明显程度
    if label == "abnormal":
        confidence = 0.85 + 0.1 * (len(journey["observations"]) / 8)
    else:
        confidence = 0.88 + 0.08 * (len(journey["observations"]) / 8)
    confidence = min(0.98, confidence)

    assistant_content = json.dumps({
        "classification": label,
        "confidence": round(confidence, 2),
        "reason": reason,
    }, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }
