"""
车辆状态与物理模型。
纵向控制：IDM (Intelligent Driver Model)
横向控制：简化MOBIL换道模型
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.road_network import RoadSegment

# 驾驶员类型常量
DRIVER_RULE_FOLLOWER = "rule_follower"
DRIVER_AGGRESSIVE_AI = "aggressive_ai"
DRIVER_NORMAL_HUMAN = "normal_human"
DRIVER_RULE_BREAKER = "rule_breaker"
DRIVER_FATIGUED = "fatigued"

NORMAL_TYPES = {DRIVER_RULE_FOLLOWER, DRIVER_AGGRESSIVE_AI, DRIVER_NORMAL_HUMAN}
ABNORMAL_TYPES = {DRIVER_RULE_BREAKER, DRIVER_FATIGUED}


@dataclass
class DriverParams:
    """驾驶员参数（IDM + MOBIL + 特殊行为）"""
    desired_speed_factor: float = 1.0   # 期望速度 = speed_limit * factor
    min_gap: float = 4.0               # 最小净间距(m)
    time_headway: float = 1.8          # 期望时距(s)
    max_accel: float = 2.5             # 最大加速度(m/s^2)
    comfortable_decel: float = 2.5     # 舒适减速度(m/s^2)
    lane_change_threshold: float = 0.2 # MOBIL换道阈值
    politeness: float = 0.5            # MOBIL礼让系数
    reaction_delay: int = 0            # 反应延迟(ticks)
    noise_std: float = 0.1            # 加速度噪声标准差
    # 特殊行为参数
    red_light_run_prob: float = 0.0    # 闯红灯概率
    monitor_awareness: float = 0.0     # 对监测点的感知概率(0=不感知)
    lane_drift_amplitude: float = 0.0  # 车道漂移振幅(0=无漂移)
    micro_sleep_prob: float = 0.0      # 微睡眠概率(每tick)
    micro_sleep_duration: int = 0      # 微睡眠持续(ticks)


@dataclass
class Vehicle:
    """车辆状态"""
    id: str
    driver_type: str
    params: DriverParams
    current_segment: str
    lane: int = 0
    position: float = 0.0          # 沿路段位置(m)
    speed: float = 0.0             # 当前速度(m/s)
    acceleration: float = 0.0      # 当前加速度(m/s^2)
    lane_offset: float = 0.0       # 车道内横向偏移(用于漂移)
    # 路径
    route: List[str] = field(default_factory=list)  # 计划路段序列
    route_index: int = 0
    # 内部状态
    _reaction_buffer: List[float] = field(default_factory=list)
    _micro_sleep_remaining: int = 0
    _drift_phase: float = 0.0      # 漂移相位
    _ticks_alive: int = 0
    _lane_changes: int = 0         # 换道次数统计
    _finished: bool = False        # 是否已离开网络

    @property
    def is_abnormal(self) -> bool:
        return self.driver_type in ABNORMAL_TYPES

    @property
    def label(self) -> str:
        return "abnormal" if self.is_abnormal else "normal"

    def desired_speed(self, speed_limit: float) -> float:
        return speed_limit * self.params.desired_speed_factor

    def compute_idm_acceleration(
        self,
        speed_limit: float,
        gap: float,
        delta_v: float,
    ) -> float:
        """
        IDM跟驰模型计算加速度。
        gap: 与前车净间距(m)，无前车时传入一个大值
        delta_v: 与前车的速度差(正=接近前车)
        """
        v = self.speed
        v0 = self.desired_speed(speed_limit)
        a_max = self.params.max_accel
        b = self.params.comfortable_decel
        s0 = self.params.min_gap
        T = self.params.time_headway

        # 期望间距
        s_star = s0 + max(0.0, v * T + v * delta_v / (2.0 * math.sqrt(a_max * b)))

        # IDM加速度
        if gap < 0.1:
            gap = 0.1  # 防止除零

        free_road_term = 1.0 - (v / v0) ** 4 if v0 > 0 else 0.0
        interaction_term = (s_star / gap) ** 2

        accel = a_max * (free_road_term - interaction_term)

        # 添加噪声
        if self.params.noise_std > 0:
            accel += random.gauss(0, self.params.noise_std)

        # 限幅
        accel = max(-self.params.comfortable_decel * 2, min(a_max, accel))

        return accel

    def compute_mobil_lane_change(
        self,
        current_lane_accel: float,
        target_lane_accel: float,
        current_follower_accel_change: float,
        target_follower_accel_change: float,
        safe_gap: bool,
    ) -> bool:
        """
        简化MOBIL模型判断是否换道。
        返回True表示应该换道。
        """
        if not safe_gap:
            return False

        p = self.params.politeness
        threshold = self.params.lane_change_threshold

        # 换道收益 = 自身收益 - 礼让系数 * 对后车的不利影响
        advantage = (target_lane_accel - current_lane_accel
                     - p * (target_follower_accel_change + current_follower_accel_change))

        return advantage > threshold

    def update_position(self, dt: float):
        """更新位置和速度"""
        # 微睡眠状态：不加速不减速，保持当前速度
        if self._micro_sleep_remaining > 0:
            self._micro_sleep_remaining -= 1
            # 仅靠惯性前进
            self.position += self.speed * dt
            self._ticks_alive += 1
            return

        # 应用反应延迟
        if self.params.reaction_delay > 0:
            self._reaction_buffer.append(self.acceleration)
            if len(self._reaction_buffer) > self.params.reaction_delay:
                effective_accel = self._reaction_buffer.pop(0)
            else:
                effective_accel = 0.0  # 还没有足够的缓冲
        else:
            effective_accel = self.acceleration

        # 运动学更新
        self.speed += effective_accel * dt
        self.speed = max(0.0, self.speed)  # 不能倒车
        self.position += self.speed * dt + 0.5 * effective_accel * dt ** 2

        # 车道漂移（疲劳驾驶）
        if self.params.lane_drift_amplitude > 0:
            self._drift_phase += dt * 0.5  # 漂移频率
            self.lane_offset = self.params.lane_drift_amplitude * math.sin(self._drift_phase)

        # 微睡眠触发
        if self.params.micro_sleep_prob > 0 and random.random() < self.params.micro_sleep_prob:
            self._micro_sleep_remaining = self.params.micro_sleep_duration

        self._ticks_alive += 1

    def should_run_red_light(self) -> bool:
        if self.params.red_light_run_prob <= 0:
            return False
        return random.random() < self.params.red_light_run_prob

    def is_near_monitor_and_aware(self, monitor_positions: List[float], awareness_range: float = 100.0) -> bool:
        """检查是否在已知监测点附近（违规者会减速）"""
        if self.params.monitor_awareness <= 0:
            return False
        for mp in monitor_positions:
            if abs(self.position - mp) < awareness_range:
                if random.random() < self.params.monitor_awareness:
                    return True
        return False
