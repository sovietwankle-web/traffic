"""
监测点系统：记录车辆经过监测点时的观测数据，计算衍生指标。
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Observation:
    """单次监测点观测记录"""
    vehicle_id: str
    driver_type: str        # 真实标签（训练用，推理时不可见）
    monitor_id: str
    monitor_name: str
    segment_id: str
    lane: int
    timestamp: float        # 仿真时间(s)
    speed: float            # 车速(m/s)
    speed_limit: float      # 路段限速(m/s)
    acceleration: float     # 加速度(m/s^2)
    lane_offset: float = 0.0  # 车道内横向偏移


@dataclass
class MonitoringPoint:
    """监测点"""
    id: str
    segment_id: str
    position: float         # 路段上的位置(m)
    name: str               # 人类可读名称
    # 滑动窗口统计
    _recent_speeds: deque = field(default_factory=lambda: deque(maxlen=600))  # 60s窗口@10Hz
    _recent_timestamps: deque = field(default_factory=lambda: deque(maxlen=600))
    _vehicle_count_window: deque = field(default_factory=lambda: deque(maxlen=600))

    def record_passing(self, speed: float, timestamp: float):
        """记录一辆车经过"""
        self._recent_speeds.append(speed)
        self._recent_timestamps.append(timestamp)
        self._vehicle_count_window.append(timestamp)

    def get_avg_speed(self, current_time: float, window: float = 60.0) -> float:
        """获取最近window秒内的平均车速"""
        cutoff = current_time - window
        speeds = [s for s, t in zip(self._recent_speeds, self._recent_timestamps) if t >= cutoff]
        if not speeds:
            return 0.0
        return sum(speeds) / len(speeds)

    def get_traffic_density(self, current_time: float, window: float = 60.0) -> float:
        """获取最近window秒内的车流量(辆/分钟)"""
        cutoff = current_time - window
        count = sum(1 for t in self._vehicle_count_window if t >= cutoff)
        return count * (60.0 / window)


class MonitorSystem:
    """监测系统：管理所有监测点，检测车辆是否经过"""

    def __init__(self):
        self.monitors: Dict[str, MonitoringPoint] = {}
        self.observations: List[Observation] = []
        # 按路段索引监测点
        self._segment_monitors: Dict[str, List[MonitoringPoint]] = {}

    def add_monitor(self, monitor: MonitoringPoint):
        self.monitors[monitor.id] = monitor
        self._segment_monitors.setdefault(monitor.segment_id, []).append(monitor)
        # 按位置排序
        self._segment_monitors[monitor.segment_id].sort(key=lambda m: m.position)

    def get_monitors_on_segment(self, segment_id: str) -> List[MonitoringPoint]:
        return self._segment_monitors.get(segment_id, [])

    def get_monitor_positions_on_segment(self, segment_id: str) -> List[float]:
        return [m.position for m in self.get_monitors_on_segment(segment_id)]

    def check_vehicle_passing(
        self,
        vehicle_id: str,
        driver_type: str,
        segment_id: str,
        old_position: float,
        new_position: float,
        lane: int,
        speed: float,
        speed_limit: float,
        acceleration: float,
        timestamp: float,
        lane_offset: float = 0.0,
    ):
        """检查车辆是否在本tick经过了某个监测点"""
        monitors = self.get_monitors_on_segment(segment_id)
        for monitor in monitors:
            if old_position <= monitor.position < new_position:
                # 车辆经过了该监测点
                monitor.record_passing(speed, timestamp)

                avg_speed = monitor.get_avg_speed(timestamp)
                density = monitor.get_traffic_density(timestamp)

                obs = Observation(
                    vehicle_id=vehicle_id,
                    driver_type=driver_type,
                    monitor_id=monitor.id,
                    monitor_name=monitor.name,
                    segment_id=segment_id,
                    lane=lane,
                    timestamp=timestamp,
                    speed=speed,
                    speed_limit=speed_limit,
                    acceleration=acceleration,
                    lane_offset=lane_offset,
                )
                self.observations.append(obs)

    def get_observations_by_vehicle(self) -> Dict[str, List[Observation]]:
        """按车辆ID分组观测记录"""
        grouped: Dict[str, List[Observation]] = {}
        for obs in self.observations:
            grouped.setdefault(obs.vehicle_id, []).append(obs)
        # 每辆车的观测按时间排序
        for vid in grouped:
            grouped[vid].sort(key=lambda o: o.timestamp)
        return grouped

    def get_enriched_observations_by_vehicle(self) -> Dict[str, List[dict]]:
        """获取增强后的观测数据（包含衍生指标）"""
        # 先按车辆分组
        grouped = self.get_observations_by_vehicle()
        enriched: Dict[str, List[dict]] = {}

        for vid, obs_list in grouped.items():
            enriched[vid] = []
            for obs in obs_list:
                monitor = self.monitors[obs.monitor_id]
                avg_speed = monitor.get_avg_speed(obs.timestamp)
                density = monitor.get_traffic_density(obs.timestamp)
                deviation = obs.speed - avg_speed

                enriched[vid].append({
                    "vehicle_id": obs.vehicle_id,
                    "driver_type": obs.driver_type,
                    "monitor_id": obs.monitor_id,
                    "monitor_name": obs.monitor_name,
                    "segment_id": obs.segment_id,
                    "lane": obs.lane,
                    "timestamp": obs.timestamp,
                    "speed_ms": obs.speed,
                    "speed_kmh": obs.speed * 3.6,
                    "speed_limit_ms": obs.speed_limit,
                    "speed_limit_kmh": obs.speed_limit * 3.6,
                    "acceleration": obs.acceleration,
                    "avg_speed_kmh": avg_speed * 3.6,
                    "traffic_density": density,
                    "speed_deviation_kmh": deviation * 3.6,
                    "lane_offset": obs.lane_offset,
                })

        return enriched

    def reset(self):
        """清空所有观测数据"""
        self.observations.clear()
        for m in self.monitors.values():
            m._recent_speeds.clear()
            m._recent_timestamps.clear()
            m._vehicle_count_window.clear()
