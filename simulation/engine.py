"""
交通仿真引擎：tick-based，管理车辆生成、移动、路段转换、监测点记录。
"""

import random
from typing import Dict, List, Optional, Callable, Tuple

from simulation.road_network import RoadNetwork, RoadSegment
from simulation.vehicle import Vehicle, DriverParams
from simulation.monitor import MonitorSystem


class SimulationEngine:
    """交通仿真引擎"""

    def __init__(
        self,
        network: RoadNetwork,
        monitor_system: MonitorSystem,
        dt: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.network = network
        self.monitor_system = monitor_system
        self.dt = dt
        self.current_time = 0.0
        self.vehicles: Dict[str, Vehicle] = {}
        self._vehicle_counter = 0
        self._finished_vehicles: List[Vehicle] = []

        # 车辆生成配置
        self.spawn_configs: List[dict] = []
        # driver_factory: callable(driver_type) -> (DriverParams, initial_speed)
        self.driver_factory: Optional[Callable] = None

        if seed is not None:
            random.seed(seed)

    def set_driver_factory(self, factory: Callable):
        self.driver_factory = factory

    def add_spawn_config(
        self,
        segment_id: str,
        rate: float,
        driver_type_weights: Dict[str, float],
    ):
        """
        添加车辆生成配置。
        segment_id: 入口路段ID
        rate: 生成速率(辆/秒)
        driver_type_weights: {driver_type: weight} 各类型的权重
        """
        total = sum(driver_type_weights.values())
        normalized = {k: v / total for k, v in driver_type_weights.items()}
        self.spawn_configs.append({
            "segment_id": segment_id,
            "rate": rate,
            "driver_weights": normalized,
        })

    def _choose_driver_type(self, weights: Dict[str, float]) -> str:
        types = list(weights.keys())
        probs = list(weights.values())
        return random.choices(types, weights=probs, k=1)[0]

    def _spawn_vehicles(self):
        """按泊松过程生成车辆"""
        for config in self.spawn_configs:
            # 泊松过程：每tick生成概率 = rate * dt
            if random.random() < config["rate"] * self.dt:
                seg_id = config["segment_id"]
                segment = self.network.get_segment(seg_id)
                if segment is None:
                    continue

                driver_type = self._choose_driver_type(config["driver_weights"])

                if self.driver_factory is None:
                    raise RuntimeError("driver_factory not set")

                params, initial_speed = self.driver_factory(driver_type, segment)

                self._vehicle_counter += 1
                vid = f"v_{self._vehicle_counter:05d}"

                lane = random.randint(0, segment.num_lanes - 1)

                vehicle = Vehicle(
                    id=vid,
                    driver_type=driver_type,
                    params=params,
                    current_segment=seg_id,
                    lane=lane,
                    position=0.0,
                    speed=initial_speed,
                )

                # 检查入口处是否有足够间距
                if self._has_space_at_entrance(seg_id, lane, params.min_gap * 2):
                    self.vehicles[vid] = vehicle

    def _has_space_at_entrance(self, segment_id: str, lane: int, min_gap: float) -> bool:
        """检查路段入口处是否有足够空间"""
        for v in self.vehicles.values():
            if v.current_segment == segment_id and v.lane == lane and v.position < min_gap:
                return False
        return True

    def _get_vehicles_on_segment(self, segment_id: str) -> List[Vehicle]:
        """获取某路段上的所有车辆，按位置排序(从前到后)"""
        vehs = [v for v in self.vehicles.values() if v.current_segment == segment_id]
        vehs.sort(key=lambda v: -v.position)  # 位置大的在前
        return vehs

    def _get_leading_vehicle(self, vehicle: Vehicle) -> Tuple[Optional[Vehicle], float]:
        """获取同车道前方最近的车辆及间距"""
        same_lane = [
            v for v in self.vehicles.values()
            if v.current_segment == vehicle.current_segment
            and v.lane == vehicle.lane
            and v.position > vehicle.position
            and v.id != vehicle.id
        ]
        if not same_lane:
            # 检查下一路段是否有车
            connections = self.network.get_outgoing_connections(vehicle.current_segment)
            segment = self.network.get_segment(vehicle.current_segment)
            if segment and connections:
                remaining = segment.length - vehicle.position
                for conn in connections:
                    next_vehs = [
                        v for v in self.vehicles.values()
                        if v.current_segment == conn.to_segment and v.lane == vehicle.lane
                    ]
                    if next_vehs:
                        nearest = min(next_vehs, key=lambda v: v.position)
                        gap = remaining + nearest.position - 5.0  # 减去车长估计
                        return nearest, max(0.1, gap)
            return None, 1000.0  # 无前车

        nearest = min(same_lane, key=lambda v: v.position)
        gap = nearest.position - vehicle.position - 5.0  # 减去车长
        return nearest, max(0.1, gap)

    def _get_follower_vehicle(self, vehicle: Vehicle, lane: int) -> Tuple[Optional[Vehicle], float]:
        """获取指定车道后方最近的车辆及间距"""
        followers = [
            v for v in self.vehicles.values()
            if v.current_segment == vehicle.current_segment
            and v.lane == lane
            and v.position < vehicle.position
            and v.id != vehicle.id
        ]
        if not followers:
            return None, 1000.0
        nearest = max(followers, key=lambda v: v.position)
        gap = vehicle.position - nearest.position - 5.0
        return nearest, max(0.1, gap)

    def _compute_acceleration(self, vehicle: Vehicle, segment: RoadSegment) -> float:
        """计算车辆加速度"""
        speed_limit = segment.speed_limit

        # 违规者在监测点附近减速
        if vehicle.params.monitor_awareness > 0:
            monitor_positions = self.monitor_system.get_monitor_positions_on_segment(
                vehicle.current_segment
            )
            if vehicle.is_near_monitor_and_aware(monitor_positions):
                speed_limit = segment.speed_limit * 0.95  # 减速到限速以下

        # 获取前车信息
        leader, gap = self._get_leading_vehicle(vehicle)
        delta_v = vehicle.speed - leader.speed if leader else 0.0

        # 交叉口红灯检测
        intersection = self.network.get_intersection_at_end(vehicle.current_segment)
        if intersection and not intersection.is_green_for(vehicle.current_segment, self.current_time):
            if not vehicle.should_run_red_light():
                # 将红灯视为前方静止障碍物
                dist_to_stop = segment.length - vehicle.position
                if dist_to_stop < gap:
                    gap = dist_to_stop
                    delta_v = vehicle.speed  # 相当于前方有静止物体

        accel = vehicle.compute_idm_acceleration(speed_limit, gap, delta_v)
        return accel

    def _try_lane_change(self, vehicle: Vehicle, segment: RoadSegment):
        """尝试换道"""
        if segment.num_lanes <= 1 or segment.no_lane_change:
            return

        # 考虑左右两个方向
        for target_lane in [vehicle.lane - 1, vehicle.lane + 1]:
            if target_lane < 0 or target_lane >= segment.num_lanes:
                continue

            # 目标车道前车
            old_lane = vehicle.lane
            vehicle.lane = target_lane
            target_leader, target_gap = self._get_leading_vehicle(vehicle)
            vehicle.lane = old_lane

            if target_gap < vehicle.params.min_gap:
                continue

            # 计算目标车道加速度
            target_delta_v = vehicle.speed - target_leader.speed if target_leader else 0.0
            target_accel = vehicle.compute_idm_acceleration(
                segment.speed_limit, target_gap, target_delta_v
            )

            # 当前车道加速度
            current_leader, current_gap = self._get_leading_vehicle(vehicle)
            current_delta_v = vehicle.speed - current_leader.speed if current_leader else 0.0
            current_accel = vehicle.compute_idm_acceleration(
                segment.speed_limit, current_gap, current_delta_v
            )

            # 目标车道后车影响
            target_follower, target_follower_gap = self._get_follower_vehicle(vehicle, target_lane)
            target_follower_accel_change = 0.0
            if target_follower and target_follower_gap < 50:
                # 后车因为我插入会减速
                target_follower_accel_change = -1.0 * (10.0 / max(target_follower_gap, 1.0))

            # 安全间距检查
            safe = target_gap > vehicle.params.min_gap * 1.5

            if vehicle.compute_mobil_lane_change(
                current_accel, target_accel,
                0.0, target_follower_accel_change,
                safe,
            ):
                vehicle.lane = target_lane
                vehicle._lane_changes += 1
                break

    def _handle_segment_transition(self, vehicle: Vehicle):
        """处理路段转换"""
        segment = self.network.get_segment(vehicle.current_segment)
        if segment is None:
            return

        if vehicle.position < segment.length:
            return

        # 超出当前路段
        overflow = vehicle.position - segment.length
        connections = self.network.get_outgoing_connections(vehicle.current_segment)

        if not connections:
            # 到达出口，车辆完成旅程
            vehicle._finished = True
            return

        # 检查交叉口是否允许通行
        intersection = self.network.get_intersection_at_end(vehicle.current_segment)
        if intersection and not intersection.is_green_for(vehicle.current_segment, self.current_time):
            if not vehicle.should_run_red_light():
                # 红灯，停在路段末尾
                vehicle.position = segment.length - 0.1
                vehicle.speed = 0.0
                return

        # 选择下一路段
        if vehicle.route and vehicle.route_index < len(vehicle.route):
            # 按预定路线
            next_seg = vehicle.route[vehicle.route_index]
            vehicle.route_index += 1
        else:
            # 按权重随机选择
            weights = [c.weight for c in connections]
            chosen = random.choices(connections, weights=weights, k=1)[0]
            next_seg = chosen.to_segment

        next_segment = self.network.get_segment(next_seg)
        if next_segment is None:
            vehicle._finished = True
            return

        vehicle.current_segment = next_seg
        vehicle.position = overflow
        # 调整车道（如果新路段车道数不同）
        if vehicle.lane >= next_segment.num_lanes:
            vehicle.lane = next_segment.num_lanes - 1

    def step(self):
        """执行一个仿真步"""
        # 1. 生成新车辆
        self._spawn_vehicles()

        # 2. 对每辆车计算加速度和换道
        to_remove = []
        for vid, vehicle in self.vehicles.items():
            if vehicle._finished:
                to_remove.append(vid)
                continue

            segment = self.network.get_segment(vehicle.current_segment)
            if segment is None:
                to_remove.append(vid)
                continue

            # 计算加速度
            vehicle.acceleration = self._compute_acceleration(vehicle, segment)

            # 尝试换道
            self._try_lane_change(vehicle, segment)

            # 记录旧位置（用于监测点检测）
            old_position = vehicle.position

            # 更新位置
            vehicle.update_position(self.dt)

            # 监测点检测
            self.monitor_system.check_vehicle_passing(
                vehicle_id=vehicle.id,
                driver_type=vehicle.driver_type,
                segment_id=vehicle.current_segment,
                old_position=old_position,
                new_position=vehicle.position,
                lane=vehicle.lane,
                speed=vehicle.speed,
                speed_limit=segment.speed_limit,
                acceleration=vehicle.acceleration,
                timestamp=self.current_time,
                lane_offset=vehicle.lane_offset,
            )

            # 路段转换
            self._handle_segment_transition(vehicle)

            if vehicle._finished:
                to_remove.append(vid)

        # 3. 移除完成的车辆
        for vid in to_remove:
            if vid in self.vehicles:
                self._finished_vehicles.append(self.vehicles.pop(vid))

        # 4. 更新时间
        self.current_time += self.dt

    def run(self, duration: float, progress_interval: float = 60.0) -> dict:
        """
        运行仿真。
        duration: 仿真时长(s)
        progress_interval: 进度报告间隔(s)
        """
        total_ticks = int(duration / self.dt)
        next_progress = progress_interval

        for tick in range(total_ticks):
            self.step()

            if self.current_time >= next_progress:
                active = len(self.vehicles)
                finished = len(self._finished_vehicles)
                obs = len(self.monitor_system.observations)
                print(f"  t={self.current_time:.0f}s: {active} active, {finished} finished, {obs} observations")
                next_progress += progress_interval

        return {
            "duration": duration,
            "total_vehicles": self._vehicle_counter,
            "finished_vehicles": len(self._finished_vehicles),
            "active_vehicles": len(self.vehicles),
            "total_observations": len(self.monitor_system.observations),
        }

    def get_all_finished_vehicles(self) -> List[Vehicle]:
        return self._finished_vehicles

    def reset(self):
        """重置仿真状态"""
        self.current_time = 0.0
        self.vehicles.clear()
        self._finished_vehicles.clear()
        self._vehicle_counter = 0
        self.monitor_system.reset()
