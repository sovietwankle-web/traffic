from scenarios.consecutive_intersections import build_consecutive_intersections
from scenarios.highway_ramps import build_highway_ramps
from scenarios.hutong_grid import build_hutong_grid
from scenarios.interchange import build_interchange
from scenarios.roundabout import build_roundabout
from scenarios.tunnel import build_tunnel
from scenarios.bottleneck import build_bottleneck

ALL_SCENARIOS = {
    "consecutive_intersections": build_consecutive_intersections,
    "highway_ramps": build_highway_ramps,
    "hutong_grid": build_hutong_grid,
    "interchange": build_interchange,
    "roundabout": build_roundabout,
    "tunnel": build_tunnel,
    "bottleneck": build_bottleneck,
}
