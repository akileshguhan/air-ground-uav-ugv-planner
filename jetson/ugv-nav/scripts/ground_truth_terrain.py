"""Shared synthetic terrain with exactly known features, used by every test so
RPLiDAR and D455 results are compared on identical ground truth."""
import math
import numpy as np

GROUND_TRUTH = {
    "ramp_start_x": 8.0, "ramp_end_x": 14.0, "ramp_angle_deg": 12.0,
    "step_x": 18.0, "step_height_m": 0.08,
    "trench_x": 24.0, "trench_width_m": 0.35, "trench_depth_m": 1.2,
    "trench_y_min": -0.6,
}


def terrain(x, y):
    g = GROUND_TRUTH
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    z = np.zeros(np.broadcast(x, y).shape)
    ramp = np.clip((x - g["ramp_start_x"]) / (g["ramp_end_x"] - g["ramp_start_x"]), 0, 1)
    z = z + ramp * (g["ramp_end_x"] - g["ramp_start_x"]) * math.tan(math.radians(g["ramp_angle_deg"]))
    z = z + np.where(x > g["step_x"], g["step_height_m"], 0.0)
    trench = ((x > g["trench_x"]) & (x < g["trench_x"] + g["trench_width_m"]) &
              (y > g["trench_y_min"]))
    z = z - np.where(trench, g["trench_depth_m"], 0.0)
    return z + 0.012 * np.sin(2.1 * x) * np.cos(1.7 * y)
