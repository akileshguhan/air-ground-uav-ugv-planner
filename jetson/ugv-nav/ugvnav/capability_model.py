"""Private capability envelope. ALL VALUES PLACEHOLDERS. Import only from evaluator."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from .protocol import SegmentFeatures


@dataclass
class LimitPair:
    nominal: float
    hard: float


@dataclass
class UGVCapabilityModel:
    max_incline_deg: LimitPair = field(default_factory=lambda: LimitPair(15.0, 25.0))
    max_decline_deg: LimitPair = field(default_factory=lambda: LimitPair(12.0, 20.0))
    max_step_height_m: LimitPair = field(default_factory=lambda: LimitPair(0.10, 0.15))
    max_gap_width_m: LimitPair = field(default_factory=lambda: LimitPair(0.18, 0.25))
    max_roughness: LimitPair = field(default_factory=lambda: LimitPair(0.25, 0.35))
    min_ground_clearance_m: float = 0.20
    vehicle_width_m: float = 0.55
    min_turn_radius_m: float = 0.40

    def evaluate(self, f: SegmentFeatures, turning_radius_m: float = float("inf")
                 ) -> Tuple[bool, float, float, List[str]]:
        def g(a, lim):
            return 1.0 if a <= lim.nominal else 1.0 - (a - lim.nominal) / max(lim.hard - lim.nominal, 1e-6)
        s: Dict[str, float] = {"incline": g(f.max_incline_deg, self.max_incline_deg),
                               "decline": g(f.max_decline_deg, self.max_decline_deg),
                               "step": g(f.max_step_height_m, self.max_step_height_m),
                               "gap": g(f.max_gap_width_m, self.max_gap_width_m),
                               "roughness": g(f.roughness, self.max_roughness)}
        failed = [k for k, v in s.items() if v < 0]
        if f.corridor_width_m < self.vehicle_width_m:
            failed.append("corridor_width")
        if f.overhead_clearance_m < self.min_ground_clearance_m:
            failed.append("clearance")
        if turning_radius_m < self.min_turn_radius_m:
            failed.append("turn_radius")
        ok = not failed
        score = max(0.0, min(s.values())) if ok else 0.0
        return ok, score, (0.3 + 0.7 * score if ok else 0.0), failed
