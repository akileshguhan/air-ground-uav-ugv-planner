"""Shared data shapes. Python 3.8 typing only (typing.List/Optional)."""
from __future__ import annotations
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class Pose:
    timestamp: float
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float = 1.0
    SIZE_BYTES = 8 * 8

    def pack(self) -> bytes:
        return struct.pack("<8d", self.timestamp, self.x, self.y, self.z,
                           self.qx, self.qy, self.qz, self.qw)

    @staticmethod
    def unpack(buf: bytes) -> "Pose":
        return Pose(*struct.unpack("<8d", buf))


@dataclass
class ScanPoint:
    angle_deg: float
    distance_m: float
    quality: int = 15


@dataclass
class ScanSlice:
    slice_id: int
    pose: Pose
    points: List[ScanPoint] = field(default_factory=list)


@dataclass
class SegmentFeatures:
    segment_id: str
    start: np.ndarray
    end: np.ndarray
    mean_incline_deg: float
    max_incline_deg: float
    mean_decline_deg: float
    max_decline_deg: float
    max_step_height_m: float
    max_gap_width_m: float
    overhead_clearance_m: float
    corridor_width_m: float
    roughness: float
    n_points: int = 0
    along_track_spacing_m: float = 0.0
    planar_fraction: float = 1.0   # share of sampled normals used for slope
    step_legacy_m: float = 0.0     # old sorted-along-track step, diagnostics only


@dataclass
class CandidateSegmentMsg:
    segment_id: str
    start_xyz: Tuple[float, float, float]
    end_xyz: Tuple[float, float, float]
    features: Optional[SegmentFeatures] = None


@dataclass
class SegmentResultMsg:
    segment_id: str
    feasible: bool
    feasibility_score: float
    recommended_speed_factor: float
    failed_checks: List[str] = field(default_factory=list)
