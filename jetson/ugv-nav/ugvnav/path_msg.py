"""
path_msg.py -- the UGV -> drone output: the selected route plus a recommended
speed factor per segment. This is where one planning cycle ends.

Carries NO capability thresholds, only outcomes (privacy boundary).
Python 3.8 compatible.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import List, Tuple

MSG_PATH = 2  # message type on the bidirectional link


@dataclass
class PathMsg:
    path_id: int
    created_s: float
    waypoints: List[Tuple[float, float, float]] = field(default_factory=list)
    speed_factors: List[float] = field(default_factory=list)  # one per segment
    cost: float = 0.0

    def pack(self) -> bytes:
        n = len(self.waypoints)
        if n and len(self.speed_factors) != n - 1:
            raise ValueError("need exactly one speed factor per segment")
        out = struct.pack("<QdI", self.path_id, self.created_s, n)
        out += b"".join(struct.pack("<3d", *w) for w in self.waypoints)
        out += b"".join(struct.pack("<f", s) for s in self.speed_factors)
        return out + struct.pack("<d", self.cost)

    @staticmethod
    def unpack(buf: bytes) -> "PathMsg":
        pid, created, n = struct.unpack("<QdI", buf[:20])
        off = 20
        wps = []
        for _ in range(n):
            wps.append(struct.unpack("<3d", buf[off:off + 24]))
            off += 24
        spd = []
        for _ in range(max(n - 1, 0)):
            spd.append(struct.unpack("<f", buf[off:off + 4])[0])
            off += 4
        cost = struct.unpack("<d", buf[off:off + 8])[0]
        return PathMsg(pid, created, [tuple(w) for w in wps], spd, cost)

    def to_json(self) -> str:
        return json.dumps({"path_id": self.path_id, "created_s": self.created_s,
                           "waypoints": self.waypoints,
                           "speed_factors": self.speed_factors,
                           "cost": self.cost}, indent=1)
