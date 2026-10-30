"""The only module that imports capability_model (privacy boundary)."""
from __future__ import annotations
from typing import List
from .capability_model import UGVCapabilityModel
from .protocol import CandidateSegmentMsg, SegmentResultMsg


class UGVSideEvaluator:
    def __init__(self, model: UGVCapabilityModel):
        self._model = model

    def evaluate(self, msg: CandidateSegmentMsg, turning_radius_m: float = float("inf")):
        ok, sc, sp, failed = self._model.evaluate(msg.features, turning_radius_m)
        return SegmentResultMsg(msg.segment_id, ok, sc, sp, failed)

    def evaluate_all(self, msgs: List[CandidateSegmentMsg]) -> List[SegmentResultMsg]:
        return [self.evaluate(m) for m in msgs]
