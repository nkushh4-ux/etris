"""Stage 12D - Congestion / Severe-Delay alert persistence engine.

Consumes CongestionResult snapshots produced by Stage 10
TrafficAnalyticsService.congestion().  No new AI model.  No video
inference.  Stage 10 thresholds unchanged.

Policy (defaults overridable via CongestionAlertPolicy):
  CONGESTED for >= congestion_confirm_count consecutive snapshots  -> CONGESTION active
  SEVERE    for >= severe_confirm_count    consecutive snapshots   -> SEVERE_DELAY active
  FREE_FLOW or MODERATE for >= recovery_count consecutive snapshots -> clear active alert
  INSUFFICIENT_DATA: ignored (never creates / does not immediately clear)

Escalation: active CONGESTION superseded (cleared) when SEVERE is confirmed.
At most one active alert per segment at a time.

Dedup key: segment_id = "SRC->DST"  (stable Stage 10 identity).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import Enum

from backend.alerts.models import Alert, AlertSeverity, AlertStatus, AlertType
from backend.alerts.repository import AlertRepository
from backend.analytics.models import CongestionLevel, CongestionResult

RULE_VERSION = "CONGESTION-ALERT-1.0"


@dataclass(frozen=True, slots=True)
class CongestionAlertPolicy:
    """Alert-trigger persistence thresholds (consecutive snapshot counts)."""
    congestion_confirm_count: int = 3
    severe_confirm_count: int = 2
    recovery_count: int = 3

    def __post_init__(self) -> None:
        for name, val in (
            ("congestion_confirm_count", self.congestion_confirm_count),
            ("severe_confirm_count", self.severe_confirm_count),
            ("recovery_count", self.recovery_count),
        ):
            if val < 1:
                raise ValueError(f"{name} must be >= 1")


class _Phase(str, Enum):
    NORMAL = "NORMAL"
    CONGESTION_PENDING = "CONGESTION_PENDING"
    CONGESTION_ACTIVE = "CONGESTION_ACTIVE"
    SEVERE_PENDING = "SEVERE_PENDING"
    SEVERE_ACTIVE = "SEVERE_ACTIVE"
    RECOVERY_PENDING = "RECOVERY_PENDING"


@dataclass(slots=True)
class _SegMem:
    phase: _Phase = _Phase.NORMAL
    pending_count: int = 0
    recovery_count: int = 0
    first_qualifying_at: float | None = None
    confirmed_at: float | None = None
    active_alert_id: str | None = None
    last_snap: CongestionResult | None = None
    severe_count: int = 0
    severe_first_at: float | None = None


class CongestionAlertHandler:
    """Evaluate CongestionResult snapshots; emit Alerts via AlertRepository."""

    def __init__(self, repository: AlertRepository,
                 policy: CongestionAlertPolicy | None = None) -> None:
        self.repository = repository
        self.policy = policy or CongestionAlertPolicy()
        self._memory: dict[str, _SegMem] = {}
        self._seq = 0

    def process_snapshots(self, snapshots: tuple[CongestionResult, ...],
                          *, now: float | None = None) -> tuple[Alert, ...]:
        """Evaluate a batch of CongestionResult snapshots (one per segment).
        Returns all Alert objects created / updated / cleared this call."""
        ts = now if now is not None else time.time()
        out: list[Alert] = []
        for snap in snapshots:
            out.extend(self._eval(snap, ts))
        return tuple(out)

    # ------------------------------------------------------------------ internal

    @staticmethod
    def _seg_id(snap: CongestionResult) -> str:
        return f"{snap.source_camera}->{snap.target_camera}"

    def _mem(self, seg_id: str) -> _SegMem:
        if seg_id not in self._memory:
            self._memory[seg_id] = _SegMem()
        return self._memory[seg_id]

    def _active_type(self, mem: _SegMem) -> AlertType | None:
        if mem.active_alert_id is None:
            return None
        a = self.repository.get(mem.active_alert_id)
        return a.alert_type if a else None

    def _eval(self, snap: CongestionResult, ts: float) -> list[Alert]:
        seg_id = self._seg_id(snap)
        mem = self._mem(seg_id)
        mem.last_snap = snap
        lv = snap.classification
        out: list[Alert] = []

        is_severe = lv is CongestionLevel.SEVERE
        is_cong   = lv is CongestionLevel.CONGESTED
        is_light  = lv in (CongestionLevel.FREE_FLOW, CongestionLevel.MODERATE)
        is_insuf  = lv is CongestionLevel.INSUFFICIENT_DATA

        # INSUFFICIENT_DATA: preserve state, do nothing
        if is_insuf:
            return out

        if is_light:
            mem.recovery_count += 1
            mem.pending_count = 0
            mem.first_qualifying_at = None
            mem.severe_count = 0
            mem.severe_first_at = None
        else:
            mem.recovery_count = 0

        # ---- recovery path (active alert exists)
        if mem.phase in (_Phase.CONGESTION_ACTIVE, _Phase.SEVERE_ACTIVE, _Phase.RECOVERY_PENDING):
            if is_light:
                mem.phase = _Phase.RECOVERY_PENDING
                if mem.recovery_count >= self.policy.recovery_count:
                    a = self._clear(seg_id, mem, ts)
                    if a:
                        out.append(a)
                    mem.phase = _Phase.NORMAL
                    mem.recovery_count = 0
                return out
            else:
                # alerting level while active -> reset recovery, stay active
                act_type = self._active_type(mem)
                mem.phase = (_Phase.SEVERE_ACTIVE if act_type is AlertType.SEVERE_DELAY
                             else _Phase.CONGESTION_ACTIVE)

        # ---- phase transitions
        ph = mem.phase

        if ph is _Phase.NORMAL:
            if is_severe or is_cong:
                mem.phase = _Phase.SEVERE_PENDING if is_severe else _Phase.CONGESTION_PENDING
                mem.pending_count = 1
                mem.first_qualifying_at = ts

        elif ph is _Phase.CONGESTION_PENDING:
            if is_cong:
                mem.pending_count += 1
                if mem.pending_count >= self.policy.congestion_confirm_count:
                    mem.confirmed_at = ts
                    out.append(self._raise(AlertType.CONGESTION, AlertSeverity.HIGH,
                                           seg_id, snap, mem, ts))
                    mem.phase = _Phase.CONGESTION_ACTIVE
            elif is_severe:
                mem.phase = _Phase.SEVERE_PENDING
                mem.pending_count = 1
                mem.first_qualifying_at = ts
            else:
                mem.phase = _Phase.NORMAL
                mem.pending_count = 0
                mem.first_qualifying_at = None

        elif ph is _Phase.SEVERE_PENDING:
            if is_severe:
                mem.pending_count += 1
                if mem.pending_count >= self.policy.severe_confirm_count:
                    mem.confirmed_at = ts
                    out.append(self._raise(AlertType.SEVERE_DELAY, AlertSeverity.CRITICAL,
                                           seg_id, snap, mem, ts))
                    mem.phase = _Phase.SEVERE_ACTIVE
            elif is_cong:
                mem.phase = _Phase.CONGESTION_PENDING
                mem.pending_count = 1
                mem.first_qualifying_at = ts
            else:
                mem.phase = _Phase.NORMAL
                mem.pending_count = 0
                mem.first_qualifying_at = None

        elif ph is _Phase.CONGESTION_ACTIVE:
            if is_severe:
                mem.severe_count += 1
                if mem.severe_count == 1: mem.severe_first_at = ts
                if mem.severe_count >= self.policy.severe_confirm_count:
                    first_severe_at=mem.severe_first_at
                    sup = self._clear(seg_id, mem, ts)
                    if sup: out.append(sup)
                    mem.first_qualifying_at = first_severe_at or ts
                    mem.severe_count = 0; mem.severe_first_at = None
                    mem.confirmed_at = ts
                    out.append(self._raise(AlertType.SEVERE_DELAY, AlertSeverity.CRITICAL,
                                           seg_id, snap, mem, ts))
                    mem.phase = _Phase.SEVERE_ACTIVE
            elif is_cong:
                mem.severe_count = 0; mem.severe_first_at = None
                a = self._update(mem, snap, ts)
                if a:
                    out.append(a)

        elif ph is _Phase.SEVERE_ACTIVE and (is_severe or is_cong):
            a = self._update(mem, snap, ts)
            if a:
                out.append(a)

        return out

    # ---- alert CRUD ---------------------------------------------------------

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"ALT-{prefix}-{self._seq:05d}"

    def _build_meta(self, snap: CongestionResult, mem: _SegMem,
                    alert_type: AlertType) -> dict:
        codes = (["SEVERE_DELAY_CONFIRMED"] if alert_type is AlertType.SEVERE_DELAY
                 else ["CONGESTION_CONFIRMED"])
        return {
            "segment_id": self._seg_id(snap),
            "origin_camera": snap.source_camera,
            "destination_camera": snap.target_camera,
            "congestion_state": snap.classification.value,
            "delay_ratio": snap.travel_time_ratio,
            "observed_travel_time_s": snap.observed_median_travel_time_s,
            "baseline_travel_time_s": snap.baseline_travel_time_s,
            "sample_count": snap.sample_count,
            "first_qualifying_at": mem.first_qualifying_at,
            "confirmed_at": mem.confirmed_at,
            "cleared_at": None,
            "reason_codes": codes,
            "rule_version": RULE_VERSION,
        }

    def _raise(self, alert_type: AlertType, severity: AlertSeverity,
               seg_id: str, snap: CongestionResult, mem: _SegMem,
               ts: float) -> Alert:
        prefix = "CG" if alert_type is AlertType.CONGESTION else "SD"
        meta = self._build_meta(snap, mem, alert_type)
        alert = Alert(
            self._next_id(prefix), alert_type, severity, AlertStatus.ACTIVE,
            "SEGMENT", seg_id, None,
            None, None, None, None,
            mem.first_qualifying_at or ts, ts, None, 1.0,
            meta,
        )
        self.repository.save(alert)
        mem.active_alert_id = alert.alert_id
        return alert

    def _update(self, mem: _SegMem, snap: CongestionResult, ts: float) -> Alert | None:
        if mem.active_alert_id is None:
            return None
        current = self.repository.get(mem.active_alert_id)
        if current is None:
            return None
        meta = self._build_meta(snap, mem, current.alert_type)
        updated = replace(current, status=AlertStatus.UPDATED, last_updated_at=ts, metadata=meta)
        self.repository.save(updated)
        return updated

    def _clear(self, seg_id: str, mem: _SegMem, ts: float) -> Alert | None:
        if mem.active_alert_id is None:
            return None
        current = self.repository.get(mem.active_alert_id)
        if current is None:
            mem.active_alert_id = None
            return None
        meta = dict(current.metadata)
        meta["cleared_at"] = ts
        cleared = replace(current, status=AlertStatus.CLEARED,
                          last_updated_at=ts, cleared_at=ts, metadata=meta)
        self.repository.save(cleared)
        mem.active_alert_id = None
        mem.confirmed_at = None
        mem.first_qualifying_at = None
        mem.pending_count = 0
        mem.severe_count = 0
        mem.severe_first_at = None
        return cleared
