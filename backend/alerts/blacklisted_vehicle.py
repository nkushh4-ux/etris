import logging
import queue
import threading

from backend.alerts.models import Alert, AlertSeverity, AlertStatus, AlertType
from cv.ocr.models import FusionStatus


class BlacklistedVehicleAlertHandler:
    rule_version="BLACKLIST-1.0"
    def __init__(self,alerts,watchlist): self.alerts=alerts; self.watchlist=watchlist; self._sequence=0
    def handle_completed(self,completed):
        fused=completed.fused_plate_result
        if fused.status is not FusionStatus.ACCEPTED or not fused.plate_text: return ()
        created=[]
        for entry in self.watchlist.find_active_exact(fused.plate_text,completed.timestamp):
            key=(completed.camera_id,entry.watchlist_id,completed.track_id,AlertType.BLACKLISTED_VEHICLE)
            if self.alerts.get_active_by_key(key) is not None: continue
            self._sequence+=1
            severity=AlertSeverity(entry.severity.value)
            metadata={"plate_text":fused.plate_text,"plate_confidence":fused.confidence,
                "vehicle_class":None,"watchlist_id":entry.watchlist_id,"watchlist_reason":entry.reason,
                "watchlist_severity":entry.severity.value,"watchlist_source":entry.source,
                "demo_watchlist_entry":entry.source.upper().startswith("DEMO"),
                "detected_at":completed.timestamp.isoformat(),"rule_version":self.rule_version,
                "reason_codes":["ACCEPTED_ANPR_EXACT_WATCHLIST_MATCH"]}
            alert=Alert(f"ALT-BL-{self._sequence:05d}",AlertType.BLACKLISTED_VEHICLE,severity,
                AlertStatus.ACTIVE,completed.camera_id,entry.watchlist_id,completed.track_id,None,None,
                fused.plate_text,"ACCEPTED",completed.timestamp.timestamp(),completed.timestamp.timestamp(),None,
                fused.confidence,metadata)
            self.alerts.save(alert); created.append(alert)
        return tuple(created)


class BlacklistAlertIngestor:
    def __init__(self,handler,max_pending=64):
        self.handler=handler; self._queue=queue.Queue(maxsize=max_pending); self._stop=threading.Event()
        self._thread=threading.Thread(target=self._run,name="etris-blacklist-alert-writer",daemon=True); self._thread.start()
    def submit(self,completed):
        try: self._queue.put_nowait(completed); return True
        except queue.Full: return False
    def _run(self):
        while not self._stop.is_set():
            try: item=self._queue.get(timeout=.25)
            except queue.Empty: continue
            try: self.handler.handle_completed(item)
            except Exception: logging.getLogger(__name__).exception("Blacklist alert processing failed")
            finally: self._queue.task_done()
    def stop(self): self._stop.set(); self._thread.join(timeout=2)
