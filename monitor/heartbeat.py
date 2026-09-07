"""Heartbeat writer: proves the trader is alive and records state."""

from __future__ import annotations

import json
import time
from datetime import datetime

from config import HEARTBEAT_FILE


class Heartbeat:
    def __init__(self, db) -> None:
        self.db = db

    def tick(self, state: str, message: str = "ok") -> dict:
        payload = {
            "ts": time.time(),
            "iso": datetime.now().isoformat(timespec="seconds"),
            "state": state,
            "message": message,
        }
        with open(HEARTBEAT_FILE, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
        self.db.save_heartbeat(state, message)
        return payload
