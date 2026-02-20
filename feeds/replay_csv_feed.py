from __future__ import annotations

import csv
import time
from pathlib import Path
from threading import Event, Thread
from typing import Optional

from feeds.base import FeedPlugin


class ReplayCSVFeedPlugin(FeedPlugin):
    def __init__(self, bus, csv_path: str | Path, *, speed: float = 0.0, loop: bool = False) -> None:
        super().__init__(bus)
        self.csv_path = Path(csv_path)
        self.speed = max(0.0, float(speed))
        self.loop = loop
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while self._running and not self._stop_event.is_set():
            prev_ts: Optional[float] = None
            with self.csv_path.open('r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self._stop_event.is_set():
                        return
                    ts = float(row.get('timestamp') or time.time())
                    if self.speed > 0 and prev_ts is not None:
                        dt = max(0.0, (ts - prev_ts) / self.speed)
                        if dt > 0:
                            time.sleep(min(dt, 0.2))
                    prev_ts = ts
                    self.bus.publish('tick', {
                        'ts': ts,
                        'sym': str(row['symbol']),
                        'price': float(row['price']),
                        'source': 'replay_csv',
                    })
            if not self.loop:
                break
        self._running = False
