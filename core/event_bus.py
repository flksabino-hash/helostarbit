from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Any, Callable, DefaultDict


class EventBus:
    def __init__(self) -> None:
        self._subs: DefaultDict[str, list[Callable[[dict[str, Any]], None]]] = defaultdict(list)
        self._lock = RLock()

    def subscribe(self, event_type: str, callback: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            self._subs[event_type].append(callback)

    def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            callbacks = list(self._subs.get(event_type, []))
        for cb in callbacks:
            cb(payload)
