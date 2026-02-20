from __future__ import annotations

from abc import ABC, abstractmethod

from core.event_bus import EventBus


class FeedPlugin(ABC):
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._running = False

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
