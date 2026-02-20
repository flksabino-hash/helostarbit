from __future__ import annotations

import time
from threading import Event, Thread
from typing import Optional

import requests

from feeds.base import FeedPlugin


class PolymarketGammaCLOBFeedPlugin(FeedPlugin):
    GAMMA_BASE = 'https://gamma-api.polymarket.com'
    CLOB_BASE = 'https://clob.polymarket.com'

    def __init__(self, bus, market_slug: str, *, interval_s: float = 2.0) -> None:
        super().__init__(bus)
        self.market_slug = market_slug
        self.interval_s = max(0.5, float(interval_s))
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._session = requests.Session()
        self._yes_token_id: Optional[str] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _load_yes_token(self) -> Optional[str]:
        r = self._session.get(f"{self.GAMMA_BASE}/markets/slug/{self.market_slug}", timeout=10)
        r.raise_for_status()
        data = r.json()
        for tok in data.get('tokens', []):
            if str(tok.get('outcome', '')).lower() == 'yes':
                return str(tok.get('token_id'))
        return None

    def _mid(self, token_id: str) -> Optional[float]:
        r = self._session.get(f"{self.CLOB_BASE}/midpoint", params={'token_id': token_id}, timeout=10)
        r.raise_for_status()
        m = r.json().get('mid')
        return None if m is None else float(m)

    def _run(self) -> None:
        try:
            self._yes_token_id = self._load_yes_token()
            if not self._yes_token_id:
                raise RuntimeError('YES token not found')
        except Exception as exc:
            self.bus.publish('feed_error', {'source': 'polymarket', 'error': str(exc)})
            self._running = False
            return
        while self._running and not self._stop.is_set():
            try:
                mid = self._mid(self._yes_token_id)
                if mid is not None:
                    self.bus.publish('poly_tick', {
                        'ts': time.time(),
                        'slug': self.market_slug,
                        'token_id': self._yes_token_id,
                        'mid': mid,
                        'source': 'polymarket_gamma_clob',
                    })
            except Exception as exc:
                self.bus.publish('feed_error', {'source': 'polymarket', 'error': str(exc)})
            self._stop.wait(self.interval_s)
        self._running = False
