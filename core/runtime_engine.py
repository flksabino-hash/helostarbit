from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from config import Settings, load_settings
from core.econo_engine import EconoEngine
from core.event_bus import EventBus
from core.venue_selector import VenueSelector
from infra.structured_logging import get_logger
from risk_manager import PaperWallet, RiskManager


@dataclass
class RuntimeState:
    last_prices: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, dict[str, float | str]] = field(default_factory=dict)
    last_trade_ts: dict[str, float] = field(default_factory=dict)


class RuntimeEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings(None)
        self.log = get_logger('runtime_engine')
        self.bus = EventBus()
        self.econo = EconoEngine()
        self.wallet = PaperWallet(start_usdc=self.settings.PAPER_START_USDC)
        self.risk = RiskManager(self.settings, self.wallet)
        self.selector = VenueSelector()
        self.state = RuntimeState()
        self.bus.subscribe('tick', self._on_tick)
        self.bus.subscribe('poly_tick', self._on_poly_tick)
        self.bus.subscribe('feed_error', self._on_feed_error)

    def _on_tick(self, payload: dict[str, Any]) -> None:
        self._handle_price(str(payload['sym']), float(payload['price']), str(payload.get('source', 'unknown')))

    def _on_poly_tick(self, payload: dict[str, Any]) -> None:
        self._handle_price(str(payload['slug']), float(payload['mid']), 'polymarket')

    def _on_feed_error(self, payload: dict[str, Any]) -> None:
        self.log.warning('feed_error', **payload)

    def _handle_price(self, sym: str, px: float, source: str) -> None:
        self.econo.update(sym, px)
        self.state.last_prices[sym] = px
        h = self.econo.hurst(sym)
        e = self.econo.entropy(sym)
        l = self.econo.lle(sym)
        regime = self.econo.amh_regime(sym)
        self.state.metrics[sym] = {'hurst': h, 'entropy': e, 'lle': l, 'regime': regime}
        self.log.info('tick', sym=sym, px=px, src=source, hurst=h, entropy=e, lle=l, regime=regime)

        # Paper execution is optional and remains outside UI
        if len(self.econo.history[sym]) < 80:
            return
        metric_gate = (h > self.settings.HURST_MIN and e < self.settings.ENTROPY_MAX and l < self.settings.LLE_MAX)
        if not metric_gate and len(self.wallet.fills) > 0:
            return
        now = time.time()
        if now - self.state.last_trade_ts.get(sym, 0.0) < self.settings.COOLDOWN_SEC:
            return

        venue = self.selector.choose(funding=0.0, oi_delta=0.1, volatility=abs(px - 0.5), margin_health=1.0, liquidity=1.0)
        self.log.info('venue_choice', sym=sym, venue=venue.name, score=venue.score)
        if venue.name != 'paper':
            return

        marks = dict(self.state.last_prices)
        equity = self.wallet.equity(marks)
        if not self.risk.allowed_to_trade(equity):
            return
        size_usdc = self.risk.size_usdc(equity=equity, price_series=list(self.econo.history[sym]))
        qty = max(0.0, size_usdc / max(px, 1e-6))
        if qty <= 0:
            return

        self.wallet.apply_external_fill(order_id=f'auto-{int(now)}-{sym}', token_id=sym, side='BUY', price=px, size_shares=qty, ts=int(now))
        self.state.last_trade_ts[sym] = now
        self.log.info('paper_fill', sym=sym, side='BUY', qty=qty, px=px)
