"""
risk_manager.py — position sizing, slippage, drawdown caps, paper wallet.

Two execution styles supported:
1) "Scalper" (taker-ish): discrete positions per market (PaperPosition).
2) "Market maker" (OpenClaw-style): inventory ledger + resting quotes (PaperOrder).

Design goals:
- Default to PAPER trading.
- Single-source of truth for equity, inventory, open positions, and trade history.
- Tail-risk aware sizing (power-law / alpha-stable proxy via Hill estimator).
"""
from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import Settings
from utils import log_returns, hill_tail_index, clamp

log = logging.getLogger("risk")


# ──────────────────────────────────────────────────────────────────────────────
# Paper: discrete positions (baseline scalper)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class PaperPosition:
    market_id: str
    token_id: str
    symbol: str
    direction: str              # "above" / "below"
    strike: float
    expiry_ts: int

    entry_price: float          # YES price (0..1)
    shares: float               # number of shares
    usdc_spent: float
    fees_usdc: float
    opened_ts: int

    status: str = "OPEN"
    exit_ts: int = 0
    exit_price: float = 0.0
    pnl_usdc: float = 0.0

    def pnl_pct(self) -> float:
        denom = max(self.usdc_spent, 1e-9)
        return self.pnl_usdc / denom


# ──────────────────────────────────────────────────────────────────────────────
# Paper: market-maker orders + inventory ledger
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class PaperOrder:
    order_id: str
    token_id: str
    side: str            # BUY / SELL
    price: float
    size_shares: float
    created_ts: int


@dataclass(slots=True)
class PaperFill:
    order_id: str
    token_id: str
    side: str
    price: float
    size_shares: float
    fee_usdc: float
    ts: int


class PaperWallet:
    def __init__(self, start_usdc: float = 1000.0):
        self.start_usdc = float(start_usdc)
        self.balance_usdc = float(start_usdc)

        # scalper positions
        self.positions: Dict[str, PaperPosition] = {}
        self.closed: List[PaperPosition] = []

        # maker inventory (per token_id)
        self.inv_shares: Dict[str, float] = {}
        self.inv_avg_cost: Dict[str, float] = {}
        self.realized_pnl_usdc: float = 0.0

        # maker open orders + fills
        self.open_orders: Dict[str, PaperOrder] = {}
        self.fills: List[PaperFill] = []

    # -------- Equity / inventory --------

    def inventory_value_usdc(self, token_id: str, mid_price: float) -> float:
        sh = float(self.inv_shares.get(token_id, 0.0))
        return sh * float(mid_price)

    def equity(self, mark_prices: Dict[str, float]) -> float:
        """
        mark_prices is a dict token_id -> mid price (0..1).
        For the scalper portion, we approximate mark as current mid (if available).
        """
        eq = float(self.balance_usdc)
        # maker inventory
        for tid, sh in self.inv_shares.items():
            eq += float(sh) * float(mark_prices.get(tid, 0.0))
        # scalper open positions
        for pos in self.positions.values():
            eq += float(pos.shares) * float(mark_prices.get(pos.token_id, pos.entry_price))
        return eq

    # -------- Maker orders --------

    def place_order(self, *, token_id: str, side: str, price: float, size_shares: float, ts: int) -> PaperOrder:
        oid = "paper_" + secrets.token_hex(8)
        o = PaperOrder(order_id=oid, token_id=str(token_id), side=str(side).upper(), price=float(price), size_shares=float(size_shares), created_ts=int(ts))
        self.open_orders[oid] = o
        return o

    def cancel_order(self, order_id: str) -> None:
        self.open_orders.pop(str(order_id), None)

    def cancel_all_token(self, token_id: str) -> None:
        token_id = str(token_id)
        for oid in list(self.open_orders.keys()):
            if self.open_orders[oid].token_id == token_id:
                self.open_orders.pop(oid, None)

    def _apply_fill(self, fill: PaperFill) -> None:
        """
        Apply fill to cash + inventory + realized PnL.
        Fee model: fee is deducted from cash.
        """
        self.balance_usdc -= float(fill.fee_usdc)

        tid = fill.token_id
        sh = float(fill.size_shares)
        px = float(fill.price)
        side = fill.side.upper()

        if side == "BUY":
            cost = sh * px
            self.balance_usdc -= cost
            prev_sh = float(self.inv_shares.get(tid, 0.0))
            prev_cost = float(self.inv_avg_cost.get(tid, 0.0))
            new_sh = prev_sh + sh
            # weighted avg cost
            if new_sh > 1e-12:
                new_cost = (prev_sh * prev_cost + sh * px) / new_sh
            else:
                new_cost = 0.0
            self.inv_shares[tid] = new_sh
            self.inv_avg_cost[tid] = new_cost
        elif side == "SELL":
            proceeds = sh * px
            self.balance_usdc += proceeds
            prev_sh = float(self.inv_shares.get(tid, 0.0))
            prev_cost = float(self.inv_avg_cost.get(tid, 0.0))
            sold = min(sh, prev_sh)
            self.inv_shares[tid] = prev_sh - sold
            # realized pnl on sold shares
            self.realized_pnl_usdc += sold * (px - prev_cost)
            if self.inv_shares[tid] <= 1e-12:
                self.inv_shares.pop(tid, None)
                self.inv_avg_cost.pop(tid, None)


    def apply_external_fill(
        self,
        *,
        order_id: str,
        token_id: str,
        side: str,
        price: float,
        size_shares: float,
        fee_rate_bps: int = 0,
        ts: int,
    ) -> None:
        """
        Apply a fill coming from a live venue/user-stream into the paper ledger.

        This keeps RiskManager sizing & drawdown logic coherent even in LIVE mode.
        NOTE: This does *not* attempt perfect reconciliation (partial fills etc.);
        it is a conservative accounting update.
        """
        oid = str(order_id)
        tid = str(token_id)
        side = str(side).upper()
        px = float(price)
        sh = float(size_shares)
        fee = float(sh * px) * (float(fee_rate_bps) / 10000.0)

        fill = PaperFill(order_id=oid, token_id=tid, side=side, price=px, size_shares=sh, fee_usdc=fee, ts=int(ts))
        self.fills.append(fill)
        self._apply_fill(fill)
        # drop from open orders if present
        self.open_orders.pop(oid, None)

    def simulate_mm_fills(
        self,
        *,
        best_bid_ask: Dict[str, Tuple[float, float]],
        ts: int,
        maker_fee_bps: int = 0,
        base_fill_prob: float = 0.08,
    ) -> List[PaperFill]:
        """
        Lightweight paper fill simulator:
        - If your BUY is at/above current best_bid, it might get hit.
        - If your SELL is at/below current best_ask, it might get lifted.
        This is NOT a microstructure simulator; it just makes paper trading interactive.

        Returns fills applied.
        """
        fills: List[PaperFill] = []
        for oid, o in list(self.open_orders.items()):
            bb_ba = best_bid_ask.get(o.token_id)
            if not bb_ba:
                continue
            bb, ba = bb_ba
            side = o.side.upper()
            p_fill = base_fill_prob

            hit = False
            if side == "BUY":
                # if we're close to top of book, allow fills
                if o.price >= bb - 1e-9:
                    hit = secrets.randbelow(10_000) < int(p_fill * 10_000)
            else:
                if o.price <= ba + 1e-9:
                    hit = secrets.randbelow(10_000) < int(p_fill * 10_000)

            if hit:
                fee = (maker_fee_bps / 10_000.0) * (o.size_shares * o.price)
                f = PaperFill(order_id=oid, token_id=o.token_id, side=side, price=o.price, size_shares=o.size_shares, fee_usdc=float(fee), ts=int(ts))
                fills.append(f)
                self.fills.append(f)
                self.open_orders.pop(oid, None)
                self._apply_fill(f)

        return fills

    # -------- Scalper position lifecycle (unchanged) --------

    def open_position(self, pos: PaperPosition) -> None:
        self.balance_usdc -= float(pos.usdc_spent + pos.fees_usdc)
        self.positions[pos.market_id] = pos

    def close_position(self, market_id: str, *, exit_price: float, exit_ts: int, fees_usdc: float) -> Optional[PaperPosition]:
        pos = self.positions.get(market_id)
        if not pos:
            return None
        pos.exit_price = float(exit_price)
        pos.exit_ts = int(exit_ts)
        gross = pos.shares * pos.exit_price
        pos.pnl_usdc = gross - pos.usdc_spent - float(fees_usdc)
        pos.fees_usdc += float(fees_usdc)
        pos.status = "CLOSED"
        self.balance_usdc += gross - float(fees_usdc)
        self.positions.pop(market_id, None)
        self.closed.append(pos)
        return pos


# ──────────────────────────────────────────────────────────────────────────────
# Risk manager
# ──────────────────────────────────────────────────────────────────────────────

class RiskManager:
    def __init__(self, settings: Settings, wallet: PaperWallet):
        self.s = settings
        self.w = wallet
        self._peak_eq: float = wallet.start_usdc

    def update_drawdown(self, equity: float) -> float:
        self._peak_eq = max(self._peak_eq, float(equity))
        dd = (self._peak_eq - float(equity)) / max(self._peak_eq, 1e-9)
        return float(dd)

    def allowed_to_trade(self, equity: float) -> bool:
        dd = self.update_drawdown(equity)
        if dd >= self.s.DRAWDOWN_CAP:
            log.error("Drawdown cap hit: %.2f%% >= %.2f%%. Trading halted.", dd * 100, self.s.DRAWDOWN_CAP * 100)
            return False
        return True

    def tail_risk_multiplier(self, price_series: Sequence[float]) -> float:
        """
        If returns exhibit very fat tails (low alpha), reduce risk.
        """
        r = log_returns(price_series[-800:])
        if r.size < 120:
            return 1.0
        alpha_hat = float(hill_tail_index(r, k=min(80, max(20, r.size // 10))))
        if alpha_hat < self.s.LEVY_ALPHA_MIN:
            # sharper reduction when heavier tails
            mult = float(clamp(alpha_hat / max(self.s.LEVY_ALPHA_MIN, 1e-9), 0.2, 1.0))
            return mult
        return 1.0

    def size_usdc(self, *, equity: float, price_series: Sequence[float]) -> float:
        """
        Baseline scalper position size in USDC, using:
        - 1% risk cap
        - max per trade cap
        - fat-tail multiplier
        """
        cap = float(min(self.s.MAX_USDC_PER_TRADE, equity * self.s.RISK_PER_TRADE))
        mult = self.tail_risk_multiplier(price_series)
        return float(max(0.0, cap * mult))

    def cap_mm_size_shares(
        self,
        *,
        equity: float,
        side: str,
        token_id: str,
        price: float,
        size_shares: float,
        current_inventory_usdc: float,
    ) -> float:
        """
        Cap market-maker quote size with:
        - per-order notional cap
        - global risk cap (equity * RISK_PER_TRADE)
        - soft inventory cap
        """
        price = float(max(price, 1e-6))
        side = side.upper()
        size_shares = float(max(0.0, size_shares))

        notional_cap = float(min(self.s.MM_MAX_USDC_PER_ORDER, equity * self.s.RISK_PER_TRADE))
        size_shares = min(size_shares, notional_cap / price)

        # inventory cap: prevent quoting deeper into inventory beyond cap
        if side == "BUY":
            if current_inventory_usdc >= self.s.MM_MAX_INV_USDC:
                return 0.0
            room = self.s.MM_MAX_INV_USDC - current_inventory_usdc
            size_shares = min(size_shares, room / price)
        elif side == "SELL":
            if current_inventory_usdc <= -self.s.MM_MAX_INV_USDC:
                return 0.0
            room = self.s.MM_MAX_INV_USDC + current_inventory_usdc  # how much we can sell before crossing -cap
            size_shares = min(size_shares, max(0.0, room) / price)

        return float(max(0.0, size_shares))
