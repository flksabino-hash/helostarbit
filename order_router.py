"""
order_router.py — single choke-point for ALL live order actions.

Security goals:
- Strategies emit intents; only OrderRouter can talk to live adapters.
- Centralized risk gating + dedupe + venue allowlist hooks.
- Avoids auto-retrying non-idempotent calls (order placement/cancel).

This is intentionally conservative: if a state-changing request outcome is ambiguous,
the router marks itself "needs_reconcile" and the engine should reconcile via open orders
and/or user stream.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import Settings
from risk_manager import PaperWallet, RiskManager
from utils import utc_ts, tick_round, clamp

log = logging.getLogger("router")


@dataclass(slots=True)
class OrderIntent:
    token_id: str
    side: str                   # BUY / SELL
    price: float                # 0..1
    size_shares: float
    tick_size: float
    neg_risk: bool = False
    order_type: str = "GTC"
    post_only: bool = True      # enforced locally by non-cross checks


class OrderRouter:
    def __init__(self, s: Settings, wallet: PaperWallet, risk: RiskManager):
        self.s = s
        self.wallet = wallet
        self.risk = risk
        self.needs_reconcile: bool = False

    # ----- Safety / gating -----

    def _enforce_post_only_local(self, intent: OrderIntent, *, best_bid: float, best_ask: float) -> OrderIntent:
        """
        Batch endpoint doesn't document postOnly, so we enforce "non-marketable" locally.
        If intent would cross, we back it off by one tick (or drop size if impossible).
        """
        px = float(intent.price)
        tick = float(max(intent.tick_size, 1e-9))
        side = intent.side.upper()

        if not intent.post_only:
            return intent

        if side == "BUY":
            # must be strictly below best ask
            limit = float(best_ask) - tick
            if limit <= 0:
                return replace(intent, size_shares=0.0)
            px = min(px, limit)
            px = tick_round(px, tick, direction="down")
        else:
            # must be strictly above best bid
            limit = float(best_bid) + tick
            if limit >= 1.0:
                return replace(intent, size_shares=0.0)
            px = max(px, limit)
            px = tick_round(px, tick, direction="up")

        px = float(clamp(px, 0.0001, 0.9999))
        return replace(intent, price=float(px))

    def cap_size(self, intent: OrderIntent, *, equity: float, inventory_usdc: float) -> OrderIntent:
        """
        Apply RiskManager caps + no-naked-short for SELL.
        """
        side = intent.side.upper()
        px = float(max(intent.price, 1e-6))
        size = float(max(0.0, intent.size_shares))

        size = self.risk.cap_mm_size_shares(
            equity=equity,
            side=side,
            token_id=intent.token_id,
            price=px,
            size_shares=size,
            current_inventory_usdc=inventory_usdc,
        )

        if side == "SELL":
            have = float(self.wallet.inv_shares.get(intent.token_id, 0.0))
            size = min(size, have)

        return replace(intent, size_shares=float(max(0.0, size)))

    # ----- Execution primitives -----

    async def cancel_many(self, trader: Any, order_ids: Sequence[str], *, live: bool) -> None:
        ids = [str(x) for x in order_ids if x]
        if not ids:
            return
        if not live:
            for oid in ids:
                self.wallet.cancel_order(oid)
            return

        # Live: prefer batch cancel endpoint
        try:
            await trader.cancel_orders(ids)
        except Exception as e:
            # ambiguous cancels are less dangerous than ambiguous posts,
            # but we still flag for reconcile.
            self.needs_reconcile = True
            log.warning("cancel_many failed (needs reconcile): %s", e)

    async def place_many(self, trader: Any, intents: Sequence[OrderIntent], *, live: bool) -> List[str]:
        intents = [i for i in intents if i.size_shares > 0]
        if not intents:
            return []

        if not live:
            ids: List[str] = []
            for i in intents:
                o = self.wallet.place_order(
                    token_id=i.token_id,
                    side=i.side,
                    price=i.price,
                    size_shares=i.size_shares,
                    ts=utc_ts(),
                )
                ids.append(o.order_id)
            return ids

        # Live: never auto-retry order placement (non-idempotent)
        ids: List[str] = []
        try:
            if self.s.MM_USE_BATCH:
                batch: List[Dict[str, Any]] = []
                for i in intents:
                    signed = trader.sign_limit_order(
                        token_id=i.token_id,
                        side=i.side,
                        price=i.price,
                        size_shares=i.size_shares,
                        expiration_ts=0,
                        fee_rate_bps=int(self.s.MAKER_FEE_BPS),
                        neg_risk=bool(i.neg_risk),
                    )
                    batch.append(
                        {
                            "order": signed,
                            "owner": trader.api_key,
                            "orderType": str(i.order_type).upper(),
                            "deferExec": False,
                        }
                    )
                res = await trader.post_orders(batch)
                for r in res:
                    if r.get("success"):
                        oid = str(r.get("orderID") or "")
                        ids.append(oid)
                    else:
                        ids.append("")
            else:
                for i in intents:
                    signed = trader.sign_limit_order(
                        token_id=i.token_id,
                        side=i.side,
                        price=i.price,
                        size_shares=i.size_shares,
                        expiration_ts=0,
                        fee_rate_bps=int(self.s.MAKER_FEE_BPS),
                        neg_risk=bool(i.neg_risk),
                    )
                    r = await trader.post_order(
                        signed,
                        order_type=str(i.order_type).upper(),
                        post_only=bool(i.post_only),
                        neg_risk=bool(i.neg_risk),
                    )
                    ids.append(str(r.get("orderID") or ""))
            return ids
        except Exception as e:
            self.needs_reconcile = True
            log.error("place_many failed (needs reconcile): %s", e)
            return ["" for _ in intents]
